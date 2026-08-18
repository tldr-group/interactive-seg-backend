import numpy as np
from PIL import Image
from skimage.measure._regionprops import RegionProperties
from skimage.measure import label, regionprops

import platform
from os import path, makedirs

from typing import Literal, TYPE_CHECKING

from interactive_seg_backend.utils import logger
from interactive_seg_backend.configs import NPUIntArray, NPFloatArray, ClassInfo

sam_imported = True
try:
    from onnxruntime import InferenceSession
    from requests import get as rget
except ImportError:
    logger.warning("ONNX SAM unavailable!")
    sam_imported = False
SAM_AVAILABLE = sam_imported

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from requests import get as rget


def to_onnx_image(img: Image.Image | np.ndarray) -> NPFloatArray:
    if isinstance(img, Image.Image):  # cast
        image_np = np.array(img.convert("RGB"))  # in case it was L
    else:
        image_np = img

    if len(image_np.shape) == 2:  # grayscale
        image_np = np.stack([image_np] * 3, axis=-1)  # convert to RGB

    if np.amax(image_np) > 1:  # norm
        image_np = np.array(image_np, np.float32) / 255.0

    image_np = np.transpose(image_np, (2, 0, 1))  # convert to CHW format
    if len(image_np.shape) == 3:  # add batch dimension
        image_np = np.expand_dims(image_np, 0)
    return image_np


def to_onnx_point_prompt(points: list[tuple[int, int]], labels: list[int]) -> tuple[NPFloatArray, NPFloatArray]:
    """Map point list to ONNX format.

    Args:
        points (list[tuple[int, int]]): (x, y) for each point
        labels (list[int]): +1 for positive, 0 for negative

    Returns:
        tuple[FloatArr, FloatArr]: (B, N_query, N_points, 2), (B, N_query, N_points)
    """
    prompt_points_arr = np.array(points, dtype=np.float32)
    prompt_points_arr = prompt_points_arr[None, None, :, :]  # Add batch and n_query dimensions
    prompt_labels_arr = np.array(labels, dtype=np.float32)
    prompt_labels_arr = prompt_labels_arr[None, None, :]  # Add batch dimension
    return prompt_points_arr, prompt_labels_arr


def to_onnx_box_prompt(boxes: list[tuple[int, int, int, int]]) -> tuple[NPFloatArray, NPFloatArray]:
    """Map bboxes to 2D points and labels for ONNX model input.

    Args:
        boxes (list[tuple[int, int, int, int]]): (x, y, h, w) for each box

    Returns:
        tuple[FloatArr, FloatArr]: (B, N_query, 2 * N_boxes, 2), (B, N_query, 2 * N_boxes)
    """
    n_points = 2 * len(boxes)
    box_points_arr = np.zeros((1, 1, n_points, 2), np.float32)
    box_labels_arr = np.zeros((1, 1, n_points), np.float32)
    for x0, y0, w, h in boxes:
        box_points_arr[0, 0, :, 0] = [x0, x0 + w]
        box_points_arr[0, 0, :, 1] = [y0, y0 + h]

    for i in range(0, n_points, 2):
        # efficientSAM ONNX uses class label 2 for left edge and 3 for right edge
        box_labels_arr[0, 0, i] = 2.0
        box_labels_arr[0, 0, i + 1] = 3.0

    return box_points_arr, box_labels_arr


def _get_default_cache_path_for_platform() -> str:
    """Get default cache path for the current platform."""
    if platform.system() == "Windows":
        return path.join(path.expanduser("~"), "AppData", "Local", "isb", "models")
    elif platform.system() == "Darwin":  # macOS
        return path.join(path.expanduser("~"), "Library", "Caches", "isb", "models")
    else:  # Linux and other Unix-like systems
        return path.join(path.expanduser("~"), ".cache", "isb", "models")


def _maybe_make_cache_dir(cache_dir: str) -> None:
    """Create the cache directory if it doesn't exist."""
    if not path.exists(cache_dir):
        makedirs(cache_dir, exist_ok=True)
        logger.info(f"Creating ISB cache directory at {cache_dir}")
    else:
        pass


def _download_from_hf_with_requests(filename: str, output_path: str) -> None:
    logger.info(f"Downloading {filename} to {output_path} from huggingface")
    repo_id = "yunyangx/EfficientSAM"
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    response = rget(url, stream=True)
    # check for HTTP errors (e.g., 401 Unauthorized or 404 Not Found)
    response.raise_for_status()
    # write the file in chunks
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    logger.info("Downloaded complete!")


def download_sam_onnx_models(output_dir: str | None):
    if output_dir is None:
        output_dir = _get_default_cache_path_for_platform()
    _maybe_make_cache_dir(output_dir)
    for which in ["encoder", "decoder"]:
        model_name = f"efficientsam_ti_{which}.onnx"
        model_path = path.join(output_dir, model_name)
        _download_from_hf_with_requests(model_name, model_path)


def load_or_download_model(checkpoint: str | None, which: Literal["encoder", "decoder"]) -> InferenceSession:
    if checkpoint is not None:
        models_exist = path.exists(checkpoint)
        if models_exist:
            logger.info(f"Loading {which} model from {checkpoint}")
            return InferenceSession(checkpoint)
        else:
            raise FileNotFoundError(f"Specified checkpoint {checkpoint} does not exist.")

    cache_dir_path = _get_default_cache_path_for_platform()
    cached_model_path = path.join(cache_dir_path, f"efficientsam_ti_{which}.onnx")
    cached_model_exists = path.exists(cached_model_path)
    if cached_model_exists:
        logger.info(f"Loading {which} model from cache at {cached_model_path}")
        return InferenceSession(cached_model_path)
    else:
        download_sam_onnx_models(cache_dir_path)
        return InferenceSession(cached_model_path)


class SAMEncoderONNX:
    def __init__(self, checkpoint: str | None) -> None:
        self.session = load_or_download_model(checkpoint, "encoder")

    def get_embedding(self, img: Image.Image | np.ndarray) -> NPFloatArray:
        img_onnx = to_onnx_image(img)
        embed = self.session.run(None, {"batched_images": img_onnx})
        return embed  # type: ignore


class SAMDecoderONNX:
    def __init__(self, checkpoint: str | None) -> None:
        self.session = load_or_download_model(checkpoint, "decoder")

    def _run_model(
        self,
        embedding: NPFloatArray,
        img_size: tuple[int, int],
        prompts: NPFloatArray,
        labels: NPFloatArray,
    ) -> tuple[NPFloatArray, NPFloatArray]:
        "ES ONNX uses shared interface for point and box prompts so use helper function"
        img_size_onnx = np.array(img_size, dtype=np.int64)
        predicted_logits: NPFloatArray
        predicted_iou: NPFloatArray
        predicted_logits, predicted_iou, _ = self.session.run(
            None,
            {
                "image_embeddings": embedding[0],
                "batched_point_coords": prompts,
                "batched_point_labels": labels,
                "orig_im_size": img_size_onnx,
            },
        )  # type: ignore
        scores = predicted_iou[0, 0]
        return predicted_logits, scores

    def _process_results(
        self,
        predicted_logits: NPFloatArray,
        scores: NPFloatArray,
        threshold: bool,
        threshold_val: float,
        multimask_output: bool,
    ) -> tuple[NPFloatArray, NPFloatArray]:
        if threshold:
            predicted_logits = (predicted_logits > threshold_val).astype(np.float32)
        if multimask_output:
            logger.info(f"esam: {predicted_logits.shape} masks with {scores}")
            return predicted_logits, scores

        best_mask_idx = np.argmax(scores)
        filtered_mask = predicted_logits[0, 0, best_mask_idx]
        filtered_score = scores[best_mask_idx]
        logger.info(f"esam: {filtered_mask.shape} masks with {filtered_score:.3f}")
        return predicted_logits[0, 0, best_mask_idx], filtered_score

    def masks_from_points(
        self,
        embedding: NPFloatArray,
        img_size: tuple[int, int],
        point_prompts: list[tuple[int, int]],
        point_labels: list[int] | None,
        threshold: bool = True,
        threshold_val: float = 0.0,
        multimask_output: bool = True,
    ) -> tuple[NPFloatArray, NPFloatArray]:
        if point_labels is None:  # assume +ve if not supplied
            point_labels = [1 for _ in point_prompts]

        point_prompts_onnx, point_labels_onnx = to_onnx_point_prompt(point_prompts, point_labels)
        predicted_logits, scores = self._run_model(embedding, img_size, point_prompts_onnx, point_labels_onnx)
        return self._process_results(predicted_logits, scores, threshold, threshold_val, multimask_output)

    def masks_from_boxes(
        self,
        embedding: NPFloatArray,
        img_size: tuple[int, int],
        boxes: list[tuple[int, int, int, int]],
        threshold: bool = True,
        threshold_val: float = 0.0,
        multimask_output: bool = True,
    ) -> tuple[NPFloatArray, NPFloatArray]:
        box_prompts_onnx, box_labels_onnx = to_onnx_box_prompt(boxes)
        predicted_logits, scores = self._run_model(embedding, img_size, box_prompts_onnx, box_labels_onnx)
        return self._process_results(predicted_logits, scores, threshold, threshold_val, multimask_output)


MIN_SAM_AREA_PX = 225


def do_sam_postproc(
    seg: NPUIntArray,
    img_arr: np.ndarray,
    classes_to_process: list[ClassInfo],
    cached_embedding: np.ndarray | None = None,
    cached_regions: list[RegionProperties] | None = None,
) -> np.ndarray:
    """Perform SAM post-processing. Given a set of classes to perform SAM filtering on,
    split into regions via `label()` and `regionprop()`, and for each region of that class
    of the correct size (> ClassInfo.min_size_px or MIN_SAM_AREA_PX ), prompt SAM with the
    bounding box that region. For the resulting mask, paste it over the original segmentation.

    Args:
        seg (NPUIntArray): (H,W) array of class values
        img_arr (np.ndarray): (H,W,[C]) array of image values
        classes_to_process (list[ClassInfo]): list of classinfos to process
        cached_embedding (np.ndarray | None, optional): existing SAM embedding for image. Defaults to None.
        cached_regions (list[RegionProperties] | None, optional): existing regionprops for image. Defaults to None.

    Returns:
        np.ndarray: (H,W) processed segmentation array
    """
    out_seg = seg.copy()
    encoder = SAMEncoderONNX(None)
    decoder = SAMDecoderONNX(None)

    if cached_embedding is not None:
        embed = cached_embedding
    else:
        embed = encoder.get_embedding(img_arr)

    if cached_regions is not None:
        regions = cached_regions
    else:
        regions = regionprops(label(seg))

    regions_per_class: dict[int, list[RegionProperties]] = {class_info.value: [] for class_info in classes_to_process}
    for region in regions:
        region_seg_val: int = int(seg[region.coords[0][0], region.coords[0][1]])
        if region_seg_val in regions_per_class:
            regions_per_class[region_seg_val].append(region)

    for class_info in classes_to_process:
        if not class_info.do_sam_postproc:
            continue

        # class and size filtering - don't want to run SAM on single pixel regions
        min_size = class_info.min_size_px if class_info.min_size_px is not None else MIN_SAM_AREA_PX
        class_regions = regions_per_class[class_info.value]
        class_regions_matching_size = [r for r in class_regions if r.area > min_size]

        logger.info(f"SAM post-proc for class {class_info.name} on {len(class_regions_matching_size)} regions")

        for region in class_regions_matching_size:
            minr, minc, maxr, maxc = region.bbox
            bbox = (minc, minr, maxc - minc, maxr - minr)  # (x, y, w, h)
            # where SAM predicts, fill it with the class value
            box_mask, _ = decoder.masks_from_boxes(embed, img_arr.shape[:2], [bbox], multimask_output=False)
            # NB ESAM mask can be greater than bbox, so use full extent
            out_seg[box_mask > 0] = class_info.value

    return out_seg
