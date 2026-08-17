from onnxruntime import InferenceSession
import numpy as np
from PIL import Image

import platform
from requests import get as rget
from os import path, makedirs

import numpy.typing as npt
from typing import TypeAlias, Literal

import logging

FloatArr: TypeAlias = npt.NDArray[np.floating]


def to_onnx_image(img: Image.Image | np.ndarray) -> FloatArr:
    if isinstance(img, Image.Image):  # cast
        image_np = np.array(img.convert("RGB"))  # in case it was L
    else:
        image_np = img

    if np.amax(image_np) > 1:  # norm
        image_np /= 255.0

    image_np = np.transpose(image_np, (2, 0, 1))  # convert to CHW format
    if len(image_np.shape) == 3:  # add batch dimension
        image_np = np.expand_dims(image_np, 0)
    return image_np


def to_onnx_point_prompt(points: list[tuple[int, int]], labels: list[int]) -> tuple[FloatArr, FloatArr]:
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


def to_onnx_box_prompt(boxes: list[tuple[int, int, int, int]]) -> tuple[FloatArr, FloatArr]:
    """Map bboxes to 2D points and labels for ONNX model input.

    Args:
        boxes (list[tuple[int, int, int, int]]): (x, y, h, w) for each box

    Returns:
        tuple[FloatArr, FloatArr]: (B, N_query, 2 * N_boxes, 2), (B, N_query, 2 * N_boxes)
    """
    n_points = 2 * len(boxes)
    box_points_arr = np.zeros((1, 1, n_points, 2), np.float32)
    box_labels_arr = np.zeros((1, 1, n_points), np.float32)
    for x0, y0, h, w in boxes:
        box_points_arr[0, 0, :, 0] = [x0, x0 + w, x0, x0 + w]
        box_points_arr[0, 0, :, 1] = [y0, y0, y0 + h, y0 + h]

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
        logging.info(f"Creating ISB cache directory at {cache_dir}")
    else:
        pass


def _download_from_hf_with_requests(filename: str, output_path: str) -> None:
    repo_id = "yunyangx/EfficientSAM"
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    response = rget(url, stream=True)
    # check for HTTP errors (e.g., 401 Unauthorized or 404 Not Found)
    response.raise_for_status()
    # write the file in chunks
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    logging.info(f"Downloaded {output_path}")


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
            logging.info(f"Loading {which} model from {checkpoint}")
            return InferenceSession(checkpoint)
        else:
            raise FileNotFoundError(f"Specified checkpoint {checkpoint} does not exist.")

    cache_dir_path = _get_default_cache_path_for_platform()
    cache_model_path = path.join(cache_dir_path, f"efficientsam_ti_{which}.onnx")
    cached_model_exists = path.exists(cache_model_path)
    if cached_model_exists:
        logging.info(f"Loading {which} model from cache at {cached_model_exists}")
        return InferenceSession(cache_model_path)
    else:
        download_sam_onnx_models(cache_dir_path)
        return InferenceSession(cache_model_path)


class SAMEncoderONNX:
    def __init__(self, checkpoint: str | None) -> None:
        self.session = load_or_download_model(checkpoint, "encoder")

    def get_embedding(self, img: Image.Image | np.ndarray) -> FloatArr:
        img_onnx = to_onnx_image(img)
        embed = self.session.run(None, {"batched_images": img_onnx})
        return embed  # type: ignore


class SAMDecoderONNX:
    def __init__(self, checkpoint: str | None) -> None:
        self.session = load_or_download_model(checkpoint, "decoder")

    def _run_model(
        self,
        embedding: FloatArr,
        img_size: tuple[int, int],
        prompts: FloatArr,
        labels: FloatArr,
    ) -> tuple[FloatArr, FloatArr]:
        "ES ONNX uses shared interface for point and box prompts so use helper function"
        img_size_onnx = np.array(img_size, dtype=np.float32)[None, :]
        predicted_logits: FloatArr
        predicted_iou: FloatArr
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
        predicted_logits: FloatArr,
        scores: FloatArr,
        threshold: bool,
        threshold_val: float,
        multimask_output: bool,
    ) -> tuple[FloatArr, FloatArr]:
        if threshold:
            predicted_logits = (predicted_logits > threshold_val).astype(np.float32)
        if multimask_output:
            return predicted_logits, scores

        best_mask_idx = np.argmax(scores)
        return predicted_logits[best_mask_idx : best_mask_idx + 1], scores[best_mask_idx : best_mask_idx + 1]

    def masks_from_points(
        self,
        embedding: FloatArr,
        img_size: tuple[int, int],
        point_prompts: list[tuple[int, int]],
        point_labels: list[int] | None,
        threshold: bool = True,
        threshold_val: float = 0.0,
        multimask_output: bool = True,
    ) -> tuple[FloatArr, FloatArr]:
        if point_labels is None:  # assume +ve if not supplied
            point_labels = [1 for _ in point_prompts]

        point_prompts_onnx, point_labels_onnx = to_onnx_point_prompt(point_prompts, point_labels)
        predicted_logits, scores = self._run_model(embedding, img_size, point_prompts_onnx, point_labels_onnx)
        return self._process_results(predicted_logits, scores, threshold, threshold_val, multimask_output)

    def masks_from_boxes(
        self,
        embedding: FloatArr,
        img_size: tuple[int, int],
        boxes: list[tuple[int, int, int, int]],
        threshold: bool = True,
        threshold_val: float = 0.0,
        multimask_output: bool = True,
    ) -> tuple[FloatArr, FloatArr]:
        box_prompts_onnx, box_labels_onnx = to_onnx_box_prompt(boxes)
        predicted_logits, scores = self._run_model(embedding, img_size, box_prompts_onnx, box_labels_onnx)
        return self._process_results(predicted_logits, scores, threshold, threshold_val, multimask_output)
