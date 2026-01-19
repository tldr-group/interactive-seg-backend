import numpy as np

from interactive_seg_backend.configs import CRFParams
from interactive_seg_backend.utils import to_rgb_arr, logger

from typing import TYPE_CHECKING, Any

crf_imported = True
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    KERNEL = dcrf.FULL_KERNEL
except ImportError:
    logger.warning("CRF unavailable!")
    crf_imported = False
CRF_AVAILABLE = crf_imported

if TYPE_CHECKING:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels


default_crf_params = CRFParams()


def _get_crf(img_arr: np.ndarray, n_c: int, unary: np.ndarray, crf: CRFParams) -> "dcrf.DenseCRF2D":
    h, w, _ = img_arr.shape
    d: Any = dcrf.DenseCRF2D(w, h, n_c)
    u = np.ascontiguousarray(unary)
    d.setUnaryEnergy(u)
    d.addPairwiseGaussian(
        sxy=crf.sxy_g,
        compat=crf.compat_g,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    d.addPairwiseBilateral(
        sxy=crf.sxy_b,
        srgb=crf.s_rgb,
        rgbim=img_arr,
        compat=crf.compat_b,
        kernel=KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )
    return d


def do_crf_from_labels(labels_arr: np.ndarray, img_arr: np.ndarray, n_classes: int, crf: CRFParams) -> np.ndarray:
    """Given a multiclass (foreground) segmentation and orignal image arr,
    refine using a conditional random field with set parameters.

    :param labels_arr: arr shape (h, w) where each entry is class
    :type labels_arr: np.ndarray
    :param img_arr: img arr shape (h, w, 3)
    :type img_arr: np.ndarray
    :param n_classes: number of classes
    :type n_classes: int
    :param crf: parameters for CRF
    :type crf: CRFParams
    :return: refined segmentation, shape (h, w, 1)
    :rtype: np.ndarray
    """
    h, w, _ = img_arr.shape
    unary = unary_from_labels(labels_arr, n_classes, crf.label_confidence, zero_unsure=False)
    d: Any = _get_crf(img_arr, n_classes, unary, crf)
    Q = d.inference(crf.n_infer)
    crf_seg = np.argmax(Q, axis=0)
    crf_seg = crf_seg.reshape((h, w, 1))
    return crf_seg


def do_crf_from_probabilites(probs: np.ndarray, img_arr: np.ndarray, n_classes: int, crf: CRFParams) -> np.ndarray:
    if CRF_AVAILABLE is False:
        logger.warning("CRF unavailable, skipping CRF refinement")
        return probs
    logger.info(f"CRF: probs {probs.shape}, img {img_arr.shape}, params: {crf}")

    if len(img_arr.shape) == 2:
        img_arr = to_rgb_arr(img_arr)
    h, w, c = img_arr.shape
    probs = probs.astype(np.float32)
    probs = -np.log(probs)
    unary = np.ascontiguousarray(probs.reshape(h * w, n_classes).T)
    d = _get_crf(img_arr, n_classes, unary, crf)
    Q = d.inference(crf.n_infer)
    crf_seg = np.argmax(Q, axis=0)
    crf_seg = crf_seg.reshape((h, w, 1))
    refined = crf_seg.squeeze(-1)
    return refined
