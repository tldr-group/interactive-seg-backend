"""
Code adapted from 'expertsegmentation' (MIT):
DOI: https://doi.org/10.1016/j.actamat.2025.120993
github: https://github.com/NREL/expertsegmentation/tree/main
authors: Nina Prakash, Paul Gasper, Francois Usseglio-Viretta

Allows users to add domain-knowledge inspired losses to their XGBoost trainable segmentation.
"""

from xgboost import DMatrix, Booster
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from typing import Any, cast
from interactive_seg_backend.configs import ClassInfo, NPFloatArray, NPUIntArray
from interactive_seg_backend.classifiers import XGBCPU


class ExpertSegClassifier(XGBCPU):
    def __init__(
        self,
        class_infos: list[ClassInfo],
        n_epochs: int = 100,
        lambd_vf: float = 1.0,
        lambd_conn: float = 1.0,
        extra_args: dict[str, Any] = {},
    ) -> None:
        super().__init__(extra_args)

        self.model: Booster = Booster()

        self.class_infos = class_infos
        self.n_epochs = n_epochs
        self.lambd_vf = lambd_vf
        self.lambd_conn = lambd_conn
        # we need to do this manually
        extra_args["objective"] = "multi:softprob"

        self.params = extra_args

        self.do_vf_loss = any([(info.desired_volume_fraction is not None) for info in class_infos])
        self.do_conn_loss = any([(info.connectivity_target is not None) for info in class_infos])

        assert self.do_vf_loss or self.do_conn_loss, (
            "At least one custom loss must be enabled inside `ClassInfo`s of `TrainingConfig`"
        )

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ):
        """Fit XGBoost model to labels w.r.t custom losses defined in `ClassInfo`s of `TrainingConfig`. Currently supports volume fraction and connectivity losses,
        which can be enabled by setting `desired_volume_fraction` and `connectivity_target` fields of `ClassInfo`s respectively.

        Args:
            train_data (NPFloatArray): (h,w,c) array of features for each pixel in the image. This is different to the usual (n_samples, n_features) shape
                expected by sklearn-like APIs, as we need to be able to calculate global losses across the whole image.
            target_data (NPUIntArray): (h,w) array of integer labels for each pixel in the image. Again different to the usual (n_samples,) shape expected
                by sklearn-like APIs.
            sample_weights (NPFloatArray | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        ih, iw, c = train_data.shape
        lh, lw = target_data.shape
        assert (ih, iw) == (lh, lw), "Features and labels must be the same shape (i.e full image size)"

        train_data = train_data.reshape(-1, c)
        target_data = target_data.reshape(-1)

        label_mask = np.nonzero(target_data)
        labels = target_data[label_mask[0]]

        labels_onehot = OneHotEncoder(sparse_output=False).fit_transform(np.expand_dims(labels, -1))
        # Booster API expects DMatrices
        full_img_dmat = DMatrix(data=train_data)
        train_dmat = DMatrix(data=train_data[label_mask[0], :], label=labels - 1)

        n_classes = labels_onehot.shape[-1]
        targ_vfs = _get_target_vf_dist(n_classes, self.class_infos)

        self.params["num_class"] = n_classes
        model = Booster(self.params, [train_dmat])

        for i in range(self.n_epochs):
            full_img_pred = model.predict(full_img_dmat)
            train_pred = full_img_pred[label_mask[0]]
            # default loss
            g_softmax, h_softmax = softmax_obj(train_pred, labels_onehot)
            g, h = g_softmax, h_softmax
            # apply custom losses if enabled
            if self.do_vf_loss:
                _, g_vf_full = volume_fraction_obj(full_img_pred, self.lambd_vf, targ_vfs)
                g_vf = g_vf_full[label_mask[0]]
                g += g_vf

            model.boost(train_dmat, i, grad=g, hess=h)

        self.model = model
        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        dmat = DMatrix(data=features_flat)
        return self.model.predict(dmat)


def softmax_obj(preds: np.ndarray, labels_onehot: np.ndarray):
    grad = preds - labels_onehot
    hess = 2.0 * preds * (1.0 - preds)
    return grad, hess


def _get_target_vf_dist(n_classes: int, class_infos: list[ClassInfo]) -> NPFloatArray:
    # -1 = no target
    targ_vfs = np.zeros(n_classes) - 1
    # NB: no guarantee each class has a ClassInfo
    for info in class_infos:
        phase_idx = info.value
        info_vf = info.desired_volume_fraction
        # again no guarantee each ClassInfo has a target vf
        targ_vf = -1 if info_vf is None else info_vf
        targ_vfs[phase_idx] = targ_vf

    return targ_vfs


def volume_fraction_obj(
    whole_img_pred: np.ndarray, lambd: float, target_vf_distr: np.ndarray
) -> tuple[float, NPFloatArray]:
    n_px, n_classes = whole_img_pred.shape
    # (n_px, n_classes) binary segs which we can mean over to get predicted vfs
    pred_onehot = np.eye(n_classes)[np.argmax(whole_img_pred, axis=1)]
    pred_vf_distr = np.mean(pred_onehot, axis=0)
    # mask out classes with no target vf (i.e. where target_vf_distr is -1)
    valid_vf_mask = np.where(target_vf_distr >= 0, 1, 0)
    pred_vf_distr *= valid_vf_mask
    target_vf_distr *= valid_vf_mask

    loss = float(lambd * np.linalg.norm(pred_vf_distr - target_vf_distr) ** 2)  # global loss for the whole image
    # *global* difference in vf distribution broadcast to per pixel *class-wise* gradients
    grad_row = 2 * lambd * (pred_vf_distr - target_vf_distr)
    grad = np.tile(grad_row, (n_px, 1))
    grad = cast(NPFloatArray, grad)
    return loss, grad
