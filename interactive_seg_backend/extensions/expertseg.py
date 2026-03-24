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

from typing import Any
from interactive_seg_backend.configs import ClassInfo, Arr, UInt8Arr, NPFloatArray
from interactive_seg_backend.classifiers import XGBCPU


# want:
# - multiple objs
# - allow to define objs only for certain classes (via class info)


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

        self.params = extra_args

        self.do_vf_loss = any([(info.desired_volume_fraction is not None) for info in class_infos])
        self.do_conn_loss = any([(info.connectivity_target is not None) for info in class_infos])

        assert self.do_vf_loss or self.do_conn_loss, (
            "At least one custom loss must be enabled inside `ClassInfo`s of `TrainingConfig`"
        )

        self.ref_img_feats: None | NPFloatArray = None

    def fit(
        self,
        train_data: Arr,
        target_data: UInt8Arr,
        sample_weights: Arr | None = None,
    ):
        label_mask = np.nonzero(target_data)
        labels = target_data[label_mask[0]]
        labels_onehot = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))

        full_img_dmat = DMatrix(data=train_data)
        train_dmat = DMatrix(data=train_data[label_mask[0], :], label=labels - 1)

        model = Booster(self.params, [train_dmat])

        n_classes = labels_onehot.shape[-1]
        targ_vfs = _get_target_vf_dist(n_classes, self.class_infos)

        for i in range(self.n_epochs):
            full_img_pred = self.model.predict(full_img_dmat)
            train_pred = full_img_pred[label_mask[0]]

            g_softmax, h_softmax = softmax_obj(train_pred, labels_onehot)
            g, h = g_softmax, h_softmax

            if self.do_vf_loss:
                g_vf_full, h_vf_full, _ = volume_fraction_obj(full_img_pred, self.lambd_vf, targ_vfs)
                g_vf, h_vf = g_vf_full[label_mask[0]], h_vf_full[label_mask[0]]
                g += g_vf
                h += h_vf

            model.boost(train_dmat, i, g, h)

        self.model = model

        return self

    def predict_proba(self, features_flat: Arr) -> Arr:
        return self.model.predict(features_flat)

    # def predict(self, features: Arr) -> UInt8Arr:
    #     return super().predict(features)


# def softmax_obj(preds: np.ndarray, dtrain: DMatrix):
#     # Label is a vector of class indices for each input example
#     labels = dtrain.get_label()

#     # When objective=softprob, preds has shape (N, K). Convert the labels
#     # to one-hot encoding to match this shape.
#     labels = OneHotEncoder(sparse_output=False).fit_transform(labels.reshape(-1, 1))

#     grad = preds - labels
#     hess = 2.0 * preds * (1.0 - preds)

#     # XGBoost wants them to be returned as 1-d vectors
#     return grad.flatten(), hess.flatten()


def softmax_obj(preds: np.ndarray, labels_onehot: np.ndarray):
    grad = preds - labels_onehot
    hess = 2.0 * preds * (1.0 - preds)

    # XGBoost wants them to be returned as 1-d vectors
    return grad.flatten(), hess.flatten()


def _get_target_vf_dist(n_classes: int, class_infos: list[ClassInfo]) -> NPFloatArray:
    targ_vfs = np.zeros(n_classes)
    # NB: no guarantee each class has a ClassInfo
    for info in class_infos:
        phase_idx = info.value
        info_vf = info.desired_volume_fraction
        # again no guarantee each ClassInfo has a target vf
        targ_vf = 0 if info_vf is None else info_vf
        targ_vfs[phase_idx] = targ_vf

    return targ_vfs


def volume_fraction_obj(
    whole_img_pred: np.ndarray, lambd: float, target_vf_distr: np.ndarray
) -> tuple[NPFloatArray, NPFloatArray, int | float]:
    n_px = whole_img_pred.shape[0]
    pred_onehot = whole_img_pred / np.max(whole_img_pred, axis=0)
    # mask pred_vf where target_vf is not defined
    pred_vf_distr = np.mean(pred_onehot, axis=0) * np.where(target_vf_distr > 0)

    loss = lambd * np.linalg.norm(pred_vf_distr - target_vf_distr) ** 2  # for the whole image
    grad_row = 2 * lambd * (pred_vf_distr - target_vf_distr)

    grad = np.tile(grad_row, n_px)
    grad = np.array([grad_row] * len(whole_img_pred))
    return loss, grad.flatten(), 0.0
