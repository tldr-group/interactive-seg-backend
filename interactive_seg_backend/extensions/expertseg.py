"""
Code adapted from 'expertsegmentation' (MIT):
DOI: https://doi.org/10.1016/j.actamat.2025.120993
github: https://github.com/NREL/expertsegmentation/tree/main
authors: Nina Prakash, Paul Gasper, Francois Usseglio-Viretta

Allows users to add domain-knowledge inspired losses to their XGBoost trainable segmentation.
"""

from typing import Any
from interactive_seg_backend.configs import ClassInfo
from interactive_seg_backend.classifiers import XGBCPU


# want:
# - multiple objs
# - allow to define objs only for certain classes (via class info)


class ExpertSegClassifier(XGBCPU):
    def __init__(self, class_infos: list[ClassInfo], extra_args: dict[str, Any]) -> None:
        super().__init__(extra_args)
