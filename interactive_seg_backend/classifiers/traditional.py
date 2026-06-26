import numpy as np
from typing import Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_multiotsu
import logging

from interactive_seg_backend.configs import NPFloatArray, NPUIntArray
from .base import Classifier

logger = logging.getLogger(__name__)


class Otsu(Classifier):
    """Otsu thresholding classifier with multiclass support."""

    def __init__(self, extra_args: dict[str, Any]) -> None:
        super().__init__(extra_args)
        self.feature_idx = extra_args.pop("feature_idx", 0)
        self.kwargs = extra_args.copy()
        self.classes: NPUIntArray | None = None
        self.thresholds: np.ndarray | None = None
        self.bin_to_class: dict[int, int] | None = None
        self.class_to_right_edge: dict[int, float] | None = None

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ) -> "Otsu":
        """Fits multiotsu thresholds on the training data and maps bins to classes."""
        self.classes = np.unique(target_data)
        K = len(self.classes)

        # We perform thresholding on the specified 1D feature
        if train_data.ndim > 1 and train_data.shape[1] > 1:
            x = train_data[:, self.feature_idx]
        else:
            x = train_data.ravel()

        if K <= 1:
            self.thresholds = np.array([], dtype=np.float64)
            self.bin_to_class = {0: self.classes[0]} if K == 1 else {}
            self.class_to_right_edge = {self.classes[0]: np.inf} if K == 1 else {}
            return self

        # Determine thresholds
        try:
            thresholds = threshold_multiotsu(x, classes_num=K)
            if isinstance(thresholds, (int, float, np.integer, np.floating)):
                self.thresholds = np.array([thresholds], dtype=np.float64)
            else:
                self.thresholds = np.sort(np.asarray(thresholds, dtype=np.float64))
        except Exception as e:
            logger.warning(f"threshold_multiotsu failed: {e}. Falling back to linspace thresholds.")
            xmin, xmax = x.min(), x.max()
            if xmin == xmax:
                self.thresholds = np.array([xmin] * (K - 1), dtype=np.float64)
            else:
                self.thresholds = np.linspace(xmin, xmax, K + 1)[1:-1]

        # Map classes to bins using Hungarian matching (maximum overlap)
        bins = np.digitize(x, self.thresholds)
        overlap = np.zeros((K, K), dtype=np.int64)
        for b in range(K):
            for j, c in enumerate(self.classes):
                overlap[b, j] = np.sum((bins == b) & (target_data == c))

        row_ind, col_ind = linear_sum_assignment(-overlap)

        self.bin_to_class = {}
        for b, c_idx in zip(row_ind, col_ind):
            self.bin_to_class[b] = self.classes[c_idx]

        # Store class val -> threshold right edge
        right_edges = list(self.thresholds) + [np.inf]
        self.class_to_right_edge = {}
        for b in range(K):
            if b in self.bin_to_class:
                self.class_to_right_edge[self.bin_to_class[b]] = right_edges[b]

        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        """Applies thresholds to the data and returns one-hot probabilities."""
        if self.classes is None or self.thresholds is None or self.bin_to_class is None:
            raise ValueError("Otsu classifier must be fit before calling predict_proba")

        if features_flat.ndim > 1 and features_flat.shape[1] > 1:
            x = features_flat[:, self.feature_idx]
        else:
            x = features_flat.ravel()

        bins_test = np.digitize(x, self.thresholds)
        n_samples = len(x)
        probs = np.zeros((n_samples, len(self.classes)), dtype=np.float32)

        for b in range(len(self.classes)):
            if b in self.bin_to_class:
                class_val = self.bin_to_class[b]
                class_idx = np.where(self.classes == class_val)[0][0]
                probs[bins_test == b, class_idx] = 1.0

        return probs


class KMeansClassifier(Classifier):
    """K-Means clustering classifier."""

    def __init__(self, extra_args: dict[str, Any]) -> None:
        super().__init__(extra_args)
        self.scale = extra_args.pop("scale", False)
        self.kwargs = extra_args.copy()
        self.classes: NPUIntArray | None = None
        self.scaler: StandardScaler | None = None
        self.model: KMeans | None = None
        self.cluster_to_class: dict[int, int] | None = None
        self.class_to_centroid: dict[int, np.ndarray] | None = None

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ) -> "KMeansClassifier":
        """Fits KMeans on training data and maps clusters to classes."""
        self.classes = np.unique(target_data)
        K = len(self.classes)

        if self.scale:
            self.scaler = StandardScaler()
            train_data_scaled = self.scaler.fit_transform(train_data)
        else:
            train_data_scaled = train_data

        # Fit KMeans
        self.model = KMeans(n_clusters=K, **self.kwargs)
        cluster_labels = self.model.fit_predict(train_data_scaled)

        # Map cluster labels to class values using Hungarian matching (maximum overlap)
        overlap = np.zeros((K, K), dtype=np.int64)
        for cluster_idx in range(K):
            for j, c in enumerate(self.classes):
                overlap[cluster_idx, j] = np.sum((cluster_labels == cluster_idx) & (target_data == c))

        row_ind, col_ind = linear_sum_assignment(-overlap)

        self.cluster_to_class = {}
        for r, c in zip(row_ind, col_ind):
            self.cluster_to_class[r] = self.classes[c]

        # Store a mapping of class val -> centroids using maximum overlap
        self.class_to_centroid = {}
        for cluster_idx in range(K):
            if cluster_idx in self.cluster_to_class:
                class_val = self.cluster_to_class[cluster_idx]
                self.class_to_centroid[class_val] = self.model.cluster_centers_[cluster_idx]

        return self

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        """Predicts probabilities by assigning samples to their cluster-mapped classes."""
        if self.model is None or self.classes is None or self.cluster_to_class is None:
            raise ValueError("KMeansClassifier must be fit before calling predict_proba")

        if self.scale and self.scaler is not None:
            features_scaled = self.scaler.transform(features_flat)
        else:
            features_scaled = features_flat

        cluster_preds = self.model.predict(features_scaled)
        n_samples = len(features_flat)
        probs = np.zeros((n_samples, len(self.classes)), dtype=np.float32)

        for cluster_idx in range(len(self.classes)):
            if cluster_idx in self.cluster_to_class:
                class_val = self.cluster_to_class[cluster_idx]
                class_idx = np.where(self.classes == class_val)[0][0]
                probs[cluster_preds == cluster_idx, class_idx] = 1.0

        return probs
