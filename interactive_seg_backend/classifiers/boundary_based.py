from abc import abstractmethod
from typing import Any
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

from interactive_seg_backend.configs import NPFloatArray, NPUIntArray
from .base import Classifier
from .region_utils import merge_regions, assign_classes_by_overlap

logger = logging.getLogger(__name__)


class BoundaryBasedClassifier(Classifier):
    """Base class for boundary-based classifiers (SRM, Seeded Region Growing, Watershed)."""

    def __init__(self, extra_args: dict[str, Any]) -> None:
        super().__init__(extra_args)
        self.scale = extra_args.pop("scale", False)
        self.kwargs = extra_args.copy()
        self.classes: NPUIntArray | None = None
        self.scaler: StandardScaler | None = None
        # full_features and full_labels can be set dynamically before/during fit and predict_proba
        self.full_features: NPFloatArray | None = None
        self.full_labels: NPUIntArray | None = None
        self.region_map_: NPUIntArray | None = None
        self.labels_: NPUIntArray | None = None

    def fit(
        self,
        train_data: NPFloatArray,
        target_data: NPUIntArray,
        sample_weights: NPFloatArray | None = None,
    ) -> "BoundaryBasedClassifier":
        """Fits the scaler on the training data, stores unique classes, and builds/stores the region map."""
        self.classes = np.unique(target_data)
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)

        if self.full_features is None:
            raise ValueError("Boundary-based classifiers require full_features to be set before calling fit().")
        if self.full_labels is None:
            raise ValueError("Boundary-based classifiers require full_labels to be set before calling fit().")

        # Cast features to float32 for compatibility and precision
        full_feats_32 = self.full_features.astype(np.float32)
        h, w, c = full_feats_32.shape

        # Scale features if needed
        if self.scale and self.scaler is not None:
            flat_scaled = self.scaler.transform(full_feats_32.reshape(-1, c))
            features_scaled = flat_scaled.reshape(h, w, c).astype(np.float32)
        else:
            features_scaled = full_feats_32

        # Create initial region map
        self.region_map_ = self._create_region_map(features_scaled, self.full_labels)
        self.labels_ = self.full_labels.copy()

        return self

    @abstractmethod
    def _create_region_map(self, features: NPFloatArray, labels: NPUIntArray) -> NPUIntArray:
        """Create the initial fine-grained region map.

        To be implemented by subclasses.
        """
        pass

    def predict_proba(self, features_flat: NPFloatArray) -> NPFloatArray:
        """Apply the boundary-based segmentation to the full features."""
        if self.classes is None or self.region_map_ is None or self.labels_ is None:
            raise ValueError("Classifier must be fit before calling predict_proba")
        if self.full_features is None:
            raise ValueError(
                "Boundary-based classifiers require full_features to be set on the model. Please pass labels/image to apply()."
            )

        # Cast features to float32 for compatibility and precision
        full_feats_32 = self.full_features.astype(np.float32)
        h, w, c = full_feats_32.shape

        # Scale features if needed
        if self.scale and self.scaler is not None:
            flat_scaled = self.scaler.transform(full_feats_32.reshape(-1, c))
            features_scaled = flat_scaled.reshape(h, w, c).astype(np.float32)
        else:
            features_scaled = full_feats_32

        # Merge regions using the stored region map and the prediction-time features_scaled
        n_classes = len(self.classes)
        merged_map = merge_regions(self.region_map_, features_scaled, n_classes)

        # Assign classes based on overlap with the stored training labels
        pred_2D = assign_classes_by_overlap(merged_map, self.labels_, self.classes)

        # Convert 2D prediction to probability map of shape (H * W, K)
        n_pixels = h * w
        pred_flat = pred_2D.ravel()
        probs = np.zeros((n_pixels, len(self.classes)), dtype=np.float32)
        for idx, c_val in enumerate(self.classes):
            probs[pred_flat == c_val, idx] = 1.0

        return probs


class SeededRegionGrowing(BoundaryBasedClassifier):
    """Seeded Region Growing classifier."""

    def _create_region_map(self, features: NPFloatArray, labels: NPUIntArray) -> NPUIntArray:
        import heapq
        from scipy.ndimage import label as label_components, distance_transform_edt

        h, w, c = features.shape
        # Label seed connected components
        seeds_labeled, num_features = label_components(labels > 0)

        # If no seeds are found, return a single region
        if num_features == 0:
            return np.ones((h, w), dtype=np.int32)

        region_map = seeds_labeled.copy().astype(np.int32)

        # Compute mean feature vector for each seed component
        comp_means = {}
        for comp_id in range(1, num_features + 1):
            mask = seeds_labeled == comp_id
            comp_means[comp_id] = np.mean(features[mask], axis=0)

        pq = []
        visited = seeds_labeled > 0

        dy = [-1, 1, 0, 0]
        dx = [0, 0, -1, 1]

        seed_y, seed_x = np.nonzero(seeds_labeled)
        for y, x in zip(seed_y, seed_x):
            comp_id = seeds_labeled[y, x]
            for i in range(4):
                ny, nx = y + dy[i], x + dx[i]
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    dist = np.linalg.norm(features[ny, nx] - comp_means[comp_id])
                    threshold = self.kwargs.get("threshold", None)
                    if threshold is None or dist <= threshold:
                        heapq.heappush(pq, (dist, ny, nx, comp_id))

        while pq:
            dist, y, x, comp_id = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            region_map[y, x] = comp_id

            for i in range(4):
                ny, nx = y + dy[i], x + dx[i]
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    ndist = np.linalg.norm(features[ny, nx] - comp_means[comp_id])
                    threshold = self.kwargs.get("threshold", None)
                    if threshold is None or ndist <= threshold:
                        heapq.heappush(pq, (ndist, ny, nx, comp_id))

        unassigned = region_map == 0
        if np.any(unassigned):
            _, indices = distance_transform_edt(unassigned, return_indices=True)
            region_map[unassigned] = region_map[indices[0], indices[1]][unassigned]

        return region_map


class StatisticalRegionMerging(BoundaryBasedClassifier):
    """Statistical Region Merging classifier."""

    def _create_region_map(self, features: NPFloatArray, labels: NPUIntArray) -> NPUIntArray:
        Q = self.kwargs.get("Q", 32.0)

        H, W, C = features.shape
        n_pixels = H * W

        flat_feats = features.reshape(n_pixels, C)

        # Dynamically determine g as the maximum range across all features
        g = self.kwargs.get("g", None)
        if g is None:
            g = float(np.max(np.ptp(flat_feats, axis=0)))
            if g == 0.0:
                g = 1.0

        # Grid edges (4-connectivity)
        y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W - 1), indexing="ij")
        u_h = y_idx * W + x_idx
        v_h = y_idx * W + (x_idx + 1)

        y_idx_v, x_idx_v = np.meshgrid(np.arange(H - 1), np.arange(W), indexing="ij")
        u_v = y_idx_v * W + x_idx_v
        v_v = (y_idx_v + 1) * W + x_idx_v

        u = np.concatenate([u_h.ravel(), u_v.ravel()])
        v = np.concatenate([v_h.ravel(), v_v.ravel()])

        diffs = flat_feats[u] - flat_feats[v]
        dists = np.sqrt(np.sum(diffs**2, axis=1))

        sort_idx = np.argsort(dists)
        u_sorted = u[sort_idx]
        v_sorted = v[sort_idx]

        parent = np.arange(n_pixels, dtype=np.int32)
        pixel_count = np.ones(n_pixels, dtype=np.int32)
        sum_feats = flat_feats.copy()

        def find(i: int) -> int:
            root = i
            while parent[root] != root:
                root = parent[root]
            curr = i
            while curr != root:
                nxt = parent[curr]
                parent[curr] = root
                curr = nxt
            return root

        delta = 1.0 / (6.0 * n_pixels)
        C_const = (g**2) * np.log(2.0 / delta) / (2.0 * Q)

        for idx in range(len(u_sorted)):
            p1 = u_sorted[idx]
            p2 = v_sorted[idx]

            r1 = find(p1)
            r2 = find(p2)

            if r1 == r2:
                continue

            c1 = pixel_count[r1]
            c2 = pixel_count[r2]

            mean1 = sum_feats[r1] / c1
            mean2 = sum_feats[r2] / c2

            diff_sq = np.sum((mean1 - mean2) ** 2)
            thresh_sq = C_const * (1.0 / c1 + 1.0 / c2)

            if diff_sq <= thresh_sq:
                parent[r2] = r1
                pixel_count[r1] += c2
                sum_feats[r1] += sum_feats[r2]

        region_map = np.zeros(n_pixels, dtype=np.int32)
        for i in range(n_pixels):
            region_map[i] = find(i)

        return region_map.reshape(H, W)


class WatershedClassifier(BoundaryBasedClassifier):
    """Watershed classifier."""

    def _create_region_map(self, features: NPFloatArray, labels: NPUIntArray) -> NPUIntArray:
        import skimage.segmentation
        from interactive_seg_backend.features.multiscale_classical_cpu import singlescale_edges

        use_gradient = self.kwargs.get("use_gradient", False)

        if use_gradient:
            if features.shape[-1] > 1:
                grad = singlescale_edges(features[:, :, 0])
            else:
                grad = singlescale_edges(features[:, :, 0])
            watershed_img = grad
        else:
            watershed_img = features[:, :, 0]

        # Cast to float32 to avoid local_maxima/local_minima float16 issues
        watershed_img = watershed_img.astype(np.float32)

        region_map = skimage.segmentation.watershed(watershed_img, markers=None)
        return region_map.astype(np.int32)
