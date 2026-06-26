from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops


@dataclass
class RegionData:
    """Dataclass to hold intermediate region information for greedy merging."""

    label_id: int
    pixel_count: int
    mean_feature: np.ndarray  # shape (C,)
    member_ids: set[int]


def merge_regions(
    region_map: np.ndarray,
    features: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Greedily merges regions in region_map based on distance between their mean feature vectors

    until n_classes regions remain.

    Args:
        region_map (np.ndarray): (H, W) array of integer region labels.
        features (np.ndarray): (H, W, C) array of features.
        n_classes (int): The target number of merged regions to remain.

    Returns:
        np.ndarray: (H, W) array of merged region labels.
    """
    h, w, c = features.shape

    # Use regionprops to extract a set of regions from the label image (region_map)
    props = regionprops(region_map)

    # Build initial RegionData objects
    regions = []
    for prop in props:
        min_r, min_c, max_r, max_c = prop.bbox
        feat_sub = features[min_r:max_r, min_c:max_c]
        mask_sub = prop.image

        # Compute the mean of the features over that region
        pixels = feat_sub[mask_sub]
        if len(pixels) > 0:
            mean_feat = np.mean(pixels, axis=0)
        else:
            mean_feat = np.zeros(c)

        regions.append(
            RegionData(
                label_id=prop.label,
                pixel_count=prop.area,
                mean_feature=mean_feat,
                member_ids={prop.label},
            )
        )

    if len(regions) <= n_classes:
        return region_map

    # Optimization: if there are too many regions, keep only the largest 200,
    # and greedily merge the smaller regions into the closest large region in feature space.
    max_regions = 200
    if len(regions) > max_regions:
        # Sort by area descending
        regions = sorted(regions, key=lambda r: r.pixel_count, reverse=True)
        large_regions = regions[:max_regions]
        small_regions = regions[max_regions:]

        large_means = np.array([r.mean_feature for r in large_regions])
        for r in small_regions:
            # Find closest large region
            dists_to_large = np.linalg.norm(large_means - r.mean_feature, axis=1)
            best_idx = np.argmin(dists_to_large)
            best_large = large_regions[best_idx]

            # Merge
            new_count = best_large.pixel_count + r.pixel_count
            if new_count > 0:
                best_large.mean_feature = (
                    best_large.mean_feature * best_large.pixel_count
                    + r.mean_feature * r.pixel_count
                ) / new_count
            best_large.pixel_count = new_count
            best_large.member_ids.update(r.member_ids)

            # Update the large_means array for future steps (keep it updated)
            large_means[best_idx] = best_large.mean_feature

        regions = large_regions

    # Map to quickly lookup regions by ID
    regions_dict = {r.label_id: r for r in regions}
    active_ids = list(regions_dict.keys())

    # Precompute and maintain pairwise distances
    dists = {}
    for i in range(len(active_ids)):
        for j in range(i + 1, len(active_ids)):
            id1, id2 = active_ids[i], active_ids[j]
            dists[(id1, id2)] = np.linalg.norm(
                regions_dict[id1].mean_feature - regions_dict[id2].mean_feature
            )

    # Merge regions greedily based on feature distance
    while len(active_ids) > n_classes and dists:
        # Find the pair with minimum distance
        (id1, id2), min_dist = min(dists.items(), key=lambda item: item[1])

        r1 = regions_dict[id1]
        r2 = regions_dict[id2]

        # Merge r2 into r1
        new_count = r1.pixel_count + r2.pixel_count
        if new_count > 0:
            new_mean = (r1.mean_feature * r1.pixel_count + r2.mean_feature * r2.pixel_count) / new_count
        else:
            new_mean = r1.mean_feature

        r1.pixel_count = new_count
        r1.mean_feature = new_mean
        r1.member_ids.update(r2.member_ids)

        # Remove r2 from active list and dictionary
        del regions_dict[id2]
        active_ids.remove(id2)

        # Remove all distances involving id2
        to_remove = [k for k in dists.keys() if id2 in k]
        for k in to_remove:
            del dists[k]

        # Recalculate distances for id1
        for other_id in active_ids:
            if other_id != id1:
                key = (id1, other_id) if id1 < other_id else (other_id, id1)
                dists[key] = np.linalg.norm(
                    r1.mean_feature - regions_dict[other_id].mean_feature
                )

    # Reconstruct the merged map using a fast mapping array lookup
    max_label = int(np.max(region_map))
    mapping = np.arange(max_label + 1, dtype=np.int32)
    for r_id, r in regions_dict.items():
        for member_id in r.member_ids:
            if member_id <= max_label:
                mapping[member_id] = r.label_id

    merged_map = mapping[region_map]
    return merged_map.astype(np.int32)


def assign_classes_by_overlap(
    merged_map: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Assigns each region in merged_map to a class based on maximum overlap with labels

    using Hungarian bipartite matching.

    Args:
        merged_map (np.ndarray): (H, W) array of region labels.
        labels (np.ndarray): (H, W) array of original user labels (seeds).
        classes (np.ndarray): 1D array of unique non-zero classes present in labels.

    Returns:
        np.ndarray: (H, W) array containing assigned class values for each pixel.
    """
    unique_regions = np.unique(merged_map)
    n_regions = len(unique_regions)
    K = len(classes)

    overlap = np.zeros((n_regions, K), dtype=np.int64)
    for r_idx, r in enumerate(unique_regions):
        for c_idx, c in enumerate(classes):
            overlap[r_idx, c_idx] = np.sum((merged_map == r) & (labels == c))

    row_ind, col_ind = linear_sum_assignment(-overlap)

    region_to_class = {}
    for r_idx, c_idx in zip(row_ind, col_ind):
        region_to_class[unique_regions[r_idx]] = classes[c_idx]

    # Handle any unmatched regions (e.g. if n_regions > K)
    matched_regions = set(unique_regions[row_ind])
    for r in unique_regions:
        if r not in matched_regions:
            # Assign to class with max overlap, or default to first class
            class_overlaps = [np.sum((merged_map == r) & (labels == c)) for c in classes]
            best_c_idx = np.argmax(class_overlaps)
            region_to_class[r] = classes[best_c_idx]

    # Reconstruct the map with class labels
    pred = np.zeros_like(merged_map)
    for r, c in region_to_class.items():
        pred[merged_map == r] = c

    return pred
