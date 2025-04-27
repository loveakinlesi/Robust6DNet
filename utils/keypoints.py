
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from typing import Optional

def farthest_point_sampling(
        points: np.ndarray,
        k: int,
        random_seed: Optional[int] = 42
) -> np.ndarray:
    """Select k points from the input using farthest point sampling.

    Args:
        points: 2D array of shape (n_points, n_dimensions)
        k: Number of points to select (must be <= n_points)
        random_seed: Optional random seed for reproducibility

    Returns:
        Array of shape (k, n_dimensions) containing selected points
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2:
        raise ValueError("points must be a 2D numpy array")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(points):
        raise ValueError("k cannot be larger than number of points")

    if random_seed is not None:
        np.random.seed(random_seed)

    n_points = len(points)
    selected_indices = [np.random.randint(n_points)]
    distances = distance.cdist([points[selected_indices[0]]], points)[0]

    for _ in tqdm(range(1, k), desc="Farthest Point Sampling"):
        new_idx = np.argmax(distances)
        selected_indices.append(new_idx)
        new_distances = distance.cdist([points[new_idx]], points)[0]
        distances = np.minimum(distances, new_distances)

    return points[selected_indices]


def project_3D_points_to_2D(points_3D, R, t, K):
    """
    Projects 3D keypoints into 2D image space using P = K [R | t].
    - points_3D: (N, 3)
    - R: (3, 3)
    - t: (3, 1)
    - K: (3, 3) camera intrinsics
    Returns: (N, 2) 2D image coordinates
    """
    points_cam = (R @ points_3D.T + t).T  # (N, 3)
    points_2D = (K @ points_cam.T).T
    return np.array(points_2D[:, :2] / points_2D[:, 2:]).astype(float)

def crop_and_resize_keypoints(keypoints_2D, crop_box, target_size=(128, 128)):
    """
    Crop and resize 2D keypoints based on the provided bounding box.
    keypoints_2D: (K, 2) list or array of (x, y)
    crop_box: [x, y, w, h] bounding box in original image
    new_size: size of the crop (default 128)
    Returns: resized keypoints in the new coordinate system
    """

    x, y, w, h = crop_box
    keypoints_2D_cropped = keypoints_2D - np.array([x, y])

    # Scale keypoints to new size
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    keypoints_2D_resized = keypoints_2D_cropped * np.array([scale_x, scale_y])
    return keypoints_2D_resized.astype(float)

def map_keypoints_to_original(keypoints_128, crop_box, crop_size=128):
    """
    Map keypoints from resized crop (128x128) back to original image.
    keypoints_128: (K, 2) list or array of (x, y)
    crop_box: [x1, y1, x2, y2] bounding box in original image
    crop_size: size of the crop (default 128)
    Returns: list of (x, y) in original image coordinates
    """
    x1, y1, x2, y2 = crop_box
    crop_w = x2 - x1
    crop_h = y2 - y1

    scale_x = crop_w / crop_size
    scale_y = crop_h / crop_size

    keypoints_orig = []
    for x, y in keypoints_128:
        x_orig = x1 + x * scale_x
        y_orig = y1 + y * scale_y
        keypoints_orig.append((x_orig, y_orig))

    return keypoints_orig