import numpy as np

def gaussian(xL, yL, H, W, sigma=5):
    channel = [np.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))

    return channel

def generate_multi_gaussian_heatmaps(keypoints, H, W, sigma=5):
    """
    Efficient looped version to generate a stack of Gaussian heatmaps.
    - keypoints: (N, 2)
    - Returns: (N, H, W) heatmap array
    """
    N = keypoints.shape[0]
    heatmaps = np.zeros((N, H, W), dtype=np.float32)
    x = np.arange(W)
    y = np.arange(H)[:, None]

    for i in range(N):
        xL, yL = keypoints[i]
        heatmaps[i] = np.exp(-((x - xL) ** 2 + (y - yL) ** 2) / (2 * sigma ** 2))

    heatmaps[heatmaps > 1.0] = 1.0
    return heatmaps


def extract_argmax_coords_single(heatmap: np.ndarray):
    """Extract keypoint (x, y) from a heatmap using argmax."""
    flat_index = np.argmax(heatmap)
    y, x = np.unravel_index(flat_index, heatmap.shape)
    return float(x), float(y)

def decode_heatmaps(heatmaps: np.ndarray):
    """
    Extracts (x, y) coordinates for each keypoint from heatmaps of shape (K, H, W).
    Returns: List of (x, y) floats for each keypoint
    """
    K, H, W = heatmaps.shape
    coords = []

    for k in range(K):
        flat_idx = np.argmax(heatmaps[k])
        y, x = np.unravel_index(flat_idx, (H, W))
        coords.append((float(x), float(y)))

    return np.array(coords)  # List of (x, y)
