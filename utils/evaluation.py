import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.keypoints import project_3D_points_to_2D

def compute_pck(pred_keypoints, gt_keypoints, threshold=5.0):
    """Compute Percentage of Correct Keypoints within threshold (PCK)."""
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    correct = (distances <= threshold).astype(np.float32)
    return correct.mean() * 100  # percentage

def compute_mde(pred_keypoints, gt_keypoints):
    """Compute Mean Distance Error (MDE) in pixels."""
    distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    return distances.mean()

def estimate_pose_pnp(object_points, image_points, K, dist_coeffs=None):
    """
    Estimate pose using PnP (with or without RANSAC).
    object_points: (N, 3)
    image_points: (N, 2)
    K: (3, 3) camera intrinsic matrix
    """
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise ValueError("PnP failed.")
    return rvec, tvec

def estimate_pose_pnp_ransac(object_points, image_points, K, dist_coeffs=None, iterationsCount=5000, reprojectionError=5):
    """
    Estimate pose using PnP (with or without RANSAC).
    object_points: (N, 3)
    image_points: (N, 2)
    K: (3, 3) camera intrinsic matrix
    """
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))           
   
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, distCoeffs=dist_coeffs,
        iterationsCount=iterationsCount,
        reprojectionError=reprojectionError,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("PnP failed.")
    return rvec, tvec

def compute_reprojection_error(R, t, object_points, image_points, K, dist_coeffs=None):
    """Compute mean 2D reprojection error."""
    if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))
    projected = project_3D_points_to_2D(object_points, R, t, K)        
            
    # projected, _ = cv2.projectPoints(object_points, R, t, K, dist_coeffs)
    # projected = projected.squeeze()
    error = np.linalg.norm(projected - image_points, axis=1).mean()
    return error


def compute_add(R_pred, t_pred, R_gt, t_gt, model_points):
    """
    Compute ADD (Average Distance of Model Points) for asymmetric objects.
    """
    # pred_pts = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    # gt_pts = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)
    # add = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    # return add


    t_pred = np.array(t_pred).reshape(3)
    t_gt = np.array(t_gt).reshape(3)
    
    # Perform matrix multiplication and addition
    pred_pts = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    gt_pts = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)

    # Compute ADD
    add = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
    return add

def compute_adds(R_pred, t_pred, R_gt, t_gt, model_points):
    """
    Compute ADD-S (Average closest point distance) for symmetric objects.
    """
    pred_pts = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    gt_pts = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)
    dists = np.linalg.norm(pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :], axis=2)
    adds = np.min(dists, axis=1).mean()
    return adds
