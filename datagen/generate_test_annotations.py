import json
import argparse
import sys

import numpy as np
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.keypoints import project_3D_points_to_2D

def generate_annotations(obj_id: int):
    BOP_PATH = Path("datasets/lm")
    SCENE_GT_INFO_PATH = BOP_PATH / "test" / f"{obj_id:06d}" / "scene_gt_info.json"
    SCENE_GT_PATH = BOP_PATH / "test" / f"{obj_id:06d}" / "scene_gt.json"
    SCENE_CAMERA_PATH = BOP_PATH / "test" / f"{obj_id:06d}" / "scene_camera.json"
    RGB_DIR = BOP_PATH / "test" / f"{obj_id:06d}" / "rgb"
    KEYPOINTS_PATH = Path("data/keypoints3d") / f"{obj_id:06d}.json"
    OUTPUT_PATH = Path("data/annotations/test")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    with open(KEYPOINTS_PATH) as f:
        keypoints_data = json.load(f)
    keypoints_3D = np.array(keypoints_data["keypoints_3D"])

    # Project 3d to 2d
    with open(SCENE_CAMERA_PATH, 'r') as f:
        scene_camera = json.load(f)

    with open(SCENE_GT_PATH, 'r') as f:
        scene_gt = json.load(f)

    with open(SCENE_GT_INFO_PATH, 'r') as f:
        scene_gt_info = json.load(f)

    annotations = []
  
    for img_id_str, poses in tqdm(scene_gt.items(), desc="Generating Keypoint Annotations"):
        img_id = int(img_id_str)
        camera_data = scene_camera[img_id_str]
        obj_pose = poses[0]
        if obj_pose["obj_id"] != obj_id:
            continue  # Skip other objects

        # Camera intrinsics (3x3)
        K = np.array(camera_data["cam_K"]).reshape(3, 3)
        R = np.array(obj_pose["cam_R_m2c"]).reshape(3, 3)
        t = np.array(obj_pose["cam_t_m2c"]).reshape(3, 1) /1000
        bbox_visib = np.array(scene_gt_info[str(img_id)][0]["bbox_visib"])
        bbox_obj = np.array(scene_gt_info[str(img_id)][0]["bbox_obj"])
        
        keypoints_2D = project_3D_points_to_2D(keypoints_3D, R, t, K)

        annotations.append({
            "obj_id": obj_id,
            "image_id": img_id,
            "rgb_path": str(RGB_DIR / f"{img_id:06d}.png"),
             "bbox_visib": bbox_visib,
            "bbox_obj": bbox_obj,
            "K": K.tolist(),
            "rotation": R.tolist(),
            "translation": t.flatten().tolist(),
            "keypoints_2D": keypoints_2D.tolist(),
        })

    out_file = OUTPUT_PATH / f"{obj_id:06d}.json"
    with open(out_file, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ Saved {len(annotations)} annotations to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, required=True)
    args = parser.parse_args()

    generate_annotations(args.obj_id)
