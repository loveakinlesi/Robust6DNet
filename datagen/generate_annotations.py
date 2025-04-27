import json
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.keypoints import project_3D_points_to_2D, map_keypoints_to_original

def generate_annotations(obj_id: int):
    BOP_PATH = Path("datasets/lm/train_pbr")
    SCENE_LIST_PATH = Path("data/scene_lists") / f"{obj_id:06d}.json"
    KEYPOINTS_PATH = Path("data/keypoints3d") / f"{obj_id:06d}.json"
    OUTPUT_PATH = Path("data/annotations")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    with open(SCENE_LIST_PATH) as f:
        selection = json.load(f)

    with open(KEYPOINTS_PATH) as f:
        keypoints_data = json.load(f)
    keypoints_3D = np.array(keypoints_data["keypoints_3D"])

    annotations = []

    print(f"[INFO] Generating annotations for {len(selection)} images...")

    # Cache scene data to avoid reloading the same files
    scene_cache = {}

    for entry in tqdm(selection, desc="Processing images"):
        scene_id = entry["scene_id"]
        image_id_str = entry["image_id"]
        ann_idx = entry["ann_idx"]
        image_key = f"{scene_id}_{image_id_str}"

        if scene_id not in scene_cache:
            scene_path = BOP_PATH / scene_id
            scene_cache[scene_id] = {
                "cam_data": json.load(open(scene_path / "scene_camera.json")),
                "gt_data": json.load(open(scene_path / "scene_gt.json")),
                "gt_info": json.load(open(scene_path / "scene_gt_info.json")),
            }
        image_id = str(int(image_id_str))
        cam = scene_cache[scene_id]["cam_data"][image_id]
        gt = scene_cache[scene_id]["gt_data"][image_id][ann_idx]
        info = scene_cache[scene_id]["gt_info"][image_id][ann_idx]
        rgb_path = BOP_PATH / scene_id / "rgb" / f"{int(image_id):06d}.jpg"

        K = np.array(cam["cam_K"]).reshape(3, 3)
        R = np.array(gt["cam_R_m2c"]).reshape(3, 3)
        t = np.array(gt["cam_t_m2c"]).reshape(3, 1) / 1000
        bbox_obj = info["bbox_obj"]
        bbox_visib = info["bbox_visib"]
        keypoints_2D = project_3D_points_to_2D(keypoints_3D, R, t, K)

        annotations.append({
            "obj_id": obj_id,
            "image_id": image_key,
            "rgb_path": str(rgb_path),
            "bbox_visib": bbox_visib.tolist(),
            "bbox_obj": bbox_obj.tolist(),
            "K": K.tolist(),
            "rotation": R.tolist(),
            "translation": t.flatten().tolist(),
            "keypoints_2D": keypoints_2D.tolist()
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
