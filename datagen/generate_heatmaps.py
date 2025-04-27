import json
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.image_handling import pad_bbox
from utils.keypoints import crop_and_resize_keypoints

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.heatmap import generate_multi_gaussian_heatmaps

HEATMAP_SIZE = (128, 128)
SIGMA = 5

def generate_heatmaps(obj_id: int):
    annotation_path = Path("data/annotations") / f"{obj_id:06d}.json"
    output_dir = Path("data/heatmaps") / f"{obj_id:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    print(f"[INFO] Generating heatmaps for {len(annotations)} images")

    for ann in tqdm(annotations):
        image_id = ann["image_id"]
        bbox = pad_bbox(ann["bbox_visib"])
        keypoints_2D = np.array(ann["keypoints_2D"])
        kp_scaled = crop_and_resize_keypoints(keypoints_2D, bbox, target_size=HEATMAP_SIZE)
        heatmaps = generate_multi_gaussian_heatmaps(kp_scaled, *HEATMAP_SIZE, sigma=SIGMA)

        np.savez_compressed(output_dir / f"{image_id}.npz", heatmaps=heatmaps, keypoints=kp_scaled)

    print(f"✅ Heatmaps saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, required=True)
    args = parser.parse_args()

    generate_heatmaps(args.obj_id)
