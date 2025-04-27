import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.image_handling import crop_and_resize

CROP_SIZE = (128, 128)

def generate_crops(obj_id: int):
    annotation_path = Path("data/annotations") / f"{obj_id:06d}.json"
    output_dir = Path("data/crops") / f"{obj_id:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    print(f"[INFO] Cropping {len(annotations)} images to 128x128 for object {obj_id}")

    for ann in tqdm(annotations):
        image_id = ann["image_id"]  # e.g., "000012_0042"
        rgb_path = Path(ann["rgb_path"])
        bbox = ann["bbox_visib"]

        try:
            image = Image.open(rgb_path).convert("RGB")
            crop = crop_and_resize(image, bbox, CROP_SIZE)
            crop.save(output_dir / f"{image_id}.png")
        except Exception as e:
            print(f"[!] Failed to crop {rgb_path}: {e}")

    print(f"✅ Crops saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, required=True)
    args = parser.parse_args()

    generate_crops(args.obj_id)
