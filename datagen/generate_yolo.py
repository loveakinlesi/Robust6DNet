from collections import defaultdict
from pathlib import Path
import json
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
random.seed(42)

# === CONFIG ===
IMAGE_SIZE = (640, 480)  # width, height
SOURCE_ROOT = Path("datasets/lm/train_pbr")
TARGET_ROOT = Path("datasets/lm_yolo")

IMAGE_DIR = TARGET_ROOT / "images"
LABEL_DIR = TARGET_ROOT / "labels"

LINEMOD_CLASSES = {
    1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can",
    6: "cat", 7: "cup", 8: "driller", 9: "duck", 10: "eggbox",
    11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"
}


# === Create output dirs ===
for split in ['train', 'val']:
    (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

# === Step 1: Aggregate object annotations per image ===
image_dict = defaultdict(lambda: {
    "scene_id": None,
    "image_id": None,
    "image_path": None,
    "image_label": None,
})

def yolo_label_generator(class_id, bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"


for scene_path in tqdm(sorted(SOURCE_ROOT.glob("*/")), desc="Processing scenes"):
    scene_id = scene_path.name
    gt_path = scene_path / "scene_gt.json"
    info_path = scene_path / "scene_gt_info.json"

    if not gt_path.exists() or not info_path.exists():
        continue

    with open(gt_path) as f:
        gt_data = json.load(f)

    with open(info_path) as f:
        info_data = json.load(f)

    for img_id_str, gt_list in gt_data.items():
        info_list = info_data[img_id_str]

        image_label = ""
        image_id = f"{int(img_id_str):06d}"
        key = f"{scene_id}_{image_id}"
        img_file = f"{image_id}.jpg"
        img_path = scene_path / "rgb" / img_file

        # ✅ Read real image size
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        for idx, ann in enumerate(gt_list):
            obj_id = ann["obj_id"]
            class_id = obj_id - 1  # Adjust for zero-based index
            visib = info_list[idx].get("visib_fract", 0.0)
            bbox = info_list[idx].get("bbox_visib", None)

            if visib > 0.0:
                image_label = image_label + yolo_label_generator(class_id, bbox, img_w, img_h)
                # break  # only one instance needed per image
        if not img_path.exists() or image_label == "":
            print(f"[!] Missing image or empty label: {img_path}")
            continue

        image_dict[key]["scene_id"] = scene_id
        image_dict[key]["image_id"] = image_id
        image_dict[key]["image_path"] = img_path
        image_dict[key]["image_label"] = image_label



# === Step 2: Train/Val Split on unique images ===
all_keys = random.sample(list(image_dict.keys()), 15000)
train_keys, val_keys = train_test_split(all_keys, test_size=0.2, random_state=42)

def process_split(keys, split):
    for key in tqdm(keys, desc=f"Processing {split} set"):
        entry = image_dict[key]
        scene_id = entry["scene_id"]
        image_id = entry["image_id"]
        img_src = entry["image_path"]

        filename = f"{scene_id}_{image_id}.jpg"
        img_dst = IMAGE_DIR / split / filename
        label_dst = LABEL_DIR / split / filename.replace(".jpg", ".txt")

        shutil.copy(img_src, img_dst)

        # Save label directly
        with open(label_dst, 'w') as f:
            f.write(entry["image_label"])


process_split(train_keys, "train")
process_split(val_keys, "val")

# === Write linemod.yaml
yaml_content = f"""train: {str((IMAGE_DIR / 'train').resolve())}
val: {str((IMAGE_DIR / 'val').resolve())}

nc: {len(LINEMOD_CLASSES)}
names: {list(LINEMOD_CLASSES.values())}
"""
(TARGET_ROOT / "linemod.yaml").write_text(yaml_content)
print("\n✅ Dataset and config ready!")

# === Show preview
df = pd.DataFrame([
    {
        "image": f"{image_dict[k]['scene_id']}_{image_dict[k]['image_id']}.jpg",
        "num_objects": len(image_dict[k]["image_label"].strip().splitlines())
    }
    for k in train_keys[:5]
])
print(df)