import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def find_high_visib_instances(bop_root, obj_id, threshold=0.85):
    train_pbr = Path(bop_root) / "train_pbr"
    results = []

    for scene_path in tqdm(sorted(train_pbr.glob("*/"))):
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

            for idx, ann in enumerate(gt_list):
                if ann["obj_id"] == obj_id:
                    visib = info_list[idx].get("visib_fract", 0.0)
                    if visib >= threshold:
                        results.append({
                            "scene_id": scene_id,
                            "image_id": f"{int(img_id_str):06d}",
                            "ann_idx": idx,
                            "visib_fract": visib,
                            "bbox_obj": info_list[idx].get("bbox_obj", 0.0),
                        })
                        break  # only one instance needed per image

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, required=True)
    parser.add_argument("--bop_root", type=str, default="datasets/lm", help="Path to BOP dataset root")
    parser.add_argument("--threshold", type=float, default=0.85, help="Minimum visib_fraction required")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of entries to select")
    args = parser.parse_args()

    Path("data/scene_lists").mkdir(parents=True, exist_ok=True)

    entries = find_high_visib_instances(args.bop_root, args.obj_id, args.threshold)
    print(f"🔍 Found {len(entries)} total entries")
    
    random.seed(42)
    sampled = random.sample(entries, min(len(entries), args.max_samples))

    out_path = Path("data/scene_lists") / f"{args.obj_id:06d}.json"
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"✅ Saved {len(sampled)} sampled entries to {out_path}")