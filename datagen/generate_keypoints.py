import json
import sys
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.keypoints import farthest_point_sampling

def generate_3D_keypoints(obj_id: int, num_keypoints: int = 15):
    """
    Generate 3D keypoints for a given object ID by sampling points from the mesh surface.

    Args:
        obj_id (int): The ID of the object.
        num_keypoints (int): The number of keypoints to generate.

    Returns:
        np.ndarray: An array of shape (num_keypoints, 3) containing the sampled keypoints.
    """
    DATASET_DIR = Path("datasets/lm")
    OUTPUT_PATH = Path("data/keypoints3d") / f"{obj_id:06d}.json"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    mesh_path = DATASET_DIR / "models" / f"obj_{obj_id:06d}.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found at {mesh_path}")

    print(f"🔍 Loading mesh from {mesh_path}")
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices * 0.001 # Convert to meters
    # vertices = np.array(mesh.vertices) / 1000 # Normalize to meters


    # First keypoint = centroid
    centroid = vertices.mean(axis=0, keepdims=True)

    # Sample remaining points using FPS
    rest = farthest_point_sampling(vertices, k=num_keypoints - 1, random_seed=42)
    keypoints = np.concatenate([centroid, rest], axis=0)
    keypoints = np.array(keypoints)
    keypoints_rounded = np.round(keypoints, 8)
    print(f"✅ Generated {len(keypoints)} 3D keypoints for object {obj_id}")

    # Save keypoints to JSON
    keypoints_json = {
    "obj_id": obj_id,
    "num_keypoints": num_keypoints,
    "keypoints_3D": keypoints_rounded.tolist()
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(keypoints_json, f, indent=2)
    print(f"💾 Saved keypoints to {OUTPUT_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Generate 3D keypoints for a given object ID.")
    parser.add_argument("obj_id", type=int, help="The ID of the object.")
    parser.add_argument("--num_keypoints", type=int, default=15, help="Number of keypoints to generate.")
    args = parser.parse_args()

    generate_3D_keypoints(args.obj_id, args.num_keypoints)


if __name__ == "__main__":
    main()