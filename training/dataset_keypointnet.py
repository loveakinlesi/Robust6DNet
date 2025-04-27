from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache

class KeypointDataset(Dataset):
    def __init__(self, obj_id, root="data", split="train", val_ratio=0.2, transform=None):
        self.obj_id = f"{int(obj_id):06d}"
        self.root = Path(root)
        self.transform = transform
        
        # Cache the image list
        image_dir = self.root / "crops" / self.obj_id
        self.images = sorted(list(image_dir.glob("*.png")))
        
        # Split data once during initialization
        # Set random seed for reproducible splits
        np.random.seed(42)
        np.random.shuffle(self.images)
        val_len = int(len(self.images) * val_ratio)
        self.images = self.images[:-val_len] if split == "train" else self.images[-val_len:]
        
        # Pre-compute heatmap paths
        self.heatmap_paths = [
            Path(str(img_path).replace("crops", "heatmaps").replace(".png", ".npz"))
            for img_path in self.images
        ]

    def __len__(self):
        return len(self.images)

    @lru_cache(maxsize=128)  # Cache recently loaded items
    
    def _load_heatmap(self, path):
        return np.load(path)["heatmaps"]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = image.astype(np.float32) / 255.0  # normalize
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

        heatmap = self._load_heatmap(self.heatmap_paths[idx])
        heatmap = heatmap.astype(np.float32)
        heatmap = torch.from_numpy(heatmap)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(heatmap, dtype=torch.float32), img_path.name
