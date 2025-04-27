import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda import amp  # For mixed precision training

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.keypointnet import KeypointNet
from training.dataset_keypointnet import KeypointDataset

def main():
    parser = argparse.ArgumentParser(description="Train R6DNet (KeypointNet-based model)")
    parser.add_argument("--obj_id", type=int, required=True, help="Object ID to train on")
    parser.add_argument("--obj_name", type=str, default=None, help="Object name (default: obj_{id})")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=128, help="Image size (H, W)")
    parser.add_argument("--backbone", type=str, default="efficientnet_b1", help="Backbone model name")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone weights")
    parser.add_argument("--num_keypoints", type=int, default=15, help="Number of keypoints to detect")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--project", type=str, default="training-runs/r6dnet", help="TensorBoard + plots output dir")
    parser.add_argument("--save_dir", type=str, default="models/r6dnet", help="Where to save model + results.csv")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    args = parser.parse_args()

    # Use all available GPUs if possible
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner

    obj_name = args.obj_name or f"obj_{args.obj_id:06d}"
    print(f"Training on {obj_name} (ID: {args.obj_id})")
    print(f"Using device: {device}")

    # # === Transform with caching ===
    # transform = transforms.Compose([
    #     transforms.Resize((args.imgsz, args.imgsz)),
    #     transforms.ToTensor(),
    # ])

    # === Dataset & Dataloader with pinned memory ===
    train_set = KeypointDataset(obj_id=args.obj_id, split="train")
    val_set = KeypointDataset(obj_id=args.obj_id, split="val")

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )

    # === Model ===
    model = KeypointNet(
        num_keypoints=args.num_keypoints,
        backbone=args.backbone,
        pretrained=args.pretrained
    )

    # === Train ===
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        obj_name=obj_name,
        project=args.project,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        imgsz=args.imgsz
    )

if __name__ == "__main__":
    main()