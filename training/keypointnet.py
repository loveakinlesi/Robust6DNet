from pathlib import Path
import sys
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.combined_loss import CombinedLoss


class KeypointNet(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        backbone: str = "efficientnet_b1",
        pretrained: bool = True,
        head_channels: int = 128,
        output_size: tuple = (128, 128),
        upsample_mode: str = "transpose",
        use_sigmoid: bool = True,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.output_size = output_size
        self.use_sigmoid = use_sigmoid

        self.backbone = self._get_backbone(backbone, pretrained)
        backbone_out_channels = self._get_backbone_output_channels(backbone)

        self.head = self._build_head(
            in_channels=backbone_out_channels,
            head_channels=head_channels,
            output_size=output_size,
            upsample_mode=upsample_mode,
            use_sigmoid=use_sigmoid,
            dropout_prob=dropout_prob
        )

    def _get_backbone(self, name, pretrained):
        if name.startswith("resnet"):
            from torchvision.models import (
                resnet18, resnet34, resnet50,
                ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
            )
            model_fn = {
                "resnet18": resnet18,
                "resnet34": resnet34,
                "resnet50": resnet50
            }[name]
            weight_enum = {
                "resnet18": ResNet18_Weights.DEFAULT,
                "resnet34": ResNet34_Weights.DEFAULT,
                "resnet50": ResNet50_Weights.DEFAULT
            }[name] if pretrained else None

        elif name.startswith("efficientnet"):
            from torchvision.models import (
                efficientnet_b0, efficientnet_b1,
                EfficientNet_B0_Weights, EfficientNet_B1_Weights
            )
            model_fn = {
                "efficientnet_b0": efficientnet_b0,
                "efficientnet_b1": efficientnet_b1
            }[name]
            weight_enum = {
                "efficientnet_b0": EfficientNet_B0_Weights.DEFAULT,
                "efficientnet_b1": EfficientNet_B1_Weights.DEFAULT
            }[name] if pretrained else None

        else:
            raise ValueError(f"Unsupported backbone: {name}")

        model = model_fn(weights=weight_enum)
        return nn.Sequential(*list(model.children())[:-2])

    def _get_backbone_output_channels(self, name):
        return {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
        }[name]


    def _build_head(self, in_channels, head_channels, output_size, upsample_mode, use_sigmoid, dropout_prob):
        """Builds the head of the model."""
        if upsample_mode not in ["transpose", "bilinear"]:
            raise ValueError("Unsupported upsample mode. Choose 'transpose' or 'bilinear'.")
        if not (0 <= dropout_prob <= 1):
            raise ValueError("Dropout probability must be between 0 and 1.")

        # Build the head
        layers = [
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_prob:
            layers.append(nn.Dropout2d(dropout_prob))

        if upsample_mode == "transpose":
            for _ in range(5):
                layers.extend([
                    nn.ConvTranspose2d(head_channels, head_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                ])
        elif upsample_mode == "bilinear":
            layers.append(nn.Upsample(size=output_size, mode="bilinear", align_corners=False))

        layers.append(nn.Conv2d(head_channels, self.num_keypoints, kernel_size=1))

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = (x - x.amin(dim=(-2, -1), keepdim=True)) / (x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + 1e-8)
        return x

    def fit(self, train_loader, val_loader, obj_name, project="training-runs/r6dnet", save_dir="models/r6dnet", epochs=150, lr=1e-4, device=None, imgsz=128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        print(f"Running training on {device}")
        self.to(device)
        self.train()

        # Initialize logging and directories
        save_dir, project_dir = Path(save_dir), Path(project) / obj_name
        save_dir.mkdir(parents=True, exist_ok=True)
        project_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(log_dir=str(project_dir / "tensorboard"))
        results_csv = project_dir / "results.csv"

        # Model summary
        print("\n📐 Model Architecture Summary")
        summary(self, input_size=(1, 3, imgsz, imgsz))

        # Training setup
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # criterion = nn.MSELoss()
        criterion = CombinedLoss(mse_weight=1.0, soft_pck_weight=1.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Initialize CSV logging
        if not results_csv.exists():
            pd.DataFrame(columns=["epoch", "time", "train/loss", "val/loss", "mse", "pck", "mde", "lr"]).to_csv(results_csv, index=False)

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            # Training phase
            self.train()
            train_loss = 0
            with tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}") as pbar:
                for images, heatmaps, keypoints, _ in pbar:
                    images, heatmaps, keypoints = images.to(device), heatmaps.to(device), keypoints.to(device)
                    optimizer.zero_grad()
                    preds = self(images)
                    loss_t, mse_loss_t, soft_pck_loss_t = criterion(preds, heatmaps, keypoints)
                    # loss = criterion(preds, heatmaps)
                    loss_t.backward()
                    optimizer.step()
                    train_loss += loss_t.item()
                    pbar.set_postfix({'loss': f'{loss_t.item():.4f}'})

            # Validation phase
            self.eval()
            val_loss = 0
            val_mse_loss = 0
            val_soft_pck_loss = 0
            
            val_preds, val_targets = [], []
            with torch.no_grad():
                for images, heatmaps, keypoints, _ in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
                    images, heatmaps, keypoints = images.to(device), heatmaps.to(device), keypoints.to(device)
                    preds = self(images)
                    loss, mse_loss, soft_pck_loss = criterion(preds, heatmaps, keypoints)
                    val_loss += loss.item()
                    val_mse_loss+= mse_loss.item()
                    val_soft_pck_loss +=soft_pck_loss.item()
                    val_preds.append(preds)
                    val_targets.append(heatmaps)

            # Compute metrics
            metrics = compute_metrics(torch.cat(val_preds), torch.cat(val_targets))
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_mse_loss/= len(val_loader)
            val_soft_pck_loss/= len(val_loader)

            scheduler.step(val_loss)

            # Logging
            log_dict = {
                'epoch': epoch,
                'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'train/loss': train_loss,
                'val/loss': val_loss,
                'metrics/mse': val_mse_loss,
                'metrics/soft_pck': val_soft_pck_loss,
                'metrics/pck': metrics['pck'],
                'metrics/mde': metrics['mde'], 
                'lr': optimizer.param_groups[0]['lr']
            }

            # Update CSV and TensorBoard
            pd.DataFrame([log_dict]).to_csv(results_csv, mode='a', header=False, index=False)
            for k, v in log_dict.items():
                if k == 'epoch':
                    continue
                elif isinstance(v, (int, float)):
                    writer.add_scalar(k, v, epoch)

            # Save checkpoints
            # Save intermediate checkpoints in project directory
            torch.save(self.state_dict(), project_dir / f"last.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), project_dir / f"best.pt")
                # Save best model in save_dir
                torch.save(self.state_dict(), save_dir / f"{obj_name}.pt")

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"📊 Metrics: {metrics}")

        writer.close()
        # Plot results
        self._plot_training_curves(results_csv, project_dir / "plots")

    def _plot_training_curves(self, results_csv, plot_dir):
        plot_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(results_csv)
        for col in ["train/loss", "val/loss", "mse", "pck", "mde"]:
            plt.figure()
            plt.plot(df["epoch"], df[col], marker='o')
            plt.title(col)
            plt.xlabel("Epoch")
            plt.ylabel(col)
            plt.grid(True)
            plt.savefig(plot_dir / f"{str(col).replace('/', '_')}.png")
            plt.close()

def compute_metrics(preds, targets, threshold=5.0):
    # Compute coordinates in one step using torch operations
    with torch.no_grad():  # Reduce memory usage during evaluation
        B, K, H, W = preds.shape
        
        # Flatten and get coordinates in a single operation
        pred_flat = preds.view(B, K, -1)
        target_flat = targets.view(B, K, -1) 
        
        # Get max indices efficiently
        pred_coords = pred_flat.argmax(dim=-1)
        target_coords = target_flat.argmax(dim=-1)
        
        # Compute x,y coordinates using tensor operations
        pred_coords_2d = torch.stack([pred_coords % W, pred_coords // W], dim=-1).float()
        target_coords_2d = torch.stack([target_coords % W, target_coords // W], dim=-1).float()
        
        # Compute distances efficiently using torch.norm
        abs_dists = torch.norm(pred_coords_2d - target_coords_2d, dim=-1)
        
        # Compute metrics
        pck = (abs_dists <= threshold).float().mean().item() * 100
        mde = abs_dists.mean().item()
        mse = torch.nn.functional.mse_loss(preds, targets, reduction='mean').item()

        return {
            "mse": mse,
            "pck": pck,
            "mde": mde
        }
