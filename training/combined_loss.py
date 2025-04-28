import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, soft_pck_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.soft_pck_weight = soft_pck_weight
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, pred_heatmaps, gt_heatmaps, gt_keypoints_2D):
        """
        pred_heatmaps: (B, N, H, W)
        gt_heatmaps: (B, N, H, W)
        gt_keypoints_2D: (B, N, 2) in pixel coordinates
        """
        B, N, H, W = pred_heatmaps.shape

        # 1. MSE Loss on heatmaps
        mse_loss = self.mse_loss_fn(pred_heatmaps, gt_heatmaps)

        # 2. Soft PCK loss
        soft_pck_loss = 0.0
        for b in range(B):
            for n in range(N):
                x, y = gt_keypoints_2D[b, n]  # (pixel coordinates)
                x = torch.clamp(x, 0, W - 1).long()
                y = torch.clamp(y, 0, H - 1).long()
                pred_value = pred_heatmaps[b, n, y, x]  # Note (y, x) order
                soft_pck_loss += (1.0 - pred_value) ** 2  # We want pred_value → 1
        
        soft_pck_loss = soft_pck_loss / (B * N)

        # Final combined loss
        total_loss = self.mse_weight * mse_loss + self.soft_pck_weight * soft_pck_loss
        return total_loss, mse_loss, soft_pck_loss
