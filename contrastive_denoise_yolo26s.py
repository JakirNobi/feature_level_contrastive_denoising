"""
Contrastive Denoising Auxiliary Modules for YOLO26s — Barlow Twins variant (v10.1).

Fixes vs v10:
  - Removed stop-gradient on the clean side. Barlow Twins is designed to
    work without it, and it was interfering with the on-diagonal term.
  - Persisted clean_neck across the forward so the pilot gap check can
    find it in on_train_batch_end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any, Dict


# ===================== ENCODERS =====================

class GeometricEncoder(nn.Module):
    def __init__(self, out_channels_list: Optional[List[int]] = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [128, 256, 512]

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

        self.p3 = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, out_channels_list[0], 3, padding=1, stride=2),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, out_channels_list[1], 3, padding=1, stride=2),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, out_channels_list[2], 3, padding=1, stride=2),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        sobel_x: torch.Tensor = self.sobel_x  # type: ignore[assignment]
        sobel_y: torch.Tensor = self.sobel_y  # type: ignore[assignment]
        gx = F.conv2d(x, sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, sobel_y, padding=1, groups=3)
        edges = torch.cat([gx, gy], dim=1)
        return [self.p3(edges), self.p4(edges), self.p5(edges)]


class SemanticEncoder(nn.Module):
    def __init__(self, out_channels_list: Optional[List[int]] = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [128, 256, 512]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
        )
        self.p3_branch = nn.Sequential(
            nn.Conv2d(128, out_channels_list[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels_list[0]), nn.SiLU(),
        )
        self.p4_branch = nn.Sequential(
            nn.Conv2d(out_channels_list[0], out_channels_list[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[1]), nn.SiLU(),
        )
        self.p5_branch = nn.Sequential(
            nn.Conv2d(out_channels_list[1], out_channels_list[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[2]), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x  = self.stem(x)
        p3 = self.p3_branch(x)
        p4 = self.p4_branch(p3)
        p5 = self.p5_branch(p4)
        return [p3, p4, p5]


# ===================== BARLOW PROJECTION HEAD =====================

class BarlowProjectionHead(nn.Module):
    def __init__(self, in_ch: int, hidden_dim: int = 512, out_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== BARLOW TWINS LOSS =====================

def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    lambda_off: float = 0.005,
    eps: float = 1e-6,
) -> torch.Tensor:
    z_a = z_a.float()
    z_b = z_b.float()
    N, D = z_a.shape

    z_a = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + eps)
    z_b = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + eps)

    c = (z_a.T @ z_b) / N

    on_diag  = (torch.diagonal(c) - 1.0).pow(2).sum()
    off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()

    return on_diag + lambda_off * off_diag


# ===================== MULTI-HOOK =====================

class MultiHook(nn.Module):
    """
    v10.1:
      - Barlow Twins loss, projection heads at 2048 dims by default.
      - NO stop-gradient on the clean side (Barlow Twins doesn't need it,
        and it was interfering with the on-diagonal term).
      - Persists clean_neck so the pilot gap check can use it.
    """

    def __init__(
        self,
        neck_channels: Optional[List[int]] = None,
        proj_hidden:   int  = 512,
        proj_out:      int  = 2048,
        use_geometric: bool = True,
        use_semantic:  bool = True,
    ) -> None:
        super().__init__()
        if neck_channels is None:
            neck_channels = [128, 256, 512]

        self.neck_channels = neck_channels
        self.use_geometric = use_geometric
        self.use_semantic  = use_semantic
        self.clean_images: Optional[torch.Tensor] = None
        self.features:     Dict[int, torch.Tensor] = {}
        self.last_clean_neck: Optional[List[torch.Tensor]] = None   # persisted

        self.proj_neck = nn.ModuleList([
            BarlowProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
        ])

        self.geo_encoder:    Optional[GeometricEncoder] = None
        self.sem_encoder:    Optional[SemanticEncoder]  = None
        self.proj_clean_geo: Optional[nn.ModuleList]    = None
        self.proj_clean_sem: Optional[nn.ModuleList]    = None

        if use_geometric:
            self.geo_encoder    = GeometricEncoder(out_channels_list=neck_channels)
            self.proj_clean_geo = nn.ModuleList([
                BarlowProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
            ])
        if use_semantic:
            self.sem_encoder    = SemanticEncoder(out_channels_list=neck_channels)
            self.proj_clean_sem = nn.ModuleList([
                BarlowProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
            ])

    def set_clean(self, clean_imgs: torch.Tensor) -> None:
        self.clean_images = clean_imgs.float()

    def hook_p3(self, module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        self.features[16] = output

    def hook_p4(self, module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        self.features[19] = output

    def hook_p5(self, module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        self.features[22] = output

    def get_neck_outputs(self) -> List[torch.Tensor]:
        return [self.features[16], self.features[19], self.features[22]]

    def compute_loss_from_features(
        self,
        noisy_neck: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.proj_neck[0].net[0].weight.device
        zero   = torch.tensor(0.0, device=device)

        if self.clean_images is None:
            return zero, zero

        # Persist clean_neck for the pilot gap check
        self.last_clean_neck = noisy_neck  # placeholder; overwritten below

        geo_loss = zero
        sem_loss = zero

        if self.use_geometric and self.geo_encoder is not None and self.proj_clean_geo is not None:
            geo_feats = self.geo_encoder(self.clean_images)
            noisy_proj = [proj(f) for proj, f in zip(self.proj_neck, noisy_neck)]
            # NO stop-gradient on the clean side for Barlow Twins
            clean_proj = [proj(f) for proj, f in zip(self.proj_clean_geo, geo_feats)]
            geo_loss = self._barlow_over_scales(noisy_proj, clean_proj)

        if self.use_semantic and self.sem_encoder is not None and self.proj_clean_sem is not None:
            sem_feats = self.sem_encoder(self.clean_images)
            noisy_proj = [proj(f) for proj, f in zip(self.proj_neck, noisy_neck)]
            clean_proj = [proj(f) for proj, f in zip(self.proj_clean_sem, sem_feats)]
            sem_loss = self._barlow_over_scales(noisy_proj, clean_proj)

        self.features = {}
        return geo_loss, sem_loss

    @staticmethod
    def _barlow_over_scales(
        noisy_proj: List[torch.Tensor],
        clean_proj: List[torch.Tensor],
    ) -> torch.Tensor:
        total = 0.0
        count = 0
        for zn, zc in zip(noisy_proj, clean_proj):
            B, D, H, W = zn.shape
            zn_flat = zn.permute(0, 2, 3, 1).reshape(B * H * W, D)
            zc_flat = zc.permute(0, 2, 3, 1).reshape(B * H * W, D)
            total = total + barlow_twins_loss(zn_flat, zc_flat)
            count += 1
        return total / count

    def get_auxiliary_param_count(self) -> int:
        count = 0
        for mod in [
            self.proj_neck, self.geo_encoder, self.sem_encoder,
            self.proj_clean_geo, self.proj_clean_sem,
        ]:
            if mod is not None:
                count += sum(p.numel() for p in mod.parameters())
        return count


# ===================== ADAPTIVE LOSS WEIGHTER =====================

class AdaptiveLossWeighter(nn.Module):
    def __init__(
        self,
        num_tasks:    int   = 2,
        warmup_epochs: int  = 10,
        total_epochs:  int  = 100,
        init_log_var:  float = 0.0,
    ) -> None:
        super().__init__()
        self.log_var       = nn.Parameter(torch.full((num_tasks,), init_log_var))
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs

    def forward(self, losses: List[torch.Tensor], epoch: int) -> torch.Tensor:
        precision = torch.exp(-self.log_var)
        total = sum(
            precision[i] * losses[i] + self.log_var[i]
            for i in range(len(losses))
        )
        ramp = epoch / max(1, self.warmup_epochs) if epoch < self.warmup_epochs else 1.0
        return ramp * total