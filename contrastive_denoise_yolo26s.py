"""
Contrastive Denoising Auxiliary Modules for YOLO26s.
Training-only components discarded at inference.

Components:
  GeometricEncoder  - Extracts clean edge features via Sobel filters
  SemanticEncoder   - Extracts clean high-level facial features
  FusionModule      - Combines geometric + semantic features
  ProjectionHead    - Maps features to contrastive embedding space
  MultiHook         - Captures neck outputs during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class GeometricEncoder(nn.Module):
    """
    Extracts clean geometric features using fixed Sobel filters
    followed by lightweight learnable refinement branches.
    Outputs: P3 (1/8), P4 (1/16), P5 (1/32).
    """

    def __init__(self, out_channels_list: Optional[List[int]] = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [128, 256, 512]

        # Create Sobel kernels as tensors
        sobel_x: torch.Tensor = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y: torch.Tensor = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Register as buffers
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

        # P3: 3 strides (1→1/2→1/4→1/8)
        self.p3: nn.Sequential = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, out_channels_list[0], 3, padding=1, stride=2),
        )
        # P4: 4 strides (1→1/2→1/4→1/8→1/16)
        self.p4: nn.Sequential = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, out_channels_list[1], 3, padding=1, stride=2),
        )
        # P5: 5 strides (1→1/2→1/4→1/8→1/16→1/32)
        self.p5: nn.Sequential = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.Conv2d(256, out_channels_list[2], 3, padding=1, stride=2),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Access buffers with explicit tensor type
        sobel_x: torch.Tensor = self.sobel_x  # type: ignore[assignment]
        sobel_y: torch.Tensor = self.sobel_y  # type: ignore[assignment]

        gx: torch.Tensor = F.conv2d(x, sobel_x, padding=1, groups=3)
        gy: torch.Tensor = F.conv2d(x, sobel_y, padding=1, groups=3)
        edges: torch.Tensor = torch.cat([gx, gy], dim=1)
        return [self.p3(edges), self.p4(edges), self.p5(edges)]


class SemanticEncoder(nn.Module):
    """
    Lightweight clean feature extractor capturing high-level
    semantic information. Outputs: P3 (1/8), P4 (1/16), P5 (1/32).
    """
    def __init__(self, out_channels_list: Optional[List[int]] = None) -> None:
        super().__init__()
        if out_channels_list is None:
            out_channels_list = [128, 256, 512]

        # Stem: 3 stride‑2 convs -> 1/8 resolution
        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(),
        )
        # P3: keep spatial size (80x80), just change channels
        self.p3_branch: nn.Sequential = nn.Sequential(
            nn.Conv2d(128, out_channels_list[0], 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels_list[0]), nn.SiLU(),
        )
        # P4: stride 2 -> 40x40
        self.p4_branch: nn.Sequential = nn.Sequential(
            nn.Conv2d(out_channels_list[0], out_channels_list[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[1]), nn.SiLU(),
        )
        # P5: stride 2 -> 20x20
        self.p5_branch: nn.Sequential = nn.Sequential(
            nn.Conv2d(out_channels_list[1], out_channels_list[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[2]), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)               # (B,128,80,80)
        p3 = self.p3_branch(x)         # (B,128,80,80)
        p4 = self.p4_branch(p3)        # (B,256,40,40)
        p5 = self.p5_branch(p4)        # (B,512,20,20)
        return [p3, p4, p5]

class FusionModule(nn.Module):
    """
    Combines geometric and semantic features into clean target
    representations using per-scale 1x1 convolutions.
    """

    def __init__(self, channels_list: Optional[List[int]] = None) -> None:
        super().__init__()
        if channels_list is None:
            channels_list = [128, 256, 512]

        self.fuse3: nn.Conv2d = nn.Conv2d(channels_list[0] * 2, channels_list[0], 1)
        self.fuse4: nn.Conv2d = nn.Conv2d(channels_list[1] * 2, channels_list[1], 1)
        self.fuse5: nn.Conv2d = nn.Conv2d(channels_list[2] * 2, channels_list[2], 1)
        self._fuse_layers: List[nn.Conv2d] = [self.fuse3, self.fuse4, self.fuse5]

    def forward(
        self,
        geo_feats: List[torch.Tensor],
        sem_feats: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        fused: List[torch.Tensor] = []
        for i, (g, s) in enumerate(zip(geo_feats, sem_feats)):
            x: torch.Tensor = torch.cat([g, s], dim=1)
            x = self._fuse_layers[i](x)
            fused.append(x)
        return fused


class ProjectionHead(nn.Module):
    """
    MLP head mapping feature maps to lower-dimensional embedding
    for contrastive loss computation.
    """

    def __init__(self, in_ch: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sampled_contrastive_loss(
    noisy_feats: List[torch.Tensor],
    clean_feats: List[torch.Tensor],
    num_samples: int = 1024,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Memory-efficient InfoNCE contrastive loss with random spatial sampling.
    """
    device: torch.device = noisy_feats[0].device
    total_loss: torch.Tensor = torch.tensor(0.0, device=device)
    num_scales: int = len(noisy_feats)

    for fn, fc in zip(noisy_feats, clean_feats):
        B: int = fn.shape[0]
        C: int = fn.shape[1]
        H: int = fn.shape[2]
        W: int = fn.shape[3]
        N: int = H * W

        fn = F.normalize(fn.view(B, C, -1), dim=1)
        fc = F.normalize(fc.view(B, C, -1), dim=1)

        n_samples: int = min(num_samples, N)
        idx: torch.Tensor = torch.randperm(N, device=fn.device)[:n_samples]

        fn_sampled: torch.Tensor = fn[:, :, idx]
        fc_sampled: torch.Tensor = fc[:, :, idx]

        pos_sim: torch.Tensor = (fn_sampled * fc_sampled).sum(dim=1) / temperature
        pos_sim = pos_sim.reshape(-1, 1)

        fn_all: torch.Tensor = fn_sampled.permute(0, 2, 1).reshape(B * n_samples, C)
        fc_all: torch.Tensor = fc_sampled.permute(0, 2, 1).reshape(B * n_samples, C)

        logits: torch.Tensor = torch.matmul(fn_all, fc_all.T) / temperature
        labels: torch.Tensor = torch.arange(B * n_samples, device=fn.device)

        total_loss = total_loss + F.cross_entropy(logits, labels)

    return total_loss / num_scales


class MultiHook(nn.Module):
    """
    Captures neck output features from YOLO26s layers 16, 19, 22.
    Supports conditional encoder activation for ablation studies.
    """

    def __init__(
        self,
        neck_channels: Optional[List[int]] = None,
        proj_hidden: int = 128,
        proj_out: int = 64,
        use_geometric: bool = True,
        use_semantic: bool = True
    ) -> None:
        super().__init__()
        if neck_channels is None:
            neck_channels = [128, 256, 512]

        self.neck_channels: List[int] = neck_channels
        self.use_geometric: bool = use_geometric
        self.use_semantic: bool = use_semantic
        self.clean_images: Optional[torch.Tensor] = None
        self.features: dict[int, torch.Tensor] = {}

        self.proj_neck: nn.ModuleList = nn.ModuleList([
            ProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
        ])

        self.geo_encoder: Optional[GeometricEncoder] = None
        self.sem_encoder: Optional[SemanticEncoder] = None
        self.fusion: Optional[FusionModule] = None
        self.proj_clean: Optional[nn.ModuleList] = None

        if use_geometric or use_semantic:
            self.proj_clean = nn.ModuleList([
                ProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
            ])

        if use_geometric:
            self.geo_encoder = GeometricEncoder(out_channels_list=neck_channels)

        if use_semantic:
            self.sem_encoder = SemanticEncoder(out_channels_list=neck_channels)

        if use_geometric and use_semantic:
            self.fusion = FusionModule(neck_channels)

    def set_clean(self, clean_imgs: torch.Tensor) -> None:
        """Store clean reference images for auxiliary encoder forward pass."""
        self.clean_images = clean_imgs

    def hook_p3(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self.features[16] = output

    def hook_p4(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self.features[19] = output

    def hook_p5(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self.features[22] = output

    def get_neck_outputs(self) -> List[torch.Tensor]:
        """Retrieve captured neck features in order [P3, P4, P5]."""
        return [self.features[16], self.features[19], self.features[22]]

    def compute_loss(self) -> torch.Tensor:
        """Compute contrastive loss based on active encoders."""
        if self.proj_neck is None:
            return torch.tensor(0.0)
        device: torch.device = next(self.proj_neck[0].parameters()).device

        if self.clean_images is None or len(self.features) < 3:
            return torch.tensor(0.0, device=device)

        if not self.use_geometric and not self.use_semantic:
            self.features = {}
            return torch.tensor(0.0, device=device)

        targets: List[List[torch.Tensor]] = []

        if self.use_geometric and self.geo_encoder is not None:
            geo_feats: List[torch.Tensor] = self.geo_encoder(self.clean_images)
            targets.append(geo_feats)

        if self.use_semantic and self.sem_encoder is not None:
            sem_feats: List[torch.Tensor] = self.sem_encoder(self.clean_images)
            targets.append(sem_feats)

        if len(targets) == 2 and self.fusion is not None:
            clean_feats: List[torch.Tensor] = self.fusion(targets[0], targets[1])
        else:
            clean_feats = targets[0]

        neck_outs: List[torch.Tensor] = self.get_neck_outputs()
        noisy_proj: List[torch.Tensor] = [
            proj(f) for proj, f in zip(self.proj_neck, neck_outs)
        ]

        if self.proj_clean is not None:
            clean_proj: List[torch.Tensor] = [
                proj(f) for proj, f in zip(self.proj_clean, clean_feats)
            ]
        else:
            self.features = {}
            return torch.tensor(0.0, device=device)

        self.features = {}
        return sampled_contrastive_loss(noisy_proj, clean_proj)

    def get_auxiliary_param_count(self) -> int:
        """Count parameters in auxiliary modules (discarded at inference)."""
        count: int = 0
        for module in [self.geo_encoder, self.sem_encoder,
                       self.fusion, self.proj_clean]:
            if module is not None:
                count += sum(p.numel() for p in module.parameters())
        return count