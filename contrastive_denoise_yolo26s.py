"""
Contrastive Denoising Auxiliary Modules for YOLO26s.
Training-only components — discarded at inference, zero overhead.

Components:
  GeometricEncoder     - Extracts clean edge features via Sobel filters + learnable convs
  SemanticEncoder      - Extracts clean high-level facial features via lightweight CNN
  FusionModule         - Combines geometric + semantic features (kept for compatibility)
  ProjectionHead       - Maps features to contrastive embedding space (64-dim)
  MultiHook            - Captures neck outputs at P3/P4/P5, computes split contrastive losses
  AdaptiveLossWeighter - Canonical Kendall et al. homoscedastic uncertainty weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any, Dict


# ===================== ENCODERS =====================

class GeometricEncoder(nn.Module):
    """Extracts clean geometric features using fixed Sobel filters + learnable strided convs."""

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
    """Lightweight CNN for high-level facial semantics."""

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
        x = self.stem(x)               # (B, 128, 80, 80)
        p3 = self.p3_branch(x)         # (B, 128, 80, 80)
        p4 = self.p4_branch(p3)        # (B, 256, 40, 40)
        p5 = self.p5_branch(p4)        # (B, 512, 20, 20)
        return [p3, p4, p5]


class FusionModule(nn.Module):
    """Combines geometric and semantic features (kept for compatibility, unused in split-loss mode)."""

    def __init__(self, channels_list: Optional[List[int]] = None) -> None:
        super().__init__()
        if channels_list is None:
            channels_list = [128, 256, 512]
        self.fuse3 = nn.Conv2d(channels_list[0] * 2, channels_list[0], 1)
        self.fuse4 = nn.Conv2d(channels_list[1] * 2, channels_list[1], 1)
        self.fuse5 = nn.Conv2d(channels_list[2] * 2, channels_list[2], 1)
        self._fuse_layers = [self.fuse3, self.fuse4, self.fuse5]

    def forward(
        self, geo_feats: List[torch.Tensor], sem_feats: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [
            self._fuse_layers[i](torch.cat([g, s], dim=1))
            for i, (g, s) in enumerate(zip(geo_feats, sem_feats))
        ]


# ===================== PROJECTION HEAD =====================

class ProjectionHead(nn.Module):
    """1×1 conv MLP projecting features to proj_out-dim embeddings for contrastive loss."""

    def __init__(self, in_ch: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== CONTRASTIVE LOSS =====================

def sampled_contrastive_loss(
    noisy_feats: List[torch.Tensor],
    clean_feats: List[torch.Tensor],
    num_samples: int = 1024,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Memory-efficient spatial InfoNCE loss averaged over P3, P4, P5.

    For each scale: samples num_samples spatial positions, builds a
    (B*N) x (B*N) similarity matrix, and applies cross-entropy with
    the diagonal as positives (matching position, same image).
    """
    device = noisy_feats[0].device
    total_loss = torch.tensor(0.0, device=device)
    num_scales = len(noisy_feats)

    for fn, fc in zip(noisy_feats, clean_feats):
        B, C, H, W = fn.shape
        N = H * W

        # L2-normalise along channel dim
        fn = F.normalize(fn.view(B, C, -1), dim=1)   # (B, C, N)
        fc = F.normalize(fc.view(B, C, -1), dim=1)   # (B, C, N)

        # Random spatial sampling for memory efficiency
        n_samples = min(num_samples, N)
        idx = torch.randperm(N, device=device)[:n_samples]

        fn_sampled = fn[:, :, idx]   # (B, C, n_samples)
        fc_sampled = fc[:, :, idx]   # (B, C, n_samples)

        # Reshape to (B*n_samples, C) for batch similarity matrix
        fn_all = fn_sampled.permute(0, 2, 1).reshape(B * n_samples, C)
        fc_all = fc_sampled.permute(0, 2, 1).reshape(B * n_samples, C)

        # Full similarity matrix: diagonal = positives
        logits = torch.matmul(fn_all, fc_all.T) / temperature   # (B*n, B*n)
        labels = torch.arange(B * n_samples, device=device)

        total_loss = total_loss + F.cross_entropy(logits, labels)

    return total_loss / num_scales


# ===================== MULTI-HOOK =====================

class MultiHook(nn.Module):
    """
    Captures YOLO neck outputs at layers 16 (P3), 19 (P4), 22 (P5) via forward hooks.
    Computes two independent spatial InfoNCE losses:
      - L_geo: noisy neck vs clean-geometric encoder
      - L_sem: noisy neck vs clean-semantic encoder

    Both losses share proj_neck (the noisy-side projection head).
    Gradient-conflict diagnostics confirmed this sharing is fine:
    geo and sem gradients on proj_neck are orthogonal (cosine ≈ 0),
    not opposing — so no architectural split is needed.
    """

    def __init__(
        self,
        neck_channels: Optional[List[int]] = None,
        proj_hidden: int = 128,
        proj_out: int = 64,
        use_geometric: bool = True,
        use_semantic: bool = True,
    ) -> None:
        super().__init__()
        if neck_channels is None:
            neck_channels = [128, 256, 512]

        self.neck_channels = neck_channels
        self.use_geometric = use_geometric
        self.use_semantic = use_semantic
        self.clean_images: Optional[torch.Tensor] = None
        self.features: Dict[int, torch.Tensor] = {}

        # Shared noisy-side projection head (one per scale)
        self.proj_neck = nn.ModuleList([
            ProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
        ])

        self.geo_encoder: Optional[GeometricEncoder] = None
        self.sem_encoder: Optional[SemanticEncoder] = None
        self.proj_clean_geo: Optional[nn.ModuleList] = None
        self.proj_clean_sem: Optional[nn.ModuleList] = None
        self.fusion: Optional[FusionModule] = None

        if use_geometric:
            self.geo_encoder = GeometricEncoder(out_channels_list=neck_channels)
            self.proj_clean_geo = nn.ModuleList([
                ProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
            ])
        if use_semantic:
            self.sem_encoder = SemanticEncoder(out_channels_list=neck_channels)
            self.proj_clean_sem = nn.ModuleList([
                ProjectionHead(ch, proj_hidden, proj_out) for ch in neck_channels
            ])
        if use_geometric and use_semantic:
            self.fusion = FusionModule(neck_channels)

    # ── Hook registration targets ────────────────────────────────────────────

    def set_clean(self, clean_imgs: torch.Tensor) -> None:
        self.clean_images = clean_imgs

    def hook_p3(self, module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        self.features[16] = output

    def hook_p4(self, module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        self.features[19] = output

    def hook_p5(self, module: nn.Module, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        self.features[22] = output

    def get_neck_outputs(self) -> List[torch.Tensor]:
        return [self.features[16], self.features[19], self.features[22]]

    # ── Loss computation ─────────────────────────────────────────────────────

    def compute_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (geo_loss, sem_loss).
        Each is 0.0 if the corresponding encoder is disabled.
        Clears the captured feature cache after use.
        """
        device = self.proj_neck[0].net[0].weight.device
        zero = torch.tensor(0.0, device=device)

        if self.clean_images is None or len(self.features) < 3:
            return zero, zero

        neck_outs = self.get_neck_outputs()
        geo_loss = zero
        sem_loss = zero

        # Geometric branch: noisy neck vs clean-edge encoder
        if self.use_geometric and self.geo_encoder is not None and self.proj_clean_geo is not None:
            geo_feats = self.geo_encoder(self.clean_images)
            noisy_proj = [proj(f) for proj, f in zip(self.proj_neck, neck_outs)]
            clean_proj = [proj(f) for proj, f in zip(self.proj_clean_geo, geo_feats)]
            geo_loss = sampled_contrastive_loss(noisy_proj, clean_proj)

        # Semantic branch: noisy neck vs clean-semantic encoder
        if self.use_semantic and self.sem_encoder is not None and self.proj_clean_sem is not None:
            sem_feats = self.sem_encoder(self.clean_images)
            noisy_proj = [proj(f) for proj, f in zip(self.proj_neck, neck_outs)]
            clean_proj = [proj(f) for proj, f in zip(self.proj_clean_sem, sem_feats)]
            sem_loss = sampled_contrastive_loss(noisy_proj, clean_proj)

        self.features = {}   # clear cache
        return geo_loss, sem_loss

    # ── Parameter counting ───────────────────────────────────────────────────

    def get_auxiliary_param_count(self) -> int:
        """
        Returns total trainable parameter count for all auxiliary modules,
        including proj_neck (training-only noisy-side projection heads).
        """
        count = 0
        for mod in [
            self.proj_neck,        # noisy-side projection — training only
            self.geo_encoder,
            self.sem_encoder,
            self.proj_clean_geo,
            self.proj_clean_sem,
            self.fusion,
        ]:
            if mod is not None:
                count += sum(p.numel() for p in mod.parameters())
        return count


# ===================== ADAPTIVE LOSS WEIGHTER =====================

class AdaptiveLossWeighter(nn.Module):
    """
    Canonical Kendall et al. homoscedastic uncertainty weighting.

    Formula (per task i):
        L_weighted_i = exp(-log_var_i) * L_i + log_var_i

    where log_var_i = log(sigma_i^2) is a learned parameter.

    The precision term exp(-log_var) up-weights tasks with low uncertainty
    (small sigma), and the regularisation term log_var prevents sigma from
    going to infinity (which would trivially minimise the precision term).

    Key design decision: raw losses are passed directly — no pre-normalisation.
    The scale difference between geo and sem losses IS the signal that log_var
    needs to adapt to. Pre-normalising would destroy that signal and leave
    log_var with nothing meaningful to learn.

    A linear warm-up ramp scales the entire auxiliary contribution from 0→1
    over the first warmup_epochs epochs, allowing the backbone to stabilise
    on the detection objective before contrastive regularisation kicks in.
    """

    def __init__(
        self,
        num_tasks: int = 2,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        init_log_var: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.full((num_tasks,), init_log_var))
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def forward(self, losses: List[torch.Tensor], epoch: int) -> torch.Tensor:
        """
        Args:
            losses : list of scalar loss tensors, e.g. [geo_loss, sem_loss]
            epoch  : current epoch (0-indexed)
        Returns:
            Scalar weighted combination with warm-up ramp applied.
        """
        precision = torch.exp(-self.log_var)
        total = sum(
            precision[i] * losses[i] + self.log_var[i]
            for i in range(len(losses))
        )

        ramp = epoch / max(1, self.warmup_epochs) if epoch < self.warmup_epochs else 1.0
        return ramp * total