#!/usr/bin/env python3
"""
CFDN-YOLO Ablation Study — Barlow Twins variant (v10.4).

Changes vs v10.3:
  - Fixed: torch.isfinite(loss) now uses .all() because Ultralytics returns
    a per-element loss tensor, not a scalar. Without .all(), Python raises
    "Boolean value of Tensor with more than one value is ambiguous".
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import sys
import types
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).parent))

from contrastive_denoise_yolo26s import MultiHook, AdaptiveLossWeighter


# ===================== CONFIG =====================

@dataclass
class AblationConfig:
    use_geometric:            bool            = True
    use_semantic:             bool            = True
    lambda_geo:               float           = 0.1
    lambda_sem:               float           = 0.1
    adaptive_weights:         bool            = False
    warmup_epochs:            int             = 10
    neck_channels:            Tuple[int, ...] = (128, 256, 512)
    proj_hidden:              int             = 512
    proj_out:                 int             = 2048
    feature_noise_std_scale:  float           = 1.0
    lambda_barlow_off:        float           = 0.005
    aux_lr:                   float           = 3e-3
    run_name:                 str             = "cfdn_barlow"
    seed:                     int             = 42
    model_variant:            str             = "yolo26s.pt"
    data_yaml:                str             = "widerface.yaml"
    epochs:                   int             = 100
    batch_size:               int             = 8
    img_size:                 int             = 640

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_run_description(self) -> str:
        if not self.use_geometric and not self.use_semantic:
            return "Baseline (clean training, no CFDN modules)"
        if self.use_geometric and not self.use_semantic:
            return "Geometric Encoder Only"
        if not self.use_geometric and self.use_semantic:
            return "Semantic Encoder Only"
        return f"Barlow Twins (Geo + Sem, λ={self.lambda_geo}/{self.lambda_sem})"


# ===================== OOM-SAFE EMBEDDING GAP (external helper) =====================

def compute_embedding_gap(
    hook: MultiHook,
    clean_neck: List[torch.Tensor],
    num_samples: int = 256,
) -> Tuple[List[float], List[float]]:
    geo_gaps, sem_gaps = [], []
    device = clean_neck[0].device

    with torch.no_grad():
        noisy_neck = []
        for f in clean_neck:
            f32 = f.detach().float()
            std = f32.std().clamp(max=10.0)
            noisy_neck.append(f32 + torch.randn_like(f32) * (std + 1e-6) * 1.0)

        def gap_for(noisy_list, clean_list):
            gaps = []
            for zn, zc in zip(noisy_list, clean_list):
                B, D, H, W = zn.shape
                N = H * W
                S = min(num_samples, N)

                zn_flat = zn.permute(0, 2, 3, 1).reshape(B, N, D).float()
                zc_flat = zc.permute(0, 2, 3, 1).reshape(B, N, D).float()

                idx = torch.randperm(N, device=device)[:S]
                zn_s = zn_flat[:, idx, :].reshape(B * S, D)
                zc_s = zc_flat[:, idx, :].reshape(B * S, D)

                zn_n = torch.nn.functional.normalize(zn_s, dim=1, eps=1e-8)
                zc_n = torch.nn.functional.normalize(zc_s, dim=1, eps=1e-8)

                sim = zn_n @ zc_n.T
                labels = torch.arange(B * S, device=device)
                pos_mean = sim[labels, labels].mean().item()
                mask = ~torch.eye(B * S, dtype=torch.bool, device=device)
                neg_mean = sim[mask].mean().item()
                gaps.append(pos_mean - neg_mean)
            return gaps

        if hook.use_geometric and hook.geo_encoder is not None and hook.proj_clean_geo is not None:
            geo_feats = hook.geo_encoder(hook.clean_images)
            geo_noisy = [p(f) for p, f in zip(hook.proj_neck, noisy_neck)]
            geo_clean = [p(f) for p, f in zip(hook.proj_clean_geo, geo_feats)]
            geo_gaps = gap_for(geo_noisy, geo_clean)

        if hook.use_semantic and hook.sem_encoder is not None and hook.proj_clean_sem is not None:
            sem_feats = hook.sem_encoder(hook.clean_images)
            sem_noisy = [p(f) for p, f in zip(hook.proj_neck, noisy_neck)]
            sem_clean = [p(f) for p, f in zip(hook.proj_clean_sem, sem_feats)]
            sem_gaps = gap_for(sem_noisy, sem_clean)

    return geo_gaps, sem_gaps


# ===================== CUSTOM TRAINER =====================

class CFDNTrainer(DetectionTrainer):

    def __init__(self, config: AblationConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfdn_config       = config
        self.hook:             Optional[MultiHook]             = None
        self.weighter:         Optional[AdaptiveLossWeighter]  = None
        self.aux_optimizer:    Optional[torch.optim.Optimizer] = None
        self.cont_losses_geo:  List[float] = []
        self.cont_losses_sem:  List[float] = []
        self.batch_count:      int = 0
        self._original_forward = None
        self._custom_forward   = None

    def setup_model(self) -> None:
        super().setup_model()
        self._attach_cfdn_modules()
        self._patch_forward()
        self.add_callback('on_train_start',     self.on_train_start)
        self.add_callback('on_train_batch_end', self.on_train_batch_end)

    def _attach_cfdn_modules(self) -> None:
        device = next(self.model.parameters()).device
        config = self.cfdn_config

        self.hook = MultiHook(
            neck_channels=list(config.neck_channels),
            proj_hidden=config.proj_hidden,
            proj_out=config.proj_out,
            use_geometric=config.use_geometric,
            use_semantic=config.use_semantic,
        ).to(device)

        if config.adaptive_weights and (config.use_geometric or config.use_semantic):
            self.weighter = AdaptiveLossWeighter(
                num_tasks=2,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epochs,
            ).to(device)

        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        if not isinstance(det_model, DetectionModel):
            raise RuntimeError("Expected DetectionModel")

        layers = det_model.model
        layers[16].register_forward_hook(self.hook.hook_p3)
        layers[19].register_forward_hook(self.hook.hook_p4)
        layers[22].register_forward_hook(self.hook.hook_p5)

        print(f"  [CFDN] Modules attached. Aux params: {self.hook.get_auxiliary_param_count():,}")
        print(f"  [CFDN] Feature noise scale: std * {config.feature_noise_std_scale}")
        print(f"  [CFDN] Barlow Twins: proj_out={config.proj_out}, "
              f"lambda_off={config.lambda_barlow_off}, aux_lr={config.aux_lr}")
        print(f"  [CFDN] Gradient clipping: main=max_norm 10.0, aux=max_norm 1.0")

    def build_optimizer(self, model, name='auto', lr=0.01, momentum=0.9, decay=0.0, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        if self.hook is not None:
            aux_params = list(self.hook.parameters())
            if self.weighter is not None:
                aux_params += list(self.weighter.parameters())
            if aux_params:
                self.aux_optimizer = torch.optim.AdamW(
                    aux_params,
                    lr=self.cfdn_config.aux_lr,
                    weight_decay=0.0,
                )
                print(f"  [CFDN] Aux AdamW lr={self.cfdn_config.aux_lr}: "
                      f"{len(aux_params)} param tensors.")
        return optimizer

    def _patch_forward(self) -> None:
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        original_forward = det_model.forward
        self._original_forward = original_forward
        config = self.cfdn_config

        def custom_forward(model_self, batch, *args, **kwargs):
            is_training = (
                isinstance(batch, dict)
                and 'img' in batch
                and 'bboxes' in batch
                and model_self.training
            )
            if not is_training:
                if isinstance(batch, dict) and 'img' in batch:
                    batch = batch['img']
                if isinstance(batch, torch.Tensor) and batch.dtype == torch.float16:
                    batch = batch.float()
                return original_forward(batch, *args, **kwargs)

            self.hook.set_clean(batch['img'])
            loss, loss_items = original_forward(batch, *args, **kwargs)

            # Guard against a non-finite detection loss (loss is a per-element tensor)
            if not torch.isfinite(loss).all():
                return loss, loss_items

            if config.use_geometric or config.use_semantic:
                with torch.amp.autocast('cuda', enabled=False):
                    clean_neck = self.hook.get_neck_outputs()

                    noisy_neck = []
                    for f in clean_neck:
                        f_fp32 = f.detach().float()
                        std = f_fp32.std().clamp(max=10.0)
                        noisy_neck.append(f_fp32 + torch.randn_like(f_fp32) * (std + 1e-6) * config.feature_noise_std_scale)

                    self.hook.features = {}
                    geo_loss, sem_loss = self.hook.compute_loss_from_features(noisy_neck)

                    aux_loss = torch.tensor(0.0, device=loss.device)
                    if config.use_geometric:
                        aux_loss = aux_loss + config.lambda_geo * geo_loss
                    if config.use_semantic:
                        aux_loss = aux_loss + config.lambda_sem * sem_loss
                    ramp = min(1.0, self.epoch / max(1, config.warmup_epochs))
                    aux_loss = ramp * aux_loss

                    if torch.isfinite(aux_loss) and aux_loss.abs() < 1e6:
                        loss = loss + aux_loss

                if not torch.isnan(geo_loss):
                    self.cont_losses_geo.append(geo_loss.item())
                if not torch.isnan(sem_loss):
                    self.cont_losses_sem.append(sem_loss.item())

            return loss, loss_items

        det_model.forward = types.MethodType(custom_forward, det_model)
        self._custom_forward = custom_forward
        print("  [CFDN] Forward patched (Barlow Twins v10.4).")

    def optimizer_step(self) -> None:
        # ── Guard 1: skip the step if the detection loss is non-finite ──
        main_loss = getattr(self, 'loss', None)
        if main_loss is not None and not torch.isfinite(main_loss).all():
            print(f"  [CFDN] Non-finite detection loss at batch "
                  f"{self.batch_count}; skipping step.")
            self.optimizer.zero_grad()
            if self.aux_optimizer is not None:
                self.aux_optimizer.zero_grad()
            return

        aux_has_grads = False
        if self.aux_optimizer is not None:
            aux_params = [p for g in self.aux_optimizer.param_groups for p in g['params']]
            aux_has_grads = any(p.grad is not None for p in aux_params)

        # ── Unscale ──
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            if aux_has_grads:
                self.scaler.unscale_(self.aux_optimizer)

        # ── Guard 2: clip main-optimizer gradients (all groups) ──
        main_params = [p for g in self.optimizer.param_groups for p in g['params']]
        torch.nn.utils.clip_grad_norm_(main_params, max_norm=10.0)

        # ── Guard 3: clip aux-optimizer gradients ──
        if aux_has_grads:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.aux_optimizer.param_groups for p in g['params']],
                max_norm=1.0,
            )

        # ── Step ──
        if self.scaler:
            self.scaler.step(self.optimizer)
            if aux_has_grads:
                self.scaler.step(self.aux_optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            if aux_has_grads:
                self.aux_optimizer.step()

        self.optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()

    def on_train_start(self, trainer) -> None:
        device = torch.device(self.device if self.device else 'cuda:0')
        if self.hook is not None:
            self.hook = self.hook.to(device)
        if self.weighter is not None:
            self.weighter = self.weighter.to(device)

    def on_train_batch_end(self, trainer) -> None:
        self.batch_count += 1
        if self.batch_count % 100 == 0:
            avg_geo = float(np.mean(self.cont_losses_geo[-100:])) if self.cont_losses_geo else 0.0
            avg_sem = float(np.mean(self.cont_losses_sem[-100:])) if self.cont_losses_sem else 0.0
            print(f"  [CFDN] Batch {self.batch_count}: Geo={avg_geo:.4f} | Sem={avg_sem:.4f}")

    def validate(self):
        return {}, 0.0

    def final_eval(self) -> None:
        pass

    def save_model(self) -> None:
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        had_custom = 'forward' in det_model.__dict__
        if had_custom:
            del det_model.__dict__['forward']

        ema_model = None
        had_ema_custom = False
        if hasattr(self, 'ema') and self.ema is not None:
            ema_model = self.ema.ema
            if hasattr(ema_model, '__dict__'):
                had_ema_custom = 'forward' in ema_model.__dict__
                if had_ema_custom:
                    del ema_model.__dict__['forward']

        try:
            super().save_model()
            if self.hook is not None:
                aux_state = {'proj_neck': self.hook.proj_neck.state_dict()}
                if self.hook.geo_encoder is not None:
                    aux_state['geo_encoder'] = self.hook.geo_encoder.state_dict()
                if self.hook.sem_encoder is not None:
                    aux_state['sem_encoder'] = self.hook.sem_encoder.state_dict()
                if self.hook.proj_clean_geo is not None:
                    aux_state['proj_clean_geo'] = self.hook.proj_clean_geo.state_dict()
                if self.hook.proj_clean_sem is not None:
                    aux_state['proj_clean_sem'] = self.hook.proj_clean_sem.state_dict()
                torch.save(aux_state, Path(self.save_dir) / "aux_modules.pt")
        finally:
            if had_custom:
                det_model.forward = types.MethodType(self._custom_forward, det_model)
            if had_ema_custom and ema_model is not None:
                ema_model.forward = types.MethodType(self._custom_forward, ema_model)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "contrastive_loss_active": bool(self.cont_losses_geo or self.cont_losses_sem),
            "mean_geo_loss": float(np.mean(self.cont_losses_geo)) if self.cont_losses_geo else 0.0,
            "mean_sem_loss": float(np.mean(self.cont_losses_sem)) if self.cont_losses_sem else 0.0,
            "adaptive_weights": self.cfdn_config.adaptive_weights,
            "final_log_var": None,
        }


# ===================== TRAINING FUNCTION =====================

def train_ablation(config: AblationConfig) -> str:
    print(f"\n{'='*70}")
    print(f"  {config.run_name}: {config.get_run_description()}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"./runs/ablation/{config.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    trainer = CFDNTrainer(
        config=config,
        overrides={
            'model':    config.model_variant,
            'data':     config.data_yaml,
            'epochs':   config.epochs,
            'batch':    config.batch_size,
            'imgsz':    config.img_size,
            'device':   'cuda:0' if torch.cuda.is_available() else 'cpu',
            'amp':      True,
            'workers':  4,
            'project':  str(output_dir.absolute()),
            'name':     'weights',
            'exist_ok': True,
            'verbose':  False,
            'seed':     config.seed,
            'val':      False,
        }
    )

    trainer.train()

    summary = {
        "run_name":         config.run_name,
        "description":      config.get_run_description(),
        "config":           config.to_dict(),
        "contrastive_loss": trainer.get_summary(),
        "auxiliary_params": trainer.hook.get_auxiliary_param_count() if trainer.hook else 0,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done: {output_dir}")
    return str(output_dir)


# ===================== ENTRY POINT =====================

def main() -> None:
    parser = argparse.ArgumentParser(description="CFDN-YOLO Barlow Twins v10.4")
    parser.add_argument('--use_geometric',
                        type=lambda x: x.lower() in ('true','1','yes'), default=True)
    parser.add_argument('--use_semantic',
                        type=lambda x: x.lower() in ('true','1','yes'), default=True)
    parser.add_argument('--lambda_geo',              type=float, default=0.1)
    parser.add_argument('--lambda_sem',              type=float, default=0.1)
    parser.add_argument('--adaptive_weights',
                        type=lambda x: x.lower() in ('true','1','yes'), default=False)
    parser.add_argument('--warmup_epochs',           type=int,   default=10)
    parser.add_argument('--proj_hidden',             type=int,   default=512)
    parser.add_argument('--proj_out',                type=int,   default=2048)
    parser.add_argument('--feature_noise_std_scale', type=float, default=1.0)
    parser.add_argument('--lambda_barlow_off',       type=float, default=0.005)
    parser.add_argument('--aux_lr',                  type=float, default=3e-3)
    parser.add_argument('--model',                   type=str,   default='yolo26s.pt')
    parser.add_argument('--data',                    type=str,   default='widerface.yaml')
    parser.add_argument('--epochs',                  type=int,   default=100)
    parser.add_argument('--batch_size',              type=int,   default=8)
    parser.add_argument('--img_size',                type=int,   default=640)
    parser.add_argument('--run_name',                type=str,   default=None)
    parser.add_argument('--seed',                    type=int,   default=42)

    args = parser.parse_args()

    if args.run_name is None:
        if not args.use_geometric and not args.use_semantic:
            args.run_name = "baseline"
        elif args.use_geometric and not args.use_semantic:
            args.run_name = "geo_only_barlow"
        elif not args.use_geometric and args.use_semantic:
            args.run_name = "sem_only_barlow"
        else:
            args.run_name = "cfdn_barlow"

    config = AblationConfig(
        use_geometric           = args.use_geometric,
        use_semantic            = args.use_semantic,
        run_name                = args.run_name,
        lambda_geo              = args.lambda_geo,
        lambda_sem              = args.lambda_sem,
        adaptive_weights        = args.adaptive_weights,
        warmup_epochs           = args.warmup_epochs,
        proj_hidden             = args.proj_hidden,
        proj_out                = args.proj_out,
        feature_noise_std_scale = args.feature_noise_std_scale,
        lambda_barlow_off       = args.lambda_barlow_off,
        aux_lr                  = args.aux_lr,
        model_variant           = args.model,
        data_yaml               = args.data,
        epochs                  = args.epochs,
        batch_size              = args.batch_size,
        img_size                = args.img_size,
        seed                    = args.seed,
    )

    os.makedirs("./runs/ablation", exist_ok=True)
    train_ablation(config)


if __name__ == '__main__':
    main()