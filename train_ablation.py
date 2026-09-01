#!/usr/bin/env python3
"""
CFDN‑YOLO Ablation Study – final custom trainer (validation OFF).
Auxiliary modules live on the trainer; training path activated only when
'bboxes' is in the batch. Validation and final_eval are skipped entirely.
Gradient-conflict diagnostic runs every 50 batches.
Uses a separate AdamW optimizer for auxiliary modules to avoid MuSGD incompatibility.
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
from utils.noise import add_noise, NoiseConfig


# ===================== CONFIG =====================

@dataclass
class AblationConfig:
    use_geometric: bool = True
    use_semantic: bool = True
    lambda_geo: float = 0.25
    lambda_sem: float = 0.25
    adaptive_weights: bool = True
    warmup_epochs: int = 10
    temperature: float = 0.07
    num_samples: int = 1024
    neck_channels: Tuple[int, ...] = (128, 256, 512)
    proj_hidden: int = 128
    proj_out: int = 64
    noise_types: Tuple[str, ...] = ('gaussian',)
    noise_params: Tuple[int, ...] = (10, 25, 50)
    run_name: str = "full_model"
    seed: int = 42
    model_variant: str = "yolo26s.pt"
    data_yaml: str = "widerface.yaml"
    epochs: int = 100
    batch_size: int = 8
    img_size: int = 640

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================== CUSTOM TRAINER =====================

class CFDNTrainer(DetectionTrainer):
    """
    Custom trainer that holds auxiliary modules as trainer attributes.
    Their parameters are updated by a separate AdamW optimizer to avoid
    incompatibility with the main MuSGD optimizer.
    Validation is skipped entirely during training – evaluate separately afterwards.
    """

    def __init__(self, config: AblationConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfdn_config = config
        self.hook: Optional[MultiHook] = None
        self.weighter: Optional[AdaptiveLossWeighter] = None
        self.aux_optimizer: Optional[torch.optim.Optimizer] = None
        self.cont_losses_geo: List[float] = []
        self.cont_losses_sem: List[float] = []
        self.batch_count: int = 0
        self._original_forward = None
        self._custom_forward = None

    def setup_model(self):
        """Load the YOLO model, create aux modules, patch forward, register callbacks."""
        super().setup_model()
        self._attach_cfdn_modules()
        self._patch_forward()
        self.add_callback('on_train_batch_end', self.on_train_batch_end)
        self.add_callback('on_train_start', self.on_train_start)

    def _attach_cfdn_modules(self):
        """Create aux modules on the trainer (not on model) and register neck hooks."""
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
                total_epochs=config.epochs
            ).to(device)

        # Register forward hooks on the inner model's neck layers
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        if isinstance(det_model, DetectionModel):
            model_layers = det_model.model
            model_layers[16].register_forward_hook(self.hook.hook_p3)
            model_layers[19].register_forward_hook(self.hook.hook_p4)
            model_layers[22].register_forward_hook(self.hook.hook_p5)
        else:
            raise RuntimeError("Unexpected model structure")

        print(f"  [CFDN] Auxiliary modules created. Params: {self.hook.get_auxiliary_param_count():,}")

    def build_optimizer(self, model, name='auto', lr=0.01, momentum=0.9, decay=0.0, iterations=1e5):
        """Build the main optimizer (MuSGD) for YOLO, and a separate AdamW for auxiliary modules."""
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)

        # Create a separate AdamW optimizer for aux modules
        if self.hook is not None:
            aux_params = list(self.hook.parameters())
            if self.weighter is not None:
                aux_params += list(self.weighter.parameters())
            if aux_params:
                self.aux_optimizer = torch.optim.AdamW(
                    aux_params, lr=1e-3, weight_decay=0.0
                )
                print(f"  [CFDN] Separate aux optimizer created with {len(aux_params)} param tensors.")
            else:
                self.aux_optimizer = None
        else:
            self.aux_optimizer = None

        return optimizer

    def _patch_forward(self):
        """Patch the DetectionModel's forward to inject noise and add contrastive loss."""
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        original_forward = det_model.forward
        self._original_forward = original_forward
        config = self.cfdn_config
        noise_cfg = NoiseConfig(config.noise_types, config.noise_params)

        def custom_forward(model_self, batch, *args, **kwargs):
            is_training = (isinstance(batch, dict) and 'img' in batch
                           and model_self.training and 'bboxes' in batch)

            if is_training:
                clean_img = batch['img'].clone()
                self.hook.set_clean(clean_img)

                noise_type = str(np.random.choice(list(noise_cfg.noise_types)))
                noise_param = float(np.random.choice(list(noise_cfg.noise_params)))
                batch['img'] = add_noise(batch['img'], noise_type=noise_type, param=noise_param)

                loss, loss_items = original_forward(batch, *args, **kwargs)

                geo_loss, sem_loss = self.hook.compute_loss()

                # ---- Gradient-conflict diagnostic (every 50 batches) ----
                if self.batch_count % 50 == 0 and config.use_geometric and config.use_semantic:
                    proj_params = list(self.hook.proj_neck.parameters())
                    geo_grads = torch.autograd.grad(
                        geo_loss, proj_params, retain_graph=True, allow_unused=True
                    )
                    sem_grads = torch.autograd.grad(
                        sem_loss, proj_params, retain_graph=True, allow_unused=True
                    )
                    geo_vec = torch.cat([
                        g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
                        for p, g in zip(proj_params, geo_grads)
                    ])
                    sem_vec = torch.cat([
                        g.view(-1) if g is not None else torch.zeros_like(p).view(-1)
                        for p, g in zip(proj_params, sem_grads)
                    ])
                    cos_sim = torch.nn.functional.cosine_similarity(
                        geo_vec.unsqueeze(0), sem_vec.unsqueeze(0)
                    ).item()
                    geo_norm = geo_vec.norm().item()
                    sem_norm = sem_vec.norm().item()
                    print(f"[GRAD CONFLICT] batch={self.batch_count} "
                          f"cosine={cos_sim:.4f} "
                          f"|geo|={geo_norm:.4f} |sem|={sem_norm:.4f}")

                if config.use_geometric or config.use_semantic:
                    if self.weighter is not None:
                        epoch = self.epoch
                        task_losses = [
                            geo_loss if config.use_geometric else torch.tensor(0.0, device=geo_loss.device),
                            sem_loss if config.use_semantic else torch.tensor(0.0, device=geo_loss.device)
                        ]
                        aux_loss = self.weighter(task_losses, epoch)
                    else:
                        aux_loss = 0.0
                        if config.use_geometric:
                            aux_loss = aux_loss + config.lambda_geo * geo_loss
                        if config.use_semantic:
                            aux_loss = aux_loss + config.lambda_sem * sem_loss
                    loss = loss + aux_loss

                if geo_loss.item() > 0:
                    self.cont_losses_geo.append(geo_loss.item())
                if sem_loss.item() > 0:
                    self.cont_losses_sem.append(sem_loss.item())

                return loss, loss_items
            else:
                if isinstance(batch, dict) and 'img' in batch:
                    batch = batch['img']
                if isinstance(batch, torch.Tensor) and batch.dtype == torch.float16:
                    batch = batch.float()
                return original_forward(batch, *args, **kwargs)

        det_model.forward = types.MethodType(custom_forward, det_model)
        self._custom_forward = custom_forward
        print("  [CFDN] Forward patching complete.")

    def optimizer_step(self):
        """Step both the YOLO optimizer and the auxiliary optimizer."""
        # Unscale gradients for both optimizers
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            if self.aux_optimizer is not None:
                self.scaler.unscale_(self.aux_optimizer)

        # Optional gradient clipping
        clip_val = getattr(self.args, 'clip_grad', 0.0) if hasattr(self.args, 'clip_grad') else 0.0
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], clip_val)
            if self.aux_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.aux_optimizer.param_groups[0]['params'], clip_val)

        # Step both optimizers
        self.optimizer.step()
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()

        # Update the scaler once
        if self.scaler:
            self.scaler.update()

        # Zero gradients for both
        self.optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()

    def on_train_start(self, trainer):
        """Move auxiliary modules to the correct device (GPU) before training."""
        device = torch.device(self.device if self.device else 'cuda:0')
        if self.hook is not None:
            self.hook = self.hook.to(device)
        if self.weighter is not None:
            self.weighter = self.weighter.to(device)

    def on_train_batch_end(self, trainer):
        """Log contrastive losses every 100 batches."""
        self.batch_count += 1
        if self.batch_count % 100 == 0:
            avg_geo = np.mean(self.cont_losses_geo[-100:]) if self.cont_losses_geo else 0.0
            avg_sem = np.mean(self.cont_losses_sem[-100:]) if self.cont_losses_sem else 0.0
            if self.cfdn_config.use_geometric or self.cfdn_config.use_semantic:
                print(f"  [CFDN] Batch {self.batch_count}: Geo Loss = {avg_geo:.4f} | Sem Loss = {avg_sem:.4f}")

    def validate(self):
        """Skip validation entirely – evaluate separately after training."""
        return {}, 0.0

    def final_eval(self):
        """Skip final evaluation – avoid checkpoint loading issues."""
        pass

    def save_model(self):
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        had_custom = 'forward' in det_model.__dict__
        if had_custom:
            del det_model.__dict__['forward']

        # Clean EMA model (if exists)
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
        finally:
            if had_custom:
                det_model.forward = types.MethodType(self._custom_forward, det_model)
            if had_ema_custom and ema_model is not None:
                ema_model.forward = types.MethodType(self._custom_forward, ema_model)

    def get_summary(self) -> Dict[str, Any]:
        if not self.cont_losses_geo and not self.cont_losses_sem:
            return {
                "contrastive_loss_active": False,
                "mean_geo_loss": 0.0,
                "mean_sem_loss": 0.0,
                "adaptive_weights": self.cfdn_config.adaptive_weights,
                "final_log_var": None,
            }
        final_log_var = None
        if self.weighter is not None:
            final_log_var = self.weighter.log_var.detach().cpu().tolist()
        return {
            "contrastive_loss_active": True,
            "mean_geo_loss": float(np.mean(self.cont_losses_geo)) if self.cont_losses_geo else 0.0,
            "mean_sem_loss": float(np.mean(self.cont_losses_sem)) if self.cont_losses_sem else 0.0,
            "adaptive_weights": self.cfdn_config.adaptive_weights,
            "final_log_var": final_log_var,
        }


# ===================== TRAINING FUNCTION =====================

def train_ablation(config: AblationConfig) -> str:
    print(f"\n{'='*70}")
    print(f"  ABLATION EXPERIMENT: {config.run_name}")
    print(f"  Geometric: {config.use_geometric} | Semantic: {config.use_semantic}")
    if config.adaptive_weights:
        print(f"  Adaptive weights: ON (warmup {config.warmup_epochs} epochs)")
    else:
        print(f"  λ_geo: {config.lambda_geo} | λ_sem: {config.lambda_sem}")
    print(f"  Seed: {config.seed}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"./runs/ablation/{config.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(str(config_path), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    trainer = CFDNTrainer(
        config=config,
        overrides={
            'model': config.model_variant,
            'data': config.data_yaml,
            'epochs': config.epochs,
            'batch': config.batch_size,
            'imgsz': config.img_size,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'amp': True,
            'workers': 4,
            'project': str(output_dir.absolute()),
            'name': 'weights',
            'exist_ok': True,
            'verbose': False,
            'seed': config.seed,
            'val': False,
        }
    )
    trainer.train()

    summary = {
        "run_name": config.run_name,
        "config": config.to_dict(),
        "contrastive_loss": trainer.get_summary(),
        "auxiliary_params": trainer.hook.get_auxiliary_param_count() if trainer.hook else 0,
    }
    summary_path = output_dir / "summary.json"
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)

    cont_summary = trainer.get_summary()
    print(f"\n  Training Summary: {config.run_name}")
    print(f"  Contrastive Loss Active: {cont_summary['contrastive_loss_active']}")
    if cont_summary['contrastive_loss_active']:
        print(f"  Mean Geo Loss: {cont_summary['mean_geo_loss']:.4f}")
        print(f"  Mean Sem Loss: {cont_summary['mean_sem_loss']:.4f}")
        if cont_summary.get('adaptive_weights') and cont_summary.get('final_log_var'):
            print(f"  Final log variances: {cont_summary['final_log_var']}")
    print(f"  Output: {output_dir}")
    return str(output_dir)


# ===================== RUNNERS =====================

def run_all_ablations(args: argparse.Namespace) -> List[str]:
    configs = [
        {"use_geometric": False, "use_semantic": False, "run_name": "baseline"},
        {"use_geometric": True,  "use_semantic": False, "run_name": "geo_only"},
        {"use_geometric": False, "use_semantic": True,  "run_name": "sem_only"},
        {"use_geometric": True,  "use_semantic": True,  "run_name": "full_model_adaptive"},
    ]
    output_dirs = []
    for cfg_dict in configs:
        config = AblationConfig(
            use_geometric=bool(cfg_dict['use_geometric']),
            use_semantic=bool(cfg_dict['use_semantic']),
            run_name=str(cfg_dict['run_name']),
            lambda_geo=float(args.lambda_geo),
            lambda_sem=float(args.lambda_sem),
            adaptive_weights=args.adaptive_weights,
            warmup_epochs=int(args.warmup_epochs),
            temperature=float(args.temperature),
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            model_variant=str(args.model),
            data_yaml=str(args.data),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            img_size=int(args.img_size),
        )
        output_dirs.append(train_ablation(config))
    return output_dirs


def run_single_ablation(args: argparse.Namespace) -> str:
    use_geo = str(args.use_geometric).lower() in ('true', '1', 'yes')
    use_sem = str(args.use_semantic).lower() in ('true', '1', 'yes')
    if args.run_name is None:
        if not use_geo and not use_sem:
            args.run_name = "baseline"
        elif use_geo and not use_sem:
            args.run_name = "geo_only"
        elif not use_geo and use_sem:
            args.run_name = "sem_only"
        elif args.adaptive_weights:
            args.run_name = "full_model_adaptive"
        else:
            args.run_name = "full_model_split"

    config = AblationConfig(
        use_geometric=use_geo,
        use_semantic=use_sem,
        run_name=str(args.run_name),
        lambda_geo=float(args.lambda_geo),
        lambda_sem=float(args.lambda_sem),
        adaptive_weights=args.adaptive_weights,
        warmup_epochs=int(args.warmup_epochs),
        temperature=float(args.temperature),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        model_variant=str(args.model),
        data_yaml=str(args.data),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        img_size=int(args.img_size),
    )
    return train_ablation(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="CFDN-YOLO Ablation Study (final)")
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--use_geometric', type=str, default='True')
    parser.add_argument('--use_semantic', type=str, default='True')
    parser.add_argument('--lambda_geo', type=float, default=0.25)
    parser.add_argument('--lambda_sem', type=float, default=0.25)
    parser.add_argument('--adaptive_weights', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--model', type=str, default='yolo26s.pt')
    parser.add_argument('--data', type=str, default='widerface.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  CFDN-YOLO ABLATION STUDY (final)")
    print(f"  Model: {args.model} | Dataset: {args.data}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    os.makedirs("./runs/ablation", exist_ok=True)

    if args.all:
        output_dirs = run_all_ablations(args)
        print(f"\n{'='*70}")
        print("  ALL ABLATION RUNS COMPLETE")
        for d in output_dirs:
            print(f"    {d}")
        print(f"{'='*70}\n")
    else:
        output_dir = run_single_ablation(args)
        print(f"\n  Run complete: {output_dir}\n")


if __name__ == '__main__':
    main()