#!/usr/bin/env python3
"""
CFDN-YOLO Ablation Study — final custom trainer (validation OFF).

Two-stream training design:
  Stream 1 — Detection: clean images → full YOLO forward → detection loss.
  Stream 2 — Contrastive: noisy images → backbone+neck (eval mode, grads on)
             → noisy neck features → InfoNCE loss vs clean encoder targets.

The gradient-conflict diagnostic has been removed. Extensive measurements
from previous runs confirmed cosine(geo_grad, sem_grad) ≈ 0 throughout
training — the two losses are orthogonal on proj_neck and no conflict exists.
Removing the diagnostic eliminates 3 extra forward passes per diagnostic
batch, which was causing OOM on 22GB GPUs.

optimizer_step is conditional on aux gradient presence: when aux_loss is
NaN and skipped, aux params receive no gradients; we skip
scaler.unscale_/scaler.step for aux_optimizer in that case to avoid
the 'No inf checks recorded' AssertionError.
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
    use_geometric:    bool            = True
    use_semantic:     bool            = True
    lambda_geo:       float           = 0.1
    lambda_sem:       float           = 0.1
    adaptive_weights: bool            = False
    warmup_epochs:    int             = 10
    temperature:      float           = 0.07
    num_samples:      int             = 1024
    neck_channels:    Tuple[int, ...] = (128, 256, 512)
    proj_hidden:      int             = 128
    proj_out:         int             = 64
    noise_types:      Tuple[str, ...] = ('gaussian',)
    noise_params:     Tuple[int, ...] = (10, 25, 50)
    run_name:         str             = "full_model"
    seed:             int             = 42
    model_variant:    str             = "yolo26s.pt"
    data_yaml:        str             = "widerface.yaml"
    epochs:           int             = 100
    batch_size:       int             = 8
    img_size:         int             = 640

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_run_description(self) -> str:
        if not self.use_geometric and not self.use_semantic:
            return "Baseline (no auxiliary encoders)"
        if self.use_geometric and not self.use_semantic:
            return "Geometric Encoder Only"
        if not self.use_geometric and self.use_semantic:
            return "Semantic Encoder Only"
        if self.adaptive_weights:
            return "Full Model (Adaptive Weights — Kendall)"
        return f"Full Model (Fixed λ_geo={self.lambda_geo}, λ_sem={self.lambda_sem})"


# ===================== CUSTOM TRAINER =====================

class CFDNTrainer(DetectionTrainer):

    def __init__(self, config: AblationConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfdn_config        = config
        self.hook:              Optional[MultiHook]             = None
        self.weighter:          Optional[AdaptiveLossWeighter]  = None
        self.aux_optimizer:     Optional[torch.optim.Optimizer] = None
        self.cont_losses_geo:   List[float] = []
        self.cont_losses_sem:   List[float] = []
        self.batch_count:       int = 0
        self._original_forward  = None
        self._custom_forward    = None

    # ── Setup ────────────────────────────────────────────────────────────────

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

        print(f"  [CFDN] Auxiliary modules created. "
              f"Params: {self.hook.get_auxiliary_param_count():,}")

    def build_optimizer(
        self, model, name='auto', lr=0.01, momentum=0.9, decay=0.0, iterations=1e5
    ):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)

        if self.hook is not None:
            aux_params = list(self.hook.parameters())
            if self.weighter is not None:
                aux_params += list(self.weighter.parameters())
            if aux_params:
                self.aux_optimizer = torch.optim.AdamW(
                    aux_params, lr=1e-3, weight_decay=0.0
                )
                print(f"  [CFDN] Aux AdamW optimizer: {len(aux_params)} param tensors.")
            else:
                self.aux_optimizer = None

        return optimizer

    # ── Forward patch ─────────────────────────────────────────────────────────

    def _patch_forward(self) -> None:
        """
        Two-stream forward:

        Pass 1 — Detection (clean images):
          Standard YOLO forward. Task-aligned assigner sees clean predictions
          → stable positive assignments for all 100 epochs. Hook features from
          this pass are discarded (we want noisy features, not clean).

        Pass 2 — Contrastive (noisy images):
          model.eval() + torch.enable_grad() + noisy image forward.
          eval mode: BatchNorm uses running stats from clean training (stable).
          enable_grad: contrastive gradients flow back through backbone+neck.
          Hook captures noisy P3/P4/P5 features. Contrastive loss computed
          against clean encoder targets stored in self.hook.clean_images.

        No gradient-conflict diagnostic in this version — previous runs
        confirmed cosine ≈ 0 throughout, no conflict exists.
        """
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        original_forward = det_model.forward
        self._original_forward = original_forward
        config = self.cfdn_config
        noise_cfg = NoiseConfig(config.noise_types, config.noise_params)

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

            # ── Pass 1: Detection on CLEAN images ────────────────────────────
            clean_img = batch['img'].clone()
            self.hook.set_clean(clean_img)

            loss, loss_items = original_forward(batch, *args, **kwargs)

            # Discard clean features — we need noisy features from Pass 2
            self.hook.features.clear()

            # ── Pass 2: Noisy forward for contrastive ─────────────────────────
            if config.use_geometric or config.use_semantic:
                noise_type  = str(np.random.choice(list(noise_cfg.noise_types)))
                noise_param = float(np.random.choice(list(noise_cfg.noise_params)))
                noisy_img   = add_noise(
                    clean_img.clone(), noise_type=noise_type, param=noise_param
                )

                was_training = model_self.training
                model_self.eval()
                with torch.enable_grad():
                    model_self(noisy_img)   # hooks capture noisy P3/P4/P5
                model_self.train(was_training)

                # ── Contrastive loss ──────────────────────────────────────────
                geo_loss, sem_loss = self.hook.compute_loss()

                if self.weighter is not None:
                    task_losses = [
                        geo_loss if config.use_geometric
                        else torch.tensor(0.0, device=geo_loss.device),
                        sem_loss if config.use_semantic
                        else torch.tensor(0.0, device=sem_loss.device),
                    ]
                    aux_loss = self.weighter(task_losses, self.epoch)
                else:
                    aux_loss = torch.tensor(0.0, device=loss.device)
                    if config.use_geometric:
                        aux_loss = aux_loss + config.lambda_geo * geo_loss
                    if config.use_semantic:
                        aux_loss = aux_loss + config.lambda_sem * sem_loss
                    ramp = min(1.0, self.epoch / max(1, config.warmup_epochs))
                    aux_loss = ramp * aux_loss

                # NaN guard: 0.0 * NaN = NaN in PyTorch
                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    print(f"  [CFDN] WARNING: aux_loss NaN/Inf at batch "
                          f"{self.batch_count}, skipping")
                    # aux params will have no gradients this step —
                    # optimizer_step handles this case safely
                else:
                    loss = loss + aux_loss

                if not torch.isnan(geo_loss) and geo_loss.item() > 0:
                    self.cont_losses_geo.append(geo_loss.item())
                if not torch.isnan(sem_loss) and sem_loss.item() > 0:
                    self.cont_losses_sem.append(sem_loss.item())

            return loss, loss_items

        det_model.forward = types.MethodType(custom_forward, det_model)
        self._custom_forward = custom_forward
        print("  [CFDN] Forward patching complete (two-stream: clean detect + noisy contrastive).")

    # ── Optimizer step ────────────────────────────────────────────────────────

    def optimizer_step(self) -> None:
        """
        Step both optimizers.

        Conditional aux handling: when aux_loss is NaN and skipped,
        aux params receive no gradients. Calling scaler.unscale_() or
        scaler.step() on an optimizer with no gradients raises:
          AssertionError: No inf checks were recorded for this optimizer.
        We check for the presence of aux gradients before any scaler
        interaction with aux_optimizer, and skip the aux step entirely
        when no gradients exist. The scaler is still updated via the
        main optimizer so its state stays consistent.

        Main model always has detection gradients so it is never skipped.
        """
        # Check aux gradient presence BEFORE any unscale calls
        aux_has_grads = False
        if self.aux_optimizer is not None:
            aux_params = [p for g in self.aux_optimizer.param_groups for p in g['params']]
            aux_has_grads = any(p.grad is not None for p in aux_params)

        # Unscale — main always, aux only when it has gradients
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            if aux_has_grads:
                self.scaler.unscale_(self.aux_optimizer)

        # Clip main model gradients (respects training config)
        clip_val = (
            getattr(self.args, 'clip_grad', 0.0)
            if hasattr(self.args, 'clip_grad') else 0.0
        )
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'], clip_val
            )

        # Always clip aux gradients at max_norm=1.0 when present
        if aux_has_grads:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.aux_optimizer.param_groups for p in g['params']],
                max_norm=1.0
            )

        # Step — scaler checks inf/nan and skips if overflow detected
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

    # ── Callbacks ─────────────────────────────────────────────────────────────

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
            if self.cfdn_config.use_geometric or self.cfdn_config.use_semantic:
                print(f"  [CFDN] Batch {self.batch_count}: "
                      f"Geo Loss = {avg_geo:.4f} | Sem Loss = {avg_sem:.4f}")

    # ── Validation / eval overrides ───────────────────────────────────────────

    def validate(self):
        return {}, 0.0

    def final_eval(self) -> None:
        pass

    # ── Checkpoint saving ─────────────────────────────────────────────────────

    def save_model(self) -> None:
        """
        Strip patched forward from det_model AND EMA model before saving.
        EMA is deepcopy'd after _patch_forward() so it inherits the patch.
        """
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
        finally:
            if had_custom:
                det_model.forward = types.MethodType(self._custom_forward, det_model)
            if had_ema_custom and ema_model is not None:
                ema_model.forward = types.MethodType(self._custom_forward, ema_model)

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        has_losses = bool(self.cont_losses_geo or self.cont_losses_sem)
        final_log_var = None
        if self.weighter is not None:
            final_log_var = self.weighter.log_var.detach().cpu().tolist()
        return {
            "contrastive_loss_active": has_losses,
            "mean_geo_loss": float(np.mean(self.cont_losses_geo)) if self.cont_losses_geo else 0.0,
            "mean_sem_loss": float(np.mean(self.cont_losses_sem)) if self.cont_losses_sem else 0.0,
            "adaptive_weights": self.cfdn_config.adaptive_weights,
            "final_log_var": final_log_var,
        }


# ===================== TRAINING FUNCTION =====================

def train_ablation(config: AblationConfig) -> str:
    print(f"\n{'='*70}")
    print(f"  ABLATION EXPERIMENT: {config.run_name}")
    print(f"  {config.get_run_description()}")
    print(f"  Seed: {config.seed}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"./runs/ablation/{config.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Config saved: {output_dir / 'config.json'}")

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
        "auxiliary_params": (
            trainer.hook.get_auxiliary_param_count() if trainer.hook else 0
        ),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    cont = trainer.get_summary()
    print(f"\n  Run complete: {config.run_name}")
    print(f"  Contrastive loss active : {cont['contrastive_loss_active']}")
    if cont['contrastive_loss_active']:
        print(f"  Mean geo loss           : {cont['mean_geo_loss']:.4f}")
        print(f"  Mean sem loss           : {cont['mean_sem_loss']:.4f}")
        if cont['adaptive_weights'] and cont['final_log_var']:
            print(f"  Final log_var           : {cont['final_log_var']}")
    print(f"  Output dir              : {output_dir}")
    return str(output_dir)


# ===================== RUNNERS =====================

def run_all_ablations(args: argparse.Namespace) -> List[str]:
    arms = [
        {"use_geometric": False, "use_semantic": False, "run_name": "baseline"},
        {"use_geometric": True,  "use_semantic": False, "run_name": "geo_only"},
        {"use_geometric": False, "use_semantic": True,  "run_name": "sem_only"},
        {"use_geometric": True,  "use_semantic": True,  "run_name": "full_model"},
    ]
    output_dirs = []
    for arm in arms:
        config = AblationConfig(
            use_geometric    = bool(arm['use_geometric']),
            use_semantic     = bool(arm['use_semantic']),
            run_name         = str(arm['run_name']),
            lambda_geo       = float(args.lambda_geo),
            lambda_sem       = float(args.lambda_sem),
            adaptive_weights = args.adaptive_weights,
            warmup_epochs    = int(args.warmup_epochs),
            temperature      = float(args.temperature),
            num_samples      = int(args.num_samples),
            seed             = int(args.seed),
            model_variant    = str(args.model),
            data_yaml        = str(args.data),
            epochs           = int(args.epochs),
            batch_size       = int(args.batch_size),
            img_size         = int(args.img_size),
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
            args.run_name = "full_model_lambda01"

    config = AblationConfig(
        use_geometric    = use_geo,
        use_semantic     = use_sem,
        run_name         = str(args.run_name),
        lambda_geo       = float(args.lambda_geo),
        lambda_sem       = float(args.lambda_sem),
        adaptive_weights = args.adaptive_weights,
        warmup_epochs    = int(args.warmup_epochs),
        temperature      = float(args.temperature),
        num_samples      = int(args.num_samples),
        seed             = int(args.seed),
        model_variant    = str(args.model),
        data_yaml        = str(args.data),
        epochs           = int(args.epochs),
        batch_size       = int(args.batch_size),
        img_size         = int(args.img_size),
    )
    return train_ablation(config)


# ===================== ENTRY POINT =====================

def main() -> None:
    parser = argparse.ArgumentParser(description="CFDN-YOLO Ablation Study")
    parser.add_argument('--all',             action='store_true')
    parser.add_argument('--use_geometric',   type=str,   default='True')
    parser.add_argument('--use_semantic',    type=str,   default='True')
    parser.add_argument('--lambda_geo',      type=float, default=0.1)
    parser.add_argument('--lambda_sem',      type=float, default=0.1)
    parser.add_argument('--adaptive_weights',
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=False)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--temperature',     type=float, default=0.07)
    parser.add_argument('--num_samples',     type=int,   default=1024)
    parser.add_argument('--model',           type=str,   default='yolo26s.pt')
    parser.add_argument('--data',            type=str,   default='widerface.yaml')
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=8)
    parser.add_argument('--img_size',        type=int,   default=640)
    parser.add_argument('--run_name',        type=str,   default=None)
    parser.add_argument('--seed',            type=int,   default=42)

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  CFDN-YOLO ABLATION STUDY")
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