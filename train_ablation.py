#!/usr/bin/env python3
"""
CFDN-YOLO Ablation Study — final custom trainer (validation OFF).

Auxiliary modules live on the trainer; the training path is activated only
when 'bboxes' is present in the batch dict.  Validation and final_eval are
skipped entirely — evaluate separately with model.val() after training.

Gradient-conflict diagnostic runs every 50 batches on proj_neck to confirm
geo and sem losses remain orthogonal (established empirically: cosine ≈ 0).

Uses a separate AdamW optimizer for auxiliary modules to avoid incompatibility
with the main MuSGD optimizer chosen by Ultralytics for YOLO26s.

Fixes applied (v3):
  1. optimizer_step() uses scaler.step() not optimizer.step() — preserves
     AMP inf/nan guard so gradient overflow safely skips the step.
  2. save_model() restores EMA forward-patch cleanup — EMA is deepcopy'd
     AFTER _patch_forward() so it inherits the patch; must be cleaned too
     or best.pt bakes in custom_forward and fails to load.
  3. aux_loss NaN guard — 0.0 * NaN = NaN in PyTorch; skip adding aux_loss
     entirely if it is NaN/Inf rather than silently poisoning detection loss.
  4. Aux gradient clipping (max_norm=1.0) always applied regardless of the
     main model's clip_grad setting — prevents proj_neck gradient spikes from
     exploding into the detection loss via the shared backward graph.
  5. Linear warmup ramp on fixed-lambda path mirrors adaptive weighter warmup.
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
    """
    Subclass of DetectionTrainer that attaches CFDN auxiliary modules to the
    trainer object (not to the model), keeping best.pt identical to a vanilla
    YOLO26s checkpoint with no auxiliary parameters.

    Key design points:
    - Auxiliary parameters are updated by a dedicated AdamW optimizer so they
      don't interact with the MuSGD schedule used for the main model.
    - Validation is skipped during training (val=False); evaluate separately.
    - save_model() temporarily removes the patched forward before saving so
      the checkpoint loads cleanly with a plain YOLO() call.
    """

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
        """Load YOLO, attach aux modules, patch forward, register callbacks."""
        super().setup_model()
        self._attach_cfdn_modules()
        self._patch_forward()
        self.add_callback('on_train_start',     self.on_train_start)
        self.add_callback('on_train_batch_end', self.on_train_batch_end)

    def _attach_cfdn_modules(self) -> None:
        """Instantiate MultiHook + optional AdaptiveLossWeighter, register neck hooks."""
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
            raise RuntimeError("Unexpected model structure — expected DetectionModel")

        layers = det_model.model
        layers[16].register_forward_hook(self.hook.hook_p3)
        layers[19].register_forward_hook(self.hook.hook_p4)
        layers[22].register_forward_hook(self.hook.hook_p5)

        print(f"  [CFDN] Auxiliary modules created. "
              f"Params: {self.hook.get_auxiliary_param_count():,}")

    def build_optimizer(
        self, model, name='auto', lr=0.01, momentum=0.9, decay=0.0, iterations=1e5
    ):
        """
        Build the main YOLO optimizer (MuSGD via Ultralytics auto-select), then
        create a separate AdamW for auxiliary parameters to avoid scheduler
        incompatibility.
        """
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
        Monkeypatches DetectionModel.forward to:
          1. Clone clean images and store them in the hook.
          2. Inject random noise into batch['img'] before the YOLO forward pass.
          3. Compute split contrastive losses (geo, sem) after the YOLO loss.
          4. Run gradient-conflict diagnostic every 50 batches on proj_neck.
          5. Add weighted auxiliary loss to the detection loss with NaN guard.
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
                # Inference / validation path — untouched
                if isinstance(batch, dict) and 'img' in batch:
                    batch = batch['img']
                if isinstance(batch, torch.Tensor) and batch.dtype == torch.float16:
                    batch = batch.float()
                return original_forward(batch, *args, **kwargs)

            # ── Training path ────────────────────────────────────────────────

            # 1. Store clean images as reference for encoders
            clean_img = batch['img'].clone()
            self.hook.set_clean(clean_img)

            # 2. Inject noise into the batch in-place
            noise_type  = str(np.random.choice(list(noise_cfg.noise_types)))
            noise_param = float(np.random.choice(list(noise_cfg.noise_params)))
            batch['img'] = add_noise(batch['img'], noise_type=noise_type, param=noise_param)

            # 3. YOLO forward pass — hooks capture noisy neck features
            loss, loss_items = original_forward(batch, *args, **kwargs)

            # 4. Compute split contrastive losses
            geo_loss, sem_loss = self.hook.compute_loss()

            # 5. Gradient-conflict diagnostic on shared proj_neck (every 50 batches)
            if self.batch_count % 50 == 0 and config.use_geometric and config.use_semantic:
                try:
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
                    print(
                        f"[GRAD CONFLICT] batch={self.batch_count} "
                        f"cosine={cos_sim:.4f} "
                        f"|geo|={geo_vec.norm().item():.4f} "
                        f"|sem|={sem_vec.norm().item():.4f}"
                    )
                except Exception as e:
                    print(f"[GRAD CONFLICT] skipped at batch {self.batch_count}: {e}")

            # 6. Combine contrastive losses with detection loss
            if config.use_geometric or config.use_semantic:
                if self.weighter is not None:
                    # Canonical Kendall adaptive weighting with warmup
                    task_losses = [
                        geo_loss if config.use_geometric
                        else torch.tensor(0.0, device=geo_loss.device),
                        sem_loss if config.use_semantic
                        else torch.tensor(0.0, device=sem_loss.device),
                    ]
                    aux_loss = self.weighter(task_losses, self.epoch)
                else:
                    # Fixed split-lambda with linear warmup ramp
                    aux_loss = torch.tensor(0.0, device=loss.device)
                    if config.use_geometric:
                        aux_loss = aux_loss + config.lambda_geo * geo_loss
                    if config.use_semantic:
                        aux_loss = aux_loss + config.lambda_sem * sem_loss
                    ramp = min(1.0, self.epoch / max(1, config.warmup_epochs))
                    aux_loss = ramp * aux_loss

                # NaN guard: 0.0 * NaN = NaN in PyTorch — skip adding entirely
                # if aux_loss is invalid rather than poisoning the detection loss.
                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    print(f"  [CFDN] WARNING: aux_loss={aux_loss.item():.4f} "
                          f"at batch {self.batch_count} — skipping this batch's aux contribution")
                else:
                    loss = loss + aux_loss

            # 7. Log raw contrastive losses for the summary
            if not torch.isnan(geo_loss) and geo_loss.item() > 0:
                self.cont_losses_geo.append(geo_loss.item())
            if not torch.isnan(sem_loss) and sem_loss.item() > 0:
                self.cont_losses_sem.append(sem_loss.item())

            return loss, loss_items

        det_model.forward = types.MethodType(custom_forward, det_model)
        self._custom_forward = custom_forward
        print("  [CFDN] Forward patching complete.")

    # ── Optimizer step ────────────────────────────────────────────────────────

    def optimizer_step(self) -> None:
        """
        Step both the main YOLO optimizer and the auxiliary AdamW optimizer.

        Uses scaler.step() (not optimizer.step()) so AMP's inf/nan guard is
        respected: if gradient overflow is detected the step is skipped and
        the loss scale is reduced, rather than blindly updating with garbage.

        unscale_() is called explicitly before clipping so gradient norms are
        computed in true fp32 scale; scaler.step() detects the prior unscale
        and skips re-unscaling.

        Aux gradients are always clipped at max_norm=1.0 regardless of the
        main model's clip_grad setting — proj_neck gradient spikes at early
        training can otherwise propagate through the shared backward graph
        and cause loss divergence.
        """
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            if self.aux_optimizer is not None:
                self.scaler.unscale_(self.aux_optimizer)

        # Main model gradient clipping (respects training config)
        clip_val = (
            getattr(self.args, 'clip_grad', 0.0)
            if hasattr(self.args, 'clip_grad') else 0.0
        )
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'], clip_val
            )

        # Aux gradient clipping — always applied, protects against early spikes
        if self.aux_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.aux_optimizer.param_groups for p in g['params']],
                max_norm=1.0
            )

        if self.scaler:
            # scaler.step() checks for inf/nan and skips the step if found
            self.scaler.step(self.optimizer)
            if self.aux_optimizer is not None:
                self.scaler.step(self.aux_optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            if self.aux_optimizer is not None:
                self.aux_optimizer.step()

        self.optimizer.zero_grad()
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_train_start(self, trainer) -> None:
        """Ensure auxiliary modules are on the correct device before training begins."""
        device = torch.device(self.device if self.device else 'cuda:0')
        if self.hook is not None:
            self.hook = self.hook.to(device)
        if self.weighter is not None:
            self.weighter = self.weighter.to(device)

    def on_train_batch_end(self, trainer) -> None:
        """Log mean contrastive losses every 100 batches."""
        self.batch_count += 1
        if self.batch_count % 100 == 0:
            avg_geo = float(np.mean(self.cont_losses_geo[-100:])) if self.cont_losses_geo else 0.0
            avg_sem = float(np.mean(self.cont_losses_sem[-100:])) if self.cont_losses_sem else 0.0
            if self.cfdn_config.use_geometric or self.cfdn_config.use_semantic:
                print(
                    f"  [CFDN] Batch {self.batch_count}: "
                    f"Geo Loss = {avg_geo:.4f} | Sem Loss = {avg_sem:.4f}"
                )

    # ── Validation / eval overrides ───────────────────────────────────────────

    def validate(self):
        """Skip in-training validation — evaluate separately after training."""
        return {}, 0.0

    def final_eval(self) -> None:
        """Skip final evaluation — avoids checkpoint-loading issues with patched forward."""
        pass

    # ── Checkpoint saving ─────────────────────────────────────────────────────

    def save_model(self) -> None:
        """
        Removes the monkeypatched forward from both det_model and the EMA model
        before calling super().save_model(), then restores both immediately after.

        Why EMA must be cleaned too:
          Ultralytics creates the EMA model via deepcopy(det_model) which happens
          AFTER _patch_forward() runs. The EMA model therefore inherits the patched
          forward in its instance __dict__. The EMA weights are what Ultralytics
          actually serialises into best.pt, so if we only clean det_model the
          checkpoint still contains custom_forward and fails to load with YOLO().
        """
        det_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Clean main model
        had_custom = 'forward' in det_model.__dict__
        if had_custom:
            del det_model.__dict__['forward']

        # Clean EMA model
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
            # Always restore — even if save raised an exception
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
    parser.add_argument('--all',             action='store_true',
                        help="Run all ablation arms sequentially")
    parser.add_argument('--use_geometric',   type=str,   default='True')
    parser.add_argument('--use_semantic',    type=str,   default='True')
    parser.add_argument('--lambda_geo',      type=float, default=0.1)
    parser.add_argument('--lambda_sem',      type=float, default=0.1)
    parser.add_argument('--adaptive_weights',
                        type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=False,
                        help="Use Kendall adaptive weighting instead of fixed lambda")
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