#!/usr/bin/env python3
"""
CFDN-YOLO Ablation — Barlow Twins v11.7.

Fixes vs v11.1:
  FIX 1 (freeze gate):   on_train_epoch_start gates on absolute epoch
                         (trainer.epoch + epoch_offset). on_train_start
                         applies the freeze immediately if resuming past it.
  FIX 2 (optimizer):     resume restores detector optimizer state from
                         ckpt['optimizer'], and aux AdamW state is saved
                         inside aux_latest.pt and restored.
  FIX 3 (validation):    val=True in overrides; validate() and
                         final_eval() stubs removed.
  FIX 4 (grad check):    moved into optimizer_step so it reads the
                         accumulated grads, not a zeroed buffer.
  FIX 5 (hooks in eval): hook_* early-return when module is eval().
  FIX 6 (LR scheduler):  scheduler state saved in aux_latest.pt and
                         restored in on_train_start.

v11.4: patch `loss` instead of `forward` so validation is untouched.
v11.5: custom_loss calls self.hook.set_clean(batch['img']) before running
       the detection forward.
v11.6: custom_loss early-returns the standard loss when the model is in
       eval mode (Ultralytics calls model.loss(batch, preds) during
       validation to accumulate val loss; without this guard, hooks are
       no-ops and features{} is empty → KeyError 16).
v11.7: optimizer_step calls self.ema.update(self.model). Ultralytics does
       this inside its own optimizer_step; overriding the method dropped
       the EMA update, so validation (which measures EMA weights) was
       frozen at the COCO init → constant mAP50 for many epochs.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import sys
import types
import shutil
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
    lambda_geo:               float           = 0.01
    lambda_sem:               float           = 0.01
    lambda_barlow_off:        float           = 0.005
    adaptive_weights:         bool            = False
    warmup_epochs:            int             = 10
    neck_channels:            Tuple[int, ...] = (128, 256, 512)
    proj_hidden:              int             = 512
    proj_out:                 int             = 2048
    feature_noise_std_scale:  float           = 1.0
    aux_lr:                   float           = 3e-3
    main_grad_clip:           float           = 10.0
    aux_grad_clip:            float           = 1.0
    freeze_backbone_from:     int             = 5
    freeze_backbone_to:       int             = 16
    freeze_bn_stats:          bool            = True
    cls_monitor_window:       int             = 100
    cls_monitor_ratio:        float           = 1.20
    cls_monitor_required:     int             = 3
    val_period:               int             = 5
    early_stop_epoch:         int             = 20
    early_stop_min_map50:     float           = 0.05
    run_name:                 str             = "cfdn_barlow_reconnected"
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
            return "Baseline (clean detection, no contrastive branch)"
        if self.use_geometric and not self.use_semantic:
            return "Geometric encoder + Barlow Twins (reconnected)"
        if not self.use_geometric and self.use_semantic:
            return "Semantic encoder + Barlow Twins (reconnected)"
        return "Full CFDN Barlow Twins (reconnected)"


def drive_ckpt_dir_for(run_name: str) -> Path:
    candidates = [
        Path(f"/content/drive/MyDrive/CFDN_YOLO/checkpoints/{run_name}"),
        Path(f"/kaggle/working/checkpoints/{run_name}"),
        Path(f"./checkpoints/{run_name}"),
    ]
    for c in candidates:
        if c.parent.exists() or c.parent.parent.exists():
            c.mkdir(parents=True, exist_ok=True)
            return c
    c = Path(f"./checkpoints/{run_name}")
    c.mkdir(parents=True, exist_ok=True)
    return c


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
        self.opt_step_count:   int = 0
        self.epoch_offset:     int = 0
        self.remaining_epochs: int = 0

        self.recent_cls:            List[float] = []
        self.cls_reference:         Optional[float] = None
        self.consecutive_violations: int = 0

        self._original_loss     = None
        self._custom_loss       = None
        self._pending_aux_state: Optional[Dict[str, Any]] = None
        self._pending_optim_state: Optional[Dict[str, Any]] = None
        self._backbone_frozen:   bool = False

    def setup_model(self) -> None:
        super().setup_model()
        self._attach_cfdn_modules()
        self._patch_loss()
        self.add_callback('on_train_start',       self.on_train_start)
        self.add_callback('on_train_batch_end',   self.on_train_batch_end)
        self.add_callback('on_train_epoch_start', self.on_train_epoch_start)
        self.add_callback('on_train_epoch_end',   self.on_train_epoch_end)
        self.add_callback('on_fit_epoch_end',     self.on_fit_epoch_end)
        self.add_callback('on_model_save',        self.on_model_save)

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

        print(f"  [CFDN] Modules attached. Aux params: "
              f"{self.hook.get_auxiliary_param_count():,}")
        print(f"  [CFDN] Barlow Twins: proj_out={config.proj_out}, "
              f"lambda_off={config.lambda_barlow_off}")
        print(f"  [CFDN] Aux loss weights: λ_geo={config.lambda_geo}, "
              f"λ_sem={config.lambda_sem}")
        print(f"  [CFDN] Freeze schedule: layers 0..{config.freeze_backbone_to - 1} "
              f"at epoch {config.freeze_backbone_from}, "
              f"BN stats frozen: {config.freeze_bn_stats}")
        print(f"  [CFDN] val_period={config.val_period}, "
              f"early_stop if mAP50 < {config.early_stop_min_map50} at epoch "
              f"≥ {config.early_stop_epoch}")

    def build_optimizer(self, model, name='auto', lr=0.01, momentum=0.9, decay=0.0, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        if self.hook is not None:
            aux_params = list(self.hook.parameters())
            if self.weighter is not None:
                aux_params += list(self.weighter.parameters())
            if aux_params:
                self.aux_optimizer = torch.optim.AdamW(
                    aux_params, lr=self.cfdn_config.aux_lr, weight_decay=0.0,
                )
                print(f"  [CFDN] Aux AdamW lr={self.cfdn_config.aux_lr}: "
                      f"{len(aux_params)} param tensors.")
        return optimizer

    def _patch_loss(self) -> None:
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        original_loss = det_model.loss
        self._original_loss = original_loss
        config = self.cfdn_config

        def custom_loss(model_self, batch, *args, **kwargs):
            if not model_self.training:
                return original_loss(batch, *args, **kwargs)

            if isinstance(batch, dict) and 'img' in batch:
                self.hook.set_clean(batch['img'])

            loss, loss_items = original_loss(batch, *args, **kwargs)

            if not torch.isfinite(loss).all():
                return loss, loss_items
            if not (config.use_geometric or config.use_semantic):
                return loss, loss_items

            with torch.amp.autocast('cuda', enabled=False):
                clean_neck = self.hook.get_neck_outputs()
                self.hook.last_clean_neck = [f.detach().clone() for f in clean_neck]

                noisy_neck = []
                for f in clean_neck:
                    f_fp32 = f.float()
                    std = f_fp32.detach().std().clamp(max=10.0)
                    noise = torch.randn_like(f_fp32) * (std + 1e-6) * config.feature_noise_std_scale
                    noisy_neck.append(f_fp32 + noise)

                self.hook.features = {}
                geo_loss, sem_loss = self.hook.compute_loss_from_features(noisy_neck)

                aux_loss = torch.tensor(0.0, device=loss.device)
                if config.use_geometric:
                    aux_loss = aux_loss + config.lambda_geo * geo_loss
                if config.use_semantic:
                    aux_loss = aux_loss + config.lambda_sem * sem_loss

                effective_epoch = self.epoch + self.epoch_offset
                ramp = min(1.0, effective_epoch / max(1, config.warmup_epochs))
                aux_loss = ramp * aux_loss

                if torch.isfinite(aux_loss) and aux_loss.abs() < 1e6:
                    loss = loss + aux_loss

            if not torch.isnan(geo_loss):
                self.cont_losses_geo.append(geo_loss.item())
            if not torch.isnan(sem_loss):
                self.cont_losses_sem.append(sem_loss.item())

            return loss, loss_items

        det_model.loss = types.MethodType(custom_loss, det_model)
        self._custom_loss = custom_loss
        print("  [CFDN] Loss patched (Barlow Twins v11.7 — eval guard + set_clean + EMA).")

    # ---------------- optimizer ----------------

    def optimizer_step(self) -> None:
        main_loss = getattr(self, 'loss', None)
        if main_loss is not None and not torch.isfinite(main_loss).all():
            print(f"  [CFDN] Non-finite loss at batch {self.batch_count}; skipping step.")
            self.optimizer.zero_grad()
            if self.aux_optimizer is not None:
                self.aux_optimizer.zero_grad()
            return

        if self.opt_step_count % 500 == 0 and self.opt_step_count > 0:
            try:
                det_model = self.model.module if hasattr(self.model, 'module') else self.model
                p16 = det_model.model[16]
                w = next((p for n, p in p16.named_parameters() if 'weight' in n), None)
                if w is not None:
                    gn = w.grad.norm().item() if w.grad is not None else 0.0
                    print(f"  [GRAD CHECK] opt_step={self.opt_step_count} "
                          f"neck L16 weight grad norm = {gn:.4f}")
            except Exception as e:
                print(f"  [GRAD CHECK] failed: {e}")

        aux_has_grads = False
        if self.aux_optimizer is not None:
            aux_params = [p for g in self.aux_optimizer.param_groups for p in g['params']]
            aux_has_grads = any(p.grad is not None for p in aux_params)

        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            if aux_has_grads:
                self.scaler.unscale_(self.aux_optimizer)

        main_params = [p for g in self.optimizer.param_groups for p in g['params']]
        if self.cfdn_config.main_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(main_params, max_norm=self.cfdn_config.main_grad_clip)
        if aux_has_grads and self.cfdn_config.aux_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.aux_optimizer.param_groups for p in g['params']],
                max_norm=self.cfdn_config.aux_grad_clip,
            )

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

        # v11.7: Ultralytics updates the EMA inside its own optimizer_step.
        # We override optimizer_step, so we must call ema.update ourselves.
        if getattr(self, 'ema', None) is not None:
            self.ema.update(self.model)

        self.opt_step_count += 1

    # ---------------- callbacks ----------------

    def on_train_start(self, trainer) -> None:
        device = torch.device(self.device if self.device else 'cuda:0')
        if self.hook is not None:
            self.hook = self.hook.to(device)
        if self.weighter is not None:
            self.weighter = self.weighter.to(device)

        if self._pending_optim_state is not None:
            try:
                self.optimizer.load_state_dict(self._pending_optim_state)
                print("  [RESUME] Detector optimizer state restored.")
            except Exception as e:
                print(f"  [RESUME] WARNING: optimizer state load failed: {e}")
            self._pending_optim_state = None

        if self._pending_aux_state is not None and self.hook is not None:
            state = self._pending_aux_state
            try:
                if self.hook.geo_encoder is not None and 'geo_encoder' in state:
                    self.hook.geo_encoder.load_state_dict(state['geo_encoder'])
                if self.hook.sem_encoder is not None and 'sem_encoder' in state:
                    self.hook.sem_encoder.load_state_dict(state['sem_encoder'])
                self.hook.proj_neck.load_state_dict(state['proj_neck'])
                if self.hook.proj_clean_geo is not None and 'proj_clean_geo' in state:
                    self.hook.proj_clean_geo.load_state_dict(state['proj_clean_geo'])
                if self.hook.proj_clean_sem is not None and 'proj_clean_sem' in state:
                    self.hook.proj_clean_sem.load_state_dict(state['proj_clean_sem'])

                if 'aux_optimizer' in state and self.aux_optimizer is not None:
                    try:
                        self.aux_optimizer.load_state_dict(state['aux_optimizer'])
                        print("  [RESUME] Aux AdamW state restored.")
                    except Exception as e:
                        print(f"  [RESUME] WARNING: aux optimizer state failed: {e}")

                if 'lr_scheduler' in state and hasattr(self, 'scheduler') and self.scheduler is not None:
                    try:
                        self.scheduler.load_state_dict(state['lr_scheduler'])
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        print(f"  [RESUME] LR scheduler state restored (lr={cur_lr:.6f}).")
                    except Exception as e:
                        print(f"  [RESUME] WARNING: scheduler restore failed: {e}")

                print("  [RESUME] Auxiliary modules restored from disk.")
            except Exception as e:
                print(f"  [RESUME] WARNING: failed to load aux modules: {e}")
            self._pending_aux_state = None

        if (self.epoch_offset >= self.cfdn_config.freeze_backbone_from
                and not self._backbone_frozen):
            self._apply_backbone_freeze(self.epoch_offset)
            self._backbone_frozen = True

    def _apply_backbone_freeze(self, absolute_epoch: int) -> None:
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        n = self.cfdn_config.freeze_backbone_to
        for i in range(0, n):
            for p in det_model.model[i].parameters():
                p.requires_grad = False
            if self.cfdn_config.freeze_bn_stats:
                det_model.model[i].eval()
        frozen = sum(p.numel() for p in det_model.model[:n].parameters())
        print(f"  [CFDN] Epoch {absolute_epoch}: froze backbone layers "
              f"0..{n - 1} ({frozen:,} params). "
              f"BN stats frozen: {self.cfdn_config.freeze_bn_stats}")

    def on_train_epoch_start(self, trainer) -> None:
        if self._backbone_frozen:
            return
        absolute_epoch = trainer.epoch + self.epoch_offset
        if absolute_epoch < self.cfdn_config.freeze_backbone_from:
            return
        self._apply_backbone_freeze(absolute_epoch)
        self._backbone_frozen = True

    def on_train_batch_end(self, trainer) -> None:
        self.batch_count += 1

        if self.batch_count % 100 == 0:
            avg_geo = float(np.mean(self.cont_losses_geo[-100:])) if self.cont_losses_geo else 0.0
            avg_sem = float(np.mean(self.cont_losses_sem[-100:])) if self.cont_losses_sem else 0.0
            print(f"  [CFDN] Batch {self.batch_count}: Geo={avg_geo:.4f} | Sem={avg_sem:.4f}")

        absolute_epoch = trainer.epoch + self.epoch_offset
        if absolute_epoch < self.cfdn_config.freeze_backbone_from:
            return

        try:
            cls_val = float(trainer.loss_items[1]) if trainer.loss_items is not None else None
        except (TypeError, IndexError):
            cls_val = None

        if cls_val is not None:
            self.recent_cls.append(cls_val)

        window = self.cfdn_config.cls_monitor_window
        if len(self.recent_cls) >= window:
            avg = sum(self.recent_cls) / len(self.recent_cls)
            self.recent_cls = []

            if self.cls_reference is None:
                self.cls_reference = avg
                print(f"  [CFDN] cls_loss reference set to {avg:.4f} "
                      f"(first full {window}-batch window after freeze)")
            else:
                ratio = self.cfdn_config.cls_monitor_ratio
                required = self.cfdn_config.cls_monitor_required
                if avg > self.cls_reference * ratio:
                    self.consecutive_violations += 1
                    print(f"  [CFDN] WARNING: cls_loss avg {avg:.4f} > "
                          f"{ratio}× reference "
                          f"({self.consecutive_violations}/{required} consecutive windows)")
                    if self.consecutive_violations >= required:
                        print(f"  [CFDN] ABORT: {required} consecutive windows above "
                              f"threshold. Stopping.")
                        trainer.stop = True
                else:
                    if self.consecutive_violations > 0:
                        print(f"  [CFDN] Recovered: cls_loss avg {avg:.4f} back below "
                              f"threshold. Counter reset.")
                    self.consecutive_violations = 0

    def on_fit_epoch_end(self, trainer) -> None:
        if trainer.epoch + 1 < self.cfdn_config.early_stop_epoch:
            return
        metrics = getattr(trainer, 'metrics', None)
        if not metrics:
            return
        map50 = metrics.get('metrics/mAP50(B)')
        if map50 is None:
            return
        if map50 < self.cfdn_config.early_stop_min_map50:
            print(f"  [PILOT] Early stop: mAP50={map50:.4f} < "
                  f"{self.cfdn_config.early_stop_min_map50} at epoch {trainer.epoch + 1}.")
            trainer.stop = True
        else:
            print(f"  [PILOT] Epoch {trainer.epoch + 1}: "
                  f"mAP50={map50:.4f} (>= threshold, continuing).")

    def on_train_epoch_end(self, trainer) -> None:
        if self.remaining_epochs > 0 and trainer.epoch + 1 >= self.remaining_epochs:
            print(f"  [CFDN] Reached target: {trainer.epoch + 1} epochs done this "
                  f"session (offset {self.epoch_offset}). Stopping.")
            trainer.stop = True

    def on_model_save(self, trainer) -> None:
        try:
            ckpt_dir = drive_ckpt_dir_for(self.cfdn_config.run_name)

            last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
            if last_pt.exists():
                shutil.copy(str(last_pt), str(ckpt_dir / "last.pt"))

            if self.hook is not None:
                aux_state: Dict[str, Any] = {'proj_neck': self.hook.proj_neck.state_dict()}
                if self.hook.geo_encoder is not None:
                    aux_state['geo_encoder'] = self.hook.geo_encoder.state_dict()
                if self.hook.sem_encoder is not None:
                    aux_state['sem_encoder'] = self.hook.sem_encoder.state_dict()
                if self.hook.proj_clean_geo is not None:
                    aux_state['proj_clean_geo'] = self.hook.proj_clean_geo.state_dict()
                if self.hook.proj_clean_sem is not None:
                    aux_state['proj_clean_sem'] = self.hook.proj_clean_sem.state_dict()
                if self.aux_optimizer is not None:
                    aux_state['aux_optimizer'] = self.aux_optimizer.state_dict()
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    aux_state['lr_scheduler'] = self.scheduler.state_dict()
                torch.save(aux_state, str(ckpt_dir / "aux_latest.pt"))

            absolute_epoch = trainer.epoch + self.epoch_offset + 1
            with open(ckpt_dir / "resume_state.json", 'w') as f:
                json.dump({
                    "epochs_done": absolute_epoch,
                    "run_name": self.cfdn_config.run_name,
                }, f)

            print(f"  [CFDN] Epoch {absolute_epoch} state saved: {ckpt_dir}")
        except Exception as e:
            print(f"  [CFDN] WARNING: failed to save state: {e}")

    def save_model(self) -> None:
        det_model = self.model.module if hasattr(self.model, 'module') else self.model
        had_custom_loss = 'loss' in det_model.__dict__
        if had_custom_loss:
            del det_model.__dict__['loss']

        ema_model = None
        had_ema_custom_loss = False
        if hasattr(self, 'ema') and self.ema is not None:
            ema_model = self.ema.ema
            if hasattr(ema_model, '__dict__'):
                had_ema_custom_loss = 'loss' in ema_model.__dict__
                if had_ema_custom_loss:
                    del ema_model.__dict__['loss']

        try:
            result = super().save_model()
        finally:
            if had_custom_loss:
                det_model.loss = types.MethodType(self._custom_loss, det_model)
            if had_ema_custom_loss and ema_model is not None:
                ema_model.loss = types.MethodType(self._custom_loss, ema_model)

        return result

    def get_summary(self) -> Dict[str, Any]:
        return {
            "contrastive_loss_active": bool(self.cont_losses_geo or self.cont_losses_sem),
            "mean_geo_loss": float(np.mean(self.cont_losses_geo)) if self.cont_losses_geo else 0.0,
            "mean_sem_loss": float(np.mean(self.cont_losses_sem)) if self.cont_losses_sem else 0.0,
            "adaptive_weights": self.cfdn_config.adaptive_weights,
            "final_log_var": None,
        }


# ===================== TRAINING FUNCTION =====================

def train_ablation(config: AblationConfig, resume: bool = False) -> str:
    print(f"\n{'='*70}")
    print(f"  {config.run_name}: {config.get_run_description()}")
    print(f"  Resume: {resume}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"./runs/ablation/{config.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [CFDN] Local output dir: {output_dir}")

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    total_epochs = config.epochs
    start_epoch  = 0
    model_ckpt   = config.model_variant
    aux_ckpt     = None
    optim_state  = None

    ckpt_dir = drive_ckpt_dir_for(config.run_name)
    print(f"  [CFDN] Checkpoint dir: {ckpt_dir}")

    if resume:
        last_pt    = ckpt_dir / "last.pt"
        aux_path   = ckpt_dir / "aux_latest.pt"
        state_path = ckpt_dir / "resume_state.json"

        if last_pt.exists():
            model_ckpt = str(last_pt)
            aux_ckpt   = aux_path if aux_path.exists() else None

            if state_path.exists():
                with open(state_path) as f:
                    start_epoch = int(json.load(f).get("epochs_done", 0))
            else:
                print("  [RESUME] WARNING: resume_state.json missing; start_epoch=0.")

            if start_epoch >= total_epochs:
                print(f"  [RESUME] Already complete ({start_epoch}/{total_epochs}).")
                return str(ckpt_dir)

            try:
                ckpt = torch.load(model_ckpt, map_location='cpu', weights_only=False)
                optim_state = ckpt.get('optimizer')
                if optim_state is not None:
                    print("  [RESUME] Detector optimizer state found in checkpoint.")
                else:
                    print("  [RESUME] WARNING: no optimizer state in checkpoint.")
                del ckpt
            except Exception as e:
                print(f"  [RESUME] WARNING: could not read optimizer state: {e}")

            print(f"  [RESUME] Found checkpoint. Epochs done: {start_epoch}.")
            print(f"  [RESUME] model_ckpt : {model_ckpt}")
            print(f"  [RESUME] aux_ckpt   : {aux_ckpt}")
        else:
            print(f"  [RESUME] No last.pt at {last_pt}; starting from scratch.")
            resume = False

    remaining_epochs = max(1, total_epochs - start_epoch)
    print(f"  [CFDN] Total epochs: {total_epochs}, "
          f"start: {start_epoch}, remaining: {remaining_epochs}")

    overrides = {
        'model':      model_ckpt,
        'data':       config.data_yaml,
        'epochs':     total_epochs,
        'batch':      config.batch_size,
        'imgsz':      config.img_size,
        'device':     'cuda:0' if torch.cuda.is_available() else 'cpu',
        'amp':        True,
        'workers':    4,
        'project':    str(output_dir.absolute()),
        'name':       'weights',
        'exist_ok':   True,
        'verbose':    False,
        'seed':       config.seed,
        'val':        True,
    }
    if resume and start_epoch > 0:
        overrides['warmup_epochs'] = 0.0

    trainer = CFDNTrainer(config=config, overrides=overrides)
    trainer.epoch_offset     = start_epoch
    trainer.remaining_epochs = remaining_epochs
    trainer._pending_optim_state = optim_state

    if aux_ckpt is not None and Path(aux_ckpt).exists():
        trainer._pending_aux_state = torch.load(aux_ckpt, map_location='cpu')
        print(f"  [RESUME] Queued aux state from {aux_ckpt}.")
    else:
        print("  [RESUME] No aux state found; starting aux modules fresh.")

    trainer.train()

    summary = {
        "run_name":         config.run_name,
        "description":      config.get_run_description(),
        "config":           config.to_dict(),
        "contrastive_loss": trainer.get_summary(),
        "auxiliary_params": trainer.hook.get_auxiliary_param_count() if trainer.hook else 0,
        "resumed_from":     start_epoch,
        "final_metrics":    {k: float(v) for k, v in (trainer.metrics or {}).items()
                             if isinstance(v, (int, float))},
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done: {output_dir}")
    return str(output_dir)


# ===================== ENTRY POINT =====================

def main() -> None:
    parser = argparse.ArgumentParser(description="CFDN-YOLO Barlow Twins v11.7")
    parser.add_argument('--use_geometric',
                        type=lambda x: x.lower() in ('true','1','yes'), default=True)
    parser.add_argument('--use_semantic',
                        type=lambda x: x.lower() in ('true','1','yes'), default=True)
    parser.add_argument('--lambda_geo',              type=float, default=0.01)
    parser.add_argument('--lambda_sem',              type=float, default=0.01)
    parser.add_argument('--lambda_barlow_off',       type=float, default=0.005)
    parser.add_argument('--adaptive_weights',
                        type=lambda x: x.lower() in ('true','1','yes'), default=False)
    parser.add_argument('--warmup_epochs',           type=int,   default=10)
    parser.add_argument('--proj_hidden',             type=int,   default=512)
    parser.add_argument('--proj_out',                type=int,   default=2048)
    parser.add_argument('--feature_noise_std_scale', type=float, default=1.0)
    parser.add_argument('--aux_lr',                  type=float, default=3e-3)
    parser.add_argument('--freeze_backbone_from',    type=int,   default=5)
    parser.add_argument('--freeze_backbone_to',      type=int,   default=16)
    parser.add_argument('--freeze_bn_stats',
                        type=lambda x: x.lower() in ('true','1','yes'), default=True)
    parser.add_argument('--val_period',              type=int,   default=5)
    parser.add_argument('--early_stop_epoch',        type=int,   default=20)
    parser.add_argument('--early_stop_min_map50',    type=float, default=0.05)
    parser.add_argument('--model',                   type=str,   default='yolo26s.pt')
    parser.add_argument('--data',                    type=str,   default='widerface.yaml')
    parser.add_argument('--epochs',                  type=int,   default=100)
    parser.add_argument('--batch_size',              type=int,   default=8)
    parser.add_argument('--img_size',                type=int,   default=640)
    parser.add_argument('--run_name',                type=str,   default=None)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--resume',                  action='store_true')

    args = parser.parse_args()

    if args.run_name is None:
        if not args.use_geometric and not args.use_semantic:
            args.run_name = "baseline"
        elif args.use_geometric and not args.use_semantic:
            args.run_name = "geo_only_reconnected"
        elif not args.use_geometric and args.use_semantic:
            args.run_name = "sem_only_reconnected"
        else:
            args.run_name = "cfdn_full_reconnected"

    config = AblationConfig(
        use_geometric           = args.use_geometric,
        use_semantic            = args.use_semantic,
        run_name                = args.run_name,
        lambda_geo              = args.lambda_geo,
        lambda_sem              = args.lambda_sem,
        lambda_barlow_off       = args.lambda_barlow_off,
        adaptive_weights        = args.adaptive_weights,
        warmup_epochs           = args.warmup_epochs,
        proj_hidden             = args.proj_hidden,
        proj_out                = args.proj_out,
        feature_noise_std_scale = args.feature_noise_std_scale,
        aux_lr                  = args.aux_lr,
        freeze_backbone_from    = args.freeze_backbone_from,
        freeze_backbone_to      = args.freeze_backbone_to,
        freeze_bn_stats         = args.freeze_bn_stats,
        val_period              = args.val_period,
        early_stop_epoch        = args.early_stop_epoch,
        early_stop_min_map50    = args.early_stop_min_map50,
        model_variant           = args.model,
        data_yaml               = args.data,
        epochs                  = args.epochs,
        batch_size              = args.batch_size,
        img_size                = args.img_size,
        seed                    = args.seed,
    )

    os.makedirs("./runs/ablation", exist_ok=True)
    train_ablation(config, resume=args.resume)


if __name__ == '__main__':
    main()