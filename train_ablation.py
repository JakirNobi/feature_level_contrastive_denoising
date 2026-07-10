#!/usr/bin/env python3
"""
CFDN‑YOLO Ablation Study – reliable training‑time injection via custom callback.
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

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).parent))

from contrastive_denoise_yolo26s import MultiHook
from utils.noise import add_noise, NoiseConfig


# ===================== CONFIG =====================

@dataclass
class AblationConfig:
    use_geometric: bool = True
    use_semantic: bool = True
    lambda_contrastive: float = 0.5
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

    def get_run_description(self) -> str:
        if not self.use_geometric and not self.use_semantic:
            return "Baseline (no auxiliary encoders)"
        if self.use_geometric and not self.use_semantic:
            return "Geometric Encoder Only"
        if not self.use_geometric and self.use_semantic:
            return "Semantic Encoder Only"
        return "Full Model (Geometric + Semantic)"


# ===================== ABLATION INJECTOR =====================

class AblationInjector:
    """
    Hooks into the Ultralytics trainer to patch the active model exactly
    when training starts, so contrastive loss is included in the backward pass.
    """

    def __init__(self, config: AblationConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.hook: Optional[MultiHook] = None
        self.noise_cfg: NoiseConfig = NoiseConfig(config.noise_types, config.noise_params)
        self.cont_losses: List[float] = []
        self.batch_count: int = 0
        self.aux_params: int = 0

    def on_train_start(self, trainer: Any) -> None:
        print("\n  [CFDN] Injecting hooks & forward patch into active trainer.model...")

        # Grab the actual model being used by the trainer
        model = trainer.model
        inner_model = model.module if hasattr(model, 'module') else model

        # Create the MultiHook and move to device
        self.hook = MultiHook(
            neck_channels=list(self.config.neck_channels),
            proj_hidden=self.config.proj_hidden,
            proj_out=self.config.proj_out,
            use_geometric=self.config.use_geometric,
            use_semantic=self.config.use_semantic,
        ).to(self.device)
        self.aux_params = self.hook.get_auxiliary_param_count()

        # Register layer hooks on the active network
        model_layers: nn.Sequential = inner_model.model
        model_layers[16].register_forward_hook(self.hook.hook_p3)
        model_layers[19].register_forward_hook(self.hook.hook_p4)
        model_layers[22].register_forward_hook(self.hook.hook_p5)

        # Save original forward
        original_forward = inner_model.forward

        def custom_forward(model_self, batch, *args, **kwargs):
            # Only intercept training batches
            if isinstance(batch, dict) and 'img' in batch and model_self.training:
                # Store clean reference
                clean_img = batch['img'].clone()
                self.hook.set_clean(clean_img)

                # Inject noise into the batch
                noise_type = str(np.random.choice(list(self.noise_cfg.noise_types)))
                noise_param = float(np.random.choice(list(self.noise_cfg.noise_params)))
                batch['img'] = add_noise(batch['img'], noise_type=noise_type, param=noise_param)

                # Standard YOLO detection forward (returns loss, loss_items)
                loss, loss_items = original_forward(batch, *args, **kwargs)

                # Compute and add contrastive loss
                cont_loss = self.hook.compute_loss()
                if isinstance(cont_loss, torch.Tensor) and cont_loss > 0:
                    loss = loss + self.config.lambda_contrastive * cont_loss
                    self.cont_losses.append(cont_loss.item())

                return loss, loss_items
            else:
                # Validation or inference – no noise, no contrastive loss
                return original_forward(batch, *args, **kwargs)

        # Patch the active model's forward method
        inner_model.forward = types.MethodType(custom_forward, inner_model)
        print(f"  [CFDN] Injection complete! Aux Params: {self.aux_params:,}")

    def on_train_batch_end(self, trainer: Any) -> None:
        self.batch_count += 1
        if self.batch_count % 100 == 0 and self.cont_losses:
            avg_loss = float(np.mean(self.cont_losses[-100:]))
            print(f"  [{self.config.run_name}] Batch {self.batch_count}: "
                  f"Avg Contrastive Loss = {avg_loss:.4f}")

    def get_summary(self) -> Dict[str, Any]:
        if not self.cont_losses:
            return {
                "contrastive_loss_active": False,
                "num_batches_with_loss": 0,
                "mean_contrastive_loss": 0.0,
            }
        return {
            "contrastive_loss_active": True,
            "num_batches_with_loss": len(self.cont_losses),
            "mean_contrastive_loss": float(np.mean(self.cont_losses)),
            "std_contrastive_loss": float(np.std(self.cont_losses)),
        }


# ===================== TRAINING FUNCTION =====================

def train_ablation(config: AblationConfig) -> str:
    print(f"\n{'='*70}")
    print(f"  ABLATION EXPERIMENT: {config.run_name}")
    print(f"  Description: {config.get_run_description()}")
    print(f"  Geometric: {config.use_geometric} | Semantic: {config.use_semantic}")
    print(f"  Lambda: {config.lambda_contrastive} | Seed: {config.seed}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir: Path = Path(f"./runs/ablation/{config.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path: Path = output_dir / "config.json"
    with open(str(config_path), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Config saved: {config_path}")

    # --------------------------------------------------
    # Load model and attach injector callbacks
    # --------------------------------------------------
    print(f"  Loading model: {config.model_variant}")
    yolo_model: YOLO = YOLO(config.model_variant)
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_model.to(device)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    injector = AblationInjector(config, device)
    yolo_model.add_callback("on_train_start", injector.on_train_start)
    yolo_model.add_callback("on_train_batch_end", injector.on_train_batch_end)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    print(f"\n  Starting training ({config.epochs} epochs, batch={config.batch_size})...")
    print(f"{'='*70}\n")

    try:
        yolo_model.train(
            data=config.data_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.img_size,
            device=device,
            amp=True,
            workers=4,
            project=str(output_dir.absolute()),
            name='weights',
            exist_ok=True,
            verbose=False,
            seed=config.seed,
        )
    except Exception as e:
        print(f"\n  ERROR during training: {e}")
        raise

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------
    summary: Dict[str, Any] = {
        "run_name": config.run_name,
        "description": config.get_run_description(),
        "config": config.to_dict(),
        "contrastive_loss": injector.get_summary(),
        "auxiliary_params": injector.aux_params,
    }
    summary_path: Path = output_dir / "summary.json"
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)

    cont_summary = injector.get_summary()
    print(f"\n  Training Summary: {config.run_name}")
    print(f"  Contrastive Loss Active: {cont_summary['contrastive_loss_active']}")
    if cont_summary['contrastive_loss_active']:
        print(f"  Mean Contrastive Loss: {cont_summary['mean_contrastive_loss']:.4f}")
    else:
        print("  Contrastive Loss: DISABLED")
    print(f"  Output: {output_dir}")
    return str(output_dir)


# ===================== ABLATION RUNNERS =====================

def run_all_ablations(args: argparse.Namespace) -> List[str]:
    configs = [
        {"use_geometric": False, "use_semantic": False, "run_name": "baseline"},
        {"use_geometric": True,  "use_semantic": False, "run_name": "geo_only"},
        {"use_geometric": False, "use_semantic": True,  "run_name": "sem_only"},
        {"use_geometric": True,  "use_semantic": True,  "run_name": "full_model"},
    ]
    output_dirs = []
    for cfg_dict in configs:
        config = AblationConfig(
            use_geometric=bool(cfg_dict['use_geometric']),
            use_semantic=bool(cfg_dict['use_semantic']),
            run_name=str(cfg_dict['run_name']),
            lambda_contrastive=float(args.lambda_contrastive),
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
        else:
            args.run_name = "full_model"
    config = AblationConfig(
        use_geometric=use_geo,
        use_semantic=use_sem,
        run_name=str(args.run_name),
        lambda_contrastive=float(args.lambda_contrastive),
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
    parser = argparse.ArgumentParser(description="CFDN-YOLO Ablation Study")
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--use_geometric', type=str, default='True')
    parser.add_argument('--use_semantic', type=str, default='True')
    parser.add_argument('--lambda_contrastive', type=float, default=0.5)
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