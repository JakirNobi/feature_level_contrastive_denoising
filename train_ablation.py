#!/usr/bin/env python3
"""
Ablation Study: Contrastive Denoising Neck for YOLO26s
Trains on full Wider Face dataset (train + val splits).

Configurations:
  1. Baseline      (no auxiliary encoders, noise augmentation only)
  2. Geometric Only (geometric encoder only)
  3. Semantic Only  (semantic encoder only)
  4. Full Model    (geometric + semantic encoders)
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).parent))

from contrastive_denoise_yolo26s import MultiHook
from utils.noise import add_noise, NoiseConfig


# ===================== ABLATION CONFIGURATION =====================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
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
    epochs: int = 300
    batch_size: int = 8
    img_size: int = 640

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)

    def get_run_description(self) -> str:
        """Human-readable description of the ablation variant."""
        if not self.use_geometric and not self.use_semantic:
            return "Baseline (no auxiliary encoders)"
        if self.use_geometric and not self.use_semantic:
            return "Geometric Encoder Only"
        if not self.use_geometric and self.use_semantic:
            return "Semantic Encoder Only"
        return "Full Model (Geometric + Semantic)"


# ===================== TRAINING CALLBACK =====================

class ContrastiveLossCallback:
    """
    Callback invoked after each training batch.
    Computes contrastive loss from the MultiHook and adds it to total loss.
    """

    def __init__(self, hook: MultiHook, lambda_val: float, run_name: str) -> None:
        self.hook: MultiHook = hook
        self.lambda_val: float = lambda_val
        self.run_name: str = run_name
        self.cont_losses: List[float] = []
        self.batch_count: int = 0

    def __call__(self, trainer: Any) -> None:
        """Called by Ultralytics after each training batch."""
        cont_loss: torch.Tensor = self.hook.compute_loss()
        self.batch_count += 1

        if cont_loss > 0:
            trainer.loss = trainer.loss + self.lambda_val * cont_loss
            self.cont_losses.append(cont_loss.item())

        # Log every 100 batches
        if self.batch_count % 100 == 0 and self.cont_losses:
            avg_loss: float = float(np.mean(self.cont_losses[-100:]))
            print(
                f"  [{self.run_name}] Batch {self.batch_count}: "
                f"Avg Contrastive Loss = {avg_loss:.4f}"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for thesis documentation."""
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
            "min_contrastive_loss": float(np.min(self.cont_losses)),
            "max_contrastive_loss": float(np.max(self.cont_losses)),
        }


# ===================== MODEL UTILITIES =====================

def _get_detection_model(yolo_model: YOLO) -> DetectionModel:
    """
    Extract the underlying DetectionModel from a YOLO wrapper.
    Performs type narrowing to satisfy PyLance.

    Args:
        yolo_model: Ultralytics YOLO model instance

    Returns:
        The internal DetectionModel

    Raises:
        RuntimeError: If inner model is None
        TypeError: If inner model is not a DetectionModel
    """
    inner_model: Any = yolo_model.model

    if inner_model is None:
        raise RuntimeError("YOLO model has no inner model attribute")

    if not isinstance(inner_model, DetectionModel):
        raise TypeError(
            f"Expected DetectionModel, got {type(inner_model).__name__}"
        )

    return inner_model


def _register_hooks(
    det_model: DetectionModel,
    hook: MultiHook,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward hooks on YOLO26s neck output layers.

    Layer mapping for YOLO26s:
      - Layer 16: C3k2 -> P3 output (1/8 scale, 128 channels)
      - Layer 19: C3k2 -> P4 output (1/16 scale, 256 channels)
      - Layer 22: C3k2 -> P5 output (1/32 scale, 512 channels)

    Args:
        det_model: The DetectionModel instance
        hook: MultiHook instance with hook_p3, hook_p4, hook_p5 callbacks

    Returns:
        List of removable hook handles
    """
    model_layers: nn.Sequential = det_model.model

    handles: List[torch.utils.hooks.RemovableHandle] = [
        model_layers[16].register_forward_hook(hook.hook_p3),  # type: ignore[arg-type]
        model_layers[19].register_forward_hook(hook.hook_p4),  # type: ignore[arg-type]
        model_layers[22].register_forward_hook(hook.hook_p5),  # type: ignore[arg-type]
    ]

    return handles


# ===================== TRAINING FUNCTION =====================

def train_ablation(config: AblationConfig) -> str:
    """
    Run a single ablation experiment.

    Args:
        config: AblationConfig with all hyperparameters

    Returns:
        Path to the output directory containing trained model and logs
    """
    print(f"\n{'='*70}")
    print(f"  ABLATION EXPERIMENT: {config.run_name}")
    print(f"  Description: {config.get_run_description()}")
    print(f"  Geometric: {config.use_geometric} | Semantic: {config.use_semantic}")
    print(f"  Lambda: {config.lambda_contrastive} | Seed: {config.seed}")
    print(f"{'='*70}\n")

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create output directory with timestamp
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir: Path = Path(f"./runs/ablation/{config.run_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reproducibility
    config_path: Path = output_dir / "config.json"
    with open(str(config_path), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"  Config saved: {config_path}")

    # --------------------------------------------------
    # Load model and force GPU
    # --------------------------------------------------
    print(f"  Loading model: {config.model_variant}")
    yolo_model: YOLO = YOLO(config.model_variant)

    # Force GPU if available
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    yolo_model.to(device)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Extract DetectionModel with proper type narrowing
    det_model: DetectionModel = _get_detection_model(yolo_model)

    # --------------------------------------------------
    # Create auxiliary modules
    # --------------------------------------------------
    print("  Creating auxiliary modules...")
    hook: MultiHook = MultiHook(
        neck_channels=list(config.neck_channels),
        proj_hidden=config.proj_hidden,
        proj_out=config.proj_out,
        use_geometric=config.use_geometric,
        use_semantic=config.use_semantic,
    ).to(device)

    aux_params: int = hook.get_auxiliary_param_count()
    print(f"  Auxiliary parameters: {aux_params:,} (discarded at inference)")

    # Verify hook configuration matches ablation config
    if hook.use_geometric != config.use_geometric:
        raise RuntimeError(
            f"Hook geometric mismatch: "
            f"{hook.use_geometric} != {config.use_geometric}"
        )
    if hook.use_semantic != config.use_semantic:
        raise RuntimeError(
            f"Hook semantic mismatch: "
            f"{hook.use_semantic} != {config.use_semantic}"
        )

    # --------------------------------------------------
    # Register forward hooks on neck layers
    # --------------------------------------------------
    hook_handles: List[torch.utils.hooks.RemovableHandle] = _register_hooks(
        det_model, hook
    )
    print("  Hooks registered on layers: 16 (P3), 19 (P4), 22 (P5)")

    # --------------------------------------------------
    # Configure noise injection
    # --------------------------------------------------
    noise_cfg: NoiseConfig = NoiseConfig(
        noise_types=config.noise_types,
        noise_params=config.noise_params,
    )

    # Save original forward method
    original_forward = det_model.forward

    def custom_forward(
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Patched forward that injects noise into input images.
        Stores clean copy for auxiliary encoder reference.
        """
        clean: torch.Tensor = x.clone()
        noise_type: str = str(np.random.choice(list(noise_cfg.noise_types)))
        noise_param: float = float(
            np.random.choice(list(noise_cfg.noise_params))
        )
        noisy: torch.Tensor = add_noise(
            x,
            noise_type=noise_type,
            param=noise_param
        )
        hook.set_clean(clean)
        return original_forward(noisy, *args, **kwargs)

    # Apply the patched forward
    det_model.forward = custom_forward  # type: ignore[method-assign]
    print(f"  Patched forward for noise injection: {noise_cfg.noise_types}")

    # --------------------------------------------------
    # Register contrastive loss callback
    # --------------------------------------------------
    callback: ContrastiveLossCallback = ContrastiveLossCallback(
        hook=hook,
        lambda_val=config.lambda_contrastive,
        run_name=config.run_name,
    )
    yolo_model.add_callback("on_train_batch_end", callback)

    # --------------------------------------------------
    # Train the model
    # --------------------------------------------------
    print(
        f"\n  Starting training "
        f"({config.epochs} epochs, batch={config.batch_size})..."
    )
    print(f"{'='*70}\n")

    try:
        results: Any = yolo_model.train(
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
    finally:
        # Always restore original forward and remove hooks
        det_model.forward = original_forward  # type: ignore[method-assign]
        for h in hook_handles:
            h.remove()
        print("  Restored original forward and removed hooks.")

    # --------------------------------------------------
    # Save outputs and summary
    # --------------------------------------------------
    model_path: Path = output_dir / "weights" / "best.pt"
    if model_path.exists():
        print(f"  Model saved: {model_path}")
    else:
        print(f"  WARNING: Model not found at {model_path}")

    summary: Dict[str, Any] = {
        "run_name": config.run_name,
        "description": config.get_run_description(),
        "config": config.to_dict(),
        "contrastive_loss": callback.get_summary(),
        "auxiliary_params": aux_params,
    }

    summary_path: Path = output_dir / "summary.json"
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    cont_summary: Dict[str, Any] = callback.get_summary()
    print(f"\n  {'='*50}")
    print(f"  Training Summary: {config.run_name}")
    print(f"  {'='*50}")
    print(f"  Contrastive Loss Active: {cont_summary['contrastive_loss_active']}")
    if cont_summary['contrastive_loss_active']:
        print(f"  Mean Contrastive Loss:    {cont_summary['mean_contrastive_loss']:.4f}")
        print(f"  Std Contrastive Loss:     {cont_summary['std_contrastive_loss']:.4f}")
        print(f"  Batches with Loss:        {cont_summary['num_batches_with_loss']}")
    else:
        print("  Contrastive Loss: DISABLED (baseline mode)")
    print(f"  Auxiliary Params:         {aux_params:,}")
    print(f"  Output:                   {output_dir}")
    print(f"  {'='*50}\n")

    return str(output_dir)


# ===================== ABLATION RUNNERS =====================

def run_all_ablations(args: argparse.Namespace) -> List[str]:
    """
    Run all four ablation configurations sequentially.

    Order:
      1. Baseline (no encoders)
      2. Geometric Only
      3. Semantic Only
      4. Full Model (both encoders)
    """
    configs: List[Dict[str, Any]] = [
        {"use_geometric": False, "use_semantic": False, "run_name": "baseline"},
        {"use_geometric": True,  "use_semantic": False, "run_name": "geo_only"},
        {"use_geometric": False, "use_semantic": True,  "run_name": "sem_only"},
        {"use_geometric": True,  "use_semantic": True,  "run_name": "full_model"},
    ]

    output_dirs: List[str] = []

    for i, cfg_dict in enumerate(configs):
        print(f"\n\n{'#'*70}")
        print(f"  ABLATION RUN {i + 1} / 4: {cfg_dict['run_name']}")
        print(f"{'#'*70}")

        # Parse noise parameters
        noise_params_str: str = str(args.noise_params)
        noise_params: Tuple[int, ...] = tuple(
            int(p.strip()) for p in noise_params_str.split(',') if p.strip()
        ) if noise_params_str else (10, 25, 50)

        # Parse noise types
        noise_types_str: str = str(args.noise_types)
        noise_types: Tuple[str, ...] = tuple(
            t.strip() for t in noise_types_str.split(',') if t.strip()
        ) if noise_types_str else ('gaussian',)

        config: AblationConfig = AblationConfig(
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
            noise_types=noise_types,
            noise_params=noise_params,
        )

        output_dir: str = train_ablation(config)
        output_dirs.append(output_dir)

    return output_dirs


def run_single_ablation(args: argparse.Namespace) -> str:
    """
    Run a single ablation experiment based on CLI arguments.
    Auto-detects run name if not provided.
    """
    use_geo: bool = str(args.use_geometric).lower() in ('true', '1', 'yes')
    use_sem: bool = str(args.use_semantic).lower() in ('true', '1', 'yes')

    # Auto-detect run name
    if args.run_name is None:
        if not use_geo and not use_sem:
            args.run_name = "baseline"
        elif use_geo and not use_sem:
            args.run_name = "geo_only"
        elif not use_geo and use_sem:
            args.run_name = "sem_only"
        else:
            args.run_name = "full_model"

    # Parse noise parameters
    noise_params_str: str = str(args.noise_params)
    noise_params: Tuple[int, ...] = tuple(
        int(p.strip()) for p in noise_params_str.split(',') if p.strip()
    ) if noise_params_str else (10, 25, 50)

    # Parse noise types
    noise_types_str: str = str(args.noise_types)
    noise_types: Tuple[str, ...] = tuple(
        t.strip() for t in noise_types_str.split(',') if t.strip()
    ) if noise_types_str else ('gaussian',)

    config: AblationConfig = AblationConfig(
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
        noise_types=noise_types,
        noise_params=noise_params,
    )

    return train_ablation(config)


# ===================== MAIN ENTRY POINT =====================

def main() -> None:
    """Parse command-line arguments and run ablation study."""
    parser = argparse.ArgumentParser(
        description="Ablation Study: Contrastive Denoising Neck for YOLO26s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 4 ablation configurations
  python train_ablation.py --all

  # Run only the full model
  python train_ablation.py --use_geometric True --use_semantic True

  # Run baseline (no auxiliary encoders)
  python train_ablation.py --use_geometric False --use_semantic False

  # Geometric encoder only
  python train_ablation.py --use_geometric True --use_semantic False

  # Semantic encoder only
  python train_ablation.py --use_geometric False --use_semantic True

  # Custom hyperparameters
  python train_ablation.py --all --lambda_contrastive 0.3 --epochs 100
        """
    )

    # Run mode
    parser.add_argument(
        '--all', action='store_true',
        help='Run all 4 ablation configurations sequentially'
    )

    # Encoder toggles
    parser.add_argument(
        '--use_geometric', type=str, default='True',
        help='Enable Geometric Encoder (True/False)'
    )
    parser.add_argument(
        '--use_semantic', type=str, default='True',
        help='Enable Semantic Encoder (True/False)'
    )

    # Loss hyperparameters
    parser.add_argument(
        '--lambda_contrastive', type=float, default=0.5,
        help='Weight for contrastive loss in total loss'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.07,
        help='Temperature parameter for InfoNCE loss'
    )
    parser.add_argument(
        '--num_samples', type=int, default=1024,
        help='Number of spatial positions to sample per scale'
    )

    # Model and data
    parser.add_argument(
        '--model', type=str, default='yolo26s.pt',
        help='Base YOLO model to use'
    )
    parser.add_argument(
        '--data', type=str, default='widerface.yaml',
        help='Path to dataset YAML file'
    )

    # Training settings
    parser.add_argument(
        '--epochs', type=int, default=300,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--img_size', type=int, default=640,
        help='Input image size'
    )

    # Noise configuration
    parser.add_argument(
        '--noise_types', type=str, default='gaussian',
        help='Comma-separated noise types (gaussian,poisson,salt_pepper,speckle)'
    )
    parser.add_argument(
        '--noise_params', type=str, default='10,25,50',
        help='Comma-separated noise intensity parameters'
    )

    # Experiment tracking
    parser.add_argument(
        '--run_name', type=str, default=None,
        help='Custom run name (auto-detected from encoder flags if not set)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    args: argparse.Namespace = parser.parse_args()

    # Print system information
    print(f"\n{'='*70}")
    print("  CONTRASTIVE DENOISING ABLATION STUDY")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Lambda (contrastive): {args.lambda_contrastive}")
    print(f"  Noise Types: {args.noise_types}")
    print(f"  Noise Params: {args.noise_params}")

    # Force GPU info
    if torch.cuda.is_available():
        gpu_name: str = torch.cuda.get_device_name(0)
        vram_gb: float = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB)")
    else:
        print("  GPU: None (using CPU)")
    print(f"{'='*70}\n")

    # Create base output directory
    os.makedirs("./runs/ablation", exist_ok=True)

    # Run experiments
    if args.all:
        output_dirs: List[str] = run_all_ablations(args)
        print(f"\n{'='*70}")
        print("  ALL ABLATION RUNS COMPLETE")
        print(f"  Output directories:")
        for d in output_dirs:
            print(f"    {d}")
        print(f"{'='*70}\n")
    else:
        output_dir: str = run_single_ablation(args)
        print(f"\n  Run complete: {output_dir}\n")


if __name__ == '__main__':
    main()