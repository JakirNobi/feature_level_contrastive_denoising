#!/usr/bin/env python3
"""
Evaluation script for ablation study models.
Tests each model across multiple noise types and intensities.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import argparse
from tqdm import tqdm
import cv2

from ultralytics import YOLO
from utils.noise import add_noise


@dataclass
class EvalConfig:
    noise_types: Tuple[str, ...] = ('clean', 'gaussian', 'poisson', 'salt_pepper')
    noise_levels: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    img_size: int = 640

    def __post_init__(self) -> None:
        if not self.noise_levels:
            self.noise_levels = {
                'clean': (0,),
                'gaussian': (10, 25, 50, 75),
                'poisson': (1,),
                'salt_pepper': (1, 3, 5, 10),
            }


def compute_ap(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> float:
    """Compute Average Precision for face detection."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0:
        return 0.0

    pred_boxes = pred_boxes[pred_boxes[:, 4].argsort()[::-1]]
    tp: np.ndarray = np.zeros(len(pred_boxes))
    fp: np.ndarray = np.zeros(len(pred_boxes))
    gt_matched: np.ndarray = np.zeros(len(gt_boxes))

    for i, pred in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        x1: np.ndarray = np.maximum(pred[0], gt_boxes[:, 0])
        y1: np.ndarray = np.maximum(pred[1], gt_boxes[:, 1])
        x2: np.ndarray = np.minimum(pred[2], gt_boxes[:, 2])
        y2: np.ndarray = np.minimum(pred[3], gt_boxes[:, 3])
        inter: np.ndarray = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_pred: float = float((pred[2] - pred[0]) * (pred[3] - pred[1]))
        area_gt: np.ndarray = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        union: np.ndarray = area_pred + area_gt - inter
        ious: np.ndarray = inter / (union + 1e-10)

        max_iou: float = float(np.max(ious))
        max_idx: int = int(np.argmax(ious))

        if max_iou >= iou_threshold and gt_matched[max_idx] == 0:
            tp[i] = 1
            gt_matched[max_idx] = 1
        else:
            fp[i] = 1

    tp_cumsum: np.ndarray = np.cumsum(tp)
    fp_cumsum: np.ndarray = np.cumsum(fp)
    recalls: np.ndarray = tp_cumsum / max(len(gt_boxes), 1)
    precisions: np.ndarray = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    ap: float = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p: float = 0.0
        else:
            p = float(np.max(precisions[recalls >= t]))
        ap += p / 11.0

    return ap


def evaluate_model(
    model_path: str,
    data_dir: str,
    config: EvalConfig,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained model across multiple noise conditions."""
    model: YOLO = YOLO(model_path)
    results: Dict[str, Dict[str, float]] = {}

    img_dir: Path = Path(data_dir)
    image_files: List[Path] = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    label_dir: Path = img_dir.parent.parent / 'labels' / img_dir.name

    for noise_type in config.noise_types:
        results[noise_type] = {}
        levels: Tuple[float, ...] = config.noise_levels.get(noise_type, (0,))

        for level in levels:
            print(f"  Evaluating: {noise_type} (level={level})")
            all_ap: List[float] = []

            for img_path in tqdm(image_files, desc=f"    {noise_type}_{level}"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (config.img_size, config.img_size))
                img_tensor: torch.Tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)

                if noise_type != 'clean':
                    img_tensor = add_noise(img_tensor, noise_type, level)

                with torch.no_grad():
                    preds = model(img_tensor.to(device), verbose=False)

                pred_boxes_list: List[List[float]] = []
                if preds[0].boxes is not None:
                    boxes_np: np.ndarray = preds[0].boxes.xyxy.cpu().numpy()
                    confs_np: np.ndarray = preds[0].boxes.conf.cpu().numpy()
                    mask: np.ndarray = confs_np >= config.confidence_threshold
                    valid_boxes: np.ndarray = boxes_np[mask]
                    valid_confs: np.ndarray = confs_np[mask]
                    for j in range(len(valid_boxes)):
                        pred_boxes_list.append([
                            float(valid_boxes[j][0]), float(valid_boxes[j][1]),
                            float(valid_boxes[j][2]), float(valid_boxes[j][3]),
                            float(valid_confs[j])
                        ])

                pred_boxes: np.ndarray = np.array(pred_boxes_list)

                label_path: Path = label_dir / f"{img_path.stem}.txt"
                gt_list: List[List[float]] = []
                if label_path.exists():
                    h, w = img.shape[:2]
                    with open(str(label_path), 'r') as f:
                        for line in f:
                            parts: List[str] = line.strip().split()
                            if len(parts) >= 5:
                                cx, cy, nw, nh = map(float, parts[1:5])
                                gt_list.append([
                                    (cx - nw/2) * w, (cy - nh/2) * h,
                                    (cx + nw/2) * w, (cy + nh/2) * h
                                ])

                gt_boxes: np.ndarray = np.array(gt_list)
                ap: float = compute_ap(pred_boxes, gt_boxes, config.iou_threshold)
                all_ap.append(ap)

            results[noise_type][str(level)] = float(np.mean(all_ap))
            print(f"    AP: {results[noise_type][str(level)]:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation study models")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='./evaluation_results.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config: EvalConfig = EvalConfig()
    model_dir: Path = Path(args.model_dir)
    model_paths: List[Path] = list(model_dir.rglob('best.pt'))

    if not model_paths:
        print(f"No model weights found in {model_dir}")
        return

    print(f"Found {len(model_paths)} models to evaluate")
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_path in model_paths:
        run_name: str = model_path.parent.parent.name
        print(f"\nEvaluating: {run_name}")
        results: Dict[str, Dict[str, float]] = evaluate_model(
            str(model_path), args.data_dir, config, args.device
        )
        all_results[run_name] = results

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()