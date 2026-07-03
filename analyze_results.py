#!/usr/bin/env python3
"""
Analysis and visualization of ablation study results.
Generates tables and plots for thesis documentation.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import argparse


def plot_noise_robustness(results: Dict[str, Any], output_path: str) -> None:
    """Plot AP across noise levels for each model variant."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors: Dict[str, str] = {
        'baseline': 'red', 'geo_only': 'blue',
        'sem_only': 'green', 'full_model': 'purple',
    }
    markers: Dict[str, str] = {
        'baseline': 's', 'geo_only': 'd',
        'sem_only': '^', 'full_model': 'o',
    }

    for model_name, model_results in results.items():
        if 'gaussian' not in model_results:
            continue
        levels: List[float] = sorted([float(k) for k in model_results['gaussian'].keys()])
        aps: List[float] = [model_results['gaussian'][str(l)] for l in levels]
        color: str = colors.get(model_name, 'gray')
        marker: str = markers.get(model_name, 'x')
        label: str = model_name.replace('_', ' ').title()
        ax.plot(levels, aps, marker=marker, color=color, linewidth=2,
                markersize=8, label=label)

    ax.set_xlabel('Gaussian Noise Level (sigma)', fontsize=14)
    ax.set_ylabel('Average Precision (AP)', fontsize=14)
    ax.set_title('Face Detection Robustness to Gaussian Noise', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_ablation_comparison(results: Dict[str, Any], output_path: str) -> None:
    """Bar chart comparing clean vs noisy performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    variants: List[str] = []
    clean_aps: List[float] = []
    noisy_aps: List[float] = []

    for model_name, model_results in results.items():
        variants.append(model_name.replace('_', ' ').title())
        clean_ap: float = float(model_results.get('clean', {}).get('0', 0.0))
        clean_aps.append(clean_ap)
        gaussian_results = model_results.get('gaussian', {})
        noisy_ap: float = float(np.mean(list(gaussian_results.values())))
        noisy_aps.append(noisy_ap)

    x: np.ndarray = np.arange(len(variants))
    width: float = 0.35
    ax.bar(x - width/2, clean_aps, width, label='Clean Images', color='green', alpha=0.8)
    ax.bar(x + width/2, noisy_aps, width, label='Noisy Images', color='red', alpha=0.8)
    ax.set_xlabel('Model Variant', fontsize=14)
    ax.set_ylabel('Average Precision (AP)', fontsize=14)
    ax.set_title('Ablation Study: Clean vs Noisy Performance', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (c, n) in enumerate(zip(clean_aps, noisy_aps)):
        ax.text(i - width/2, c + 0.01, f'{c:.3f}', ha='center', fontsize=9)
        ax.text(i + width/2, n + 0.01, f'{n:.3f}', ha='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX-formatted results table."""
    header: str = r"""
\begin{table}[h]
\centering
\caption{Ablation Study: Contrastive Denoising Components}
\label{tab:ablation}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Geometric} & \textbf{Semantic} & \textbf{Clean AP} & \textbf{Noisy AP} \\
\hline
"""
    rows: List[str] = []
    for model_name, model_results in results.items():
        geo: str = r"\checkmark" if "geo" in model_name or "full" in model_name else ""
        sem: str = r"\checkmark" if "sem" in model_name or "full" in model_name else ""
        clean_ap: float = float(model_results.get('clean', {}).get('0', 0.0))
        gaussian_results = model_results.get('gaussian', {})
        noisy_ap: float = float(np.mean(list(gaussian_results.values())))
        name: str = model_name.replace('_', ' ').title()
        rows.append(f"{name} & {geo} & {sem} & {clean_ap:.4f} & {noisy_ap:.4f} \\\\")

    footer: str = r"""\hline
\end{tabular}
\end{table}
"""
    return header + '\n'.join(rows) + '\n' + footer


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument('--results', type=str, default='./evaluation_results.json')
    parser.add_argument('--output_dir', type=str, default='./analysis')
    args = parser.parse_args()

    with open(args.results, 'r') as f:
        results: Dict[str, Any] = json.load(f)

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_noise_robustness(results, str(output_dir / 'noise_robustness.png'))
    plot_ablation_comparison(results, str(output_dir / 'ablation_comparison.png'))

    latex_table: str = generate_latex_table(results)
    latex_path: Path = output_dir / 'results_table.tex'
    with open(str(latex_path), 'w') as f:
        f.write(latex_table)
    print(f"  LaTeX table saved: {latex_path}")

    print("\n" + "="*60)
    print("  ABLATION STUDY SUMMARY")
    print("="*60)
    for model_name, model_results in results.items():
        clean_ap: float = float(model_results.get('clean', {}).get('0', 0.0))
        gaussian_aps: List[float] = list(model_results.get('gaussian', {}).values())
        noisy_ap: float = float(np.mean(gaussian_aps)) if gaussian_aps else 0.0
        print(f"\n  {model_name.replace('_', ' ').title()}:")
        print(f"    Clean AP: {clean_ap:.4f}")
        print(f"    Noisy AP (avg): {noisy_ap:.4f}")
        if gaussian_aps:
            print(f"    Robustness Drop: {clean_ap - noisy_ap:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()