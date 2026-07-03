#!/bin/bash
set -e
echo "============================================================"
echo "  CONTRASTIVE DENOISING ABLATION STUDY"
echo "  Model: YOLO26s | Dataset: Wider Face (Full)"
echo "============================================================"
echo ""

python train_ablation.py \
    --all \
    --lambda_contrastive 0.5 \
    --seed 42 \
    --epochs 300 \
    --batch_size 8 \
    --noise_types "gaussian" \
    --noise_params "10,25,50" \
    --data "widerface.yaml" \
    --model "yolo26s.pt"

echo ""
echo "============================================================"
echo "  ALL ABLATION RUNS COMPLETE"
echo "============================================================"

echo ""
echo "Running evaluation..."
python evaluate_ablation.py \
    --model_dir ./runs/ablation \
    --data_dir ./datasets/widerface/images/val \
    --output ./evaluation_results.json

echo ""
echo "Generating analysis..."
python analyze_results.py \
    --results ./evaluation_results.json \
    --output_dir ./analysis

echo ""
echo "============================================================"
echo "  DONE"
echo "============================================================"