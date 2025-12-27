#!/bin/bash

# 出力ディレクトリの作成
mkdir -p results_experiment_optimized

# 共通設定
PY_CMD="python -m scripts.img2img"

# ベースラインとして計測する手法
# WACVを除外、Semantic(Weighted)を追加
METHODS_BASE="structural raw semantic random edge_rec edge_gt"

# 共通パラメータ設定 (SAは0.0固定)
SA_VAL="0.0"
DILATION_K="3" # Dilated Hybrid用のカーネルサイズ

echo "================================================================="
echo " STARTING OPTIMIZED RUN: SA=0.0, Focused Hybrid Ratios"
echo "================================================================="

# -----------------------------------------------------------------
# Group 1: Standard Setting (SNR -15dB, Rate 0.2)
# -----------------------------------------------------------------
echo "--- Group 1: SNR -15, Rate 0.2, StructAlpha ${SA_VAL} ---"

# [G1-Baseline] 共通ベースライン (Structural, Raw, Semantic, Edge等)
echo "Running G1 Baselines..."
$PY_CMD --snr -15 -r 0.2 --struct_alpha ${SA_VAL} \
    --target_methods $METHODS_BASE \
    > log_g1_baseline.txt 2>&1

# [G1-Hybrid] 有望な比率で Hybrid と Hybrid_Dilated を同時計測
# 1. Balanced (0.5 / 0.5)
echo "Running G1 Hybrid Balanced (0.5/0.5)..."
$PY_CMD --snr -15 -r 0.2 --struct_alpha ${SA_VAL} \
    --hybrid_alpha 0.5 --hybrid_beta 0.5 --dilation_kernel ${DILATION_K} \
    --target_methods hybrid hybrid_dilated \
    > log_g1_h05_05.txt 2>&1

# 2. Edge Bias (0.3 / 0.7) - 構造重視
echo "Running G1 Hybrid Edge-Bias (0.3/0.7)..."
$PY_CMD --snr -15 -r 0.2 --struct_alpha ${SA_VAL} \
    --hybrid_alpha 0.3 --hybrid_beta 0.7 --dilation_kernel ${DILATION_K} \
    --target_methods hybrid hybrid_dilated \
    > log_g1_h03_07.txt 2>&1


# -----------------------------------------------------------------
# Group 2: Low Bandwidth (SNR -15dB, Rate 0.1)
# -----------------------------------------------------------------
echo "--- Group 2: SNR -15, Rate 0.1 (Low Bandwidth) ---"

# [G2-Baseline]
echo "Running G2 Baselines..."
$PY_CMD --snr -15 -r 0.1 --struct_alpha ${SA_VAL} \
    --target_methods $METHODS_BASE \
    > log_g2_baseline.txt 2>&1

# [G2-Hybrid]
echo "Running G2 Hybrid Balanced (0.5/0.5)..."
$PY_CMD --snr -15 -r 0.1 --struct_alpha ${SA_VAL} \
    --hybrid_alpha 0.5 --hybrid_beta 0.5 --dilation_kernel ${DILATION_K} \
    --target_methods hybrid hybrid_dilated \
    > log_g2_h05_05.txt 2>&1

echo "Running G2 Hybrid Edge-Bias (0.3/0.7)..."
$PY_CMD --snr -15 -r 0.1 --struct_alpha ${SA_VAL} \
    --hybrid_alpha 0.3 --hybrid_beta 0.7 --dilation_kernel ${DILATION_K} \
    --target_methods hybrid hybrid_dilated \
    > log_g2_h03_07.txt 2>&1


# -----------------------------------------------------------------
# Group 3: Heavy Noise (SNR -20dB, Rate 0.2)
# -----------------------------------------------------------------
echo "--- Group 3: SNR -20, Rate 0.2 (Heavy Noise) ---"

# [G3-Baseline]
echo "Running G3 Baselines..."
$PY_CMD --snr -20 -r 0.2 --struct_alpha ${SA_VAL} \
    --target_methods $METHODS_BASE \
    > log_g3_baseline.txt 2>&1

# [G3-Hybrid]
echo "Running G3 Hybrid Balanced (0.5/0.5)..."
$PY_CMD --snr -20 -r 0.2 --struct_alpha ${SA_VAL} \
    --hybrid_alpha 0.5 --hybrid_beta 0.5 --dilation_kernel ${DILATION_K} \
    --target_methods hybrid hybrid_dilated \
    > log_g3_h05_05.txt 2>&1

echo "Running G3 Hybrid Edge-Bias (0.3/0.7)..."
$PY_CMD --snr -20 -r 0.2 --struct_alpha ${SA_VAL} \
    --hybrid_alpha 0.3 --hybrid_beta 0.7 --dilation_kernel ${DILATION_K} \
    --target_methods hybrid hybrid_dilated \
    > log_g3_h03_07.txt 2>&1

echo "All optimized experiments completed."
#tail -f log_g2_sa00_r0.1.txt
# nohup ./run.sh > main_process.log 2>&1 &
#python -m scripts.img2img --snr -13 -r 0.1