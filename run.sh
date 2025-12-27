#!/bin/bash

# 出力ディレクトリの作成
mkdir -p results_experiment

echo "================================================================="
echo " STARTING EXPERIMENT RUN (Total 30 runs)"
echo "================================================================="

# -----------------------------------------------------------------
# Group 1: The Golden Ratio Search (SNR -15dB, Rate 0.2) [10 runs]
# 目的: Hybrid手法における Alpha(不確実性) vs Beta(エッジ) の最適なバランスを見つける
# 設定: struct_alpha は "0.3" (ランダム性あり) で固定して探索します
# -----------------------------------------------------------------
echo "--- Group 1: Finding Optimal Hybrid Ratio (SNR -15, R 0.2, StructAlpha 0.3) ---"

# 1. Pure Uncertainty Bias
echo "Running Exp 1: Hybrid Alpha 1.0 (Pure Uncertainty)..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 1.0 --hybrid_beta 0.0 > log_g1_h10_00.txt 2>&1

# 2. Strong Uncertainty
echo "Running Exp 2: Hybrid Alpha 0.9 / Beta 0.1..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.9 --hybrid_beta 0.1 > log_g1_h09_01.txt 2>&1

# 3. Moderate Uncertainty
echo "Running Exp 3: Hybrid Alpha 0.8 / Beta 0.2..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.8 --hybrid_beta 0.2 > log_g1_h08_02.txt 2>&1

# 4. Proposed Baseline (本命: ランダムあり)
echo "Running Exp 4: Hybrid Alpha 0.7 / Beta 0.3 (Proposed, SA=0.3)..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g1_h07_03.txt 2>&1

# 5. Slight Uncertainty
echo "Running Exp 5: Hybrid Alpha 0.6 / Beta 0.4..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.6 --hybrid_beta 0.4 > log_g1_h06_04.txt 2>&1

# 6. Balanced
echo "Running Exp 6: Hybrid Alpha 0.5 / Beta 0.5..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.5 --hybrid_beta 0.5 > log_g1_h05_05.txt 2>&1

# 7. Slight Edge Bias
echo "Running Exp 7: Hybrid Alpha 0.4 / Beta 0.6..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.4 --hybrid_beta 0.6 > log_g1_h04_06.txt 2>&1

# 8. Edge Bias
echo "Running Exp 8: Hybrid Alpha 0.3 / Beta 0.7..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.3 --hybrid_beta 0.7 > log_g1_h03_07.txt 2>&1

# 9. Strong Edge Bias
echo "Running Exp 9: Hybrid Alpha 0.1 / Beta 0.9..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.1 --hybrid_beta 0.9 > log_g1_h01_09.txt 2>&1

# 10. Pure Edge (Comparison)
echo "Running Exp 10: Hybrid Alpha 0.0 (Pure Edge)..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.0 --hybrid_beta 1.0 > log_g1_h00_10.txt 2>&1


# -----------------------------------------------------------------
# Group 2: Struct Alpha 0.0 vs 0.3 Showdown [6 runs]
# 目的: ランダムノイズなし(0.0)の方が良い可能性を検証します。
# Group 1の結果(0.3)と比較するための「対抗馬」を走らせます。
# -----------------------------------------------------------------
echo "--- Group 2: Comparison with Struct Alpha 0.0 (No Randomness) ---"

# 11. Proposed (0.7/0.3) with SA=0.0  VS  Exp 4 (SA=0.3)
echo "Running Exp 11: Hybrid 0.7/0.3 with StructAlpha 0.0..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.0 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g2_sa00_h07_03.txt 2>&1

# 12. Balanced (0.5/0.5) with SA=0.0  VS  Exp 6 (SA=0.3)
echo "Running Exp 12: Hybrid 0.5/0.5 with StructAlpha 0.0..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.0 --hybrid_alpha 0.5 --hybrid_beta 0.5 > log_g2_sa00_h05_05.txt 2>&1

# 13. High Uncertainty (0.9/0.1) with SA=0.0 VS Exp 2
echo "Running Exp 13: Hybrid 0.9/0.1 with StructAlpha 0.0..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.0 --hybrid_alpha 0.9 --hybrid_beta 0.1 > log_g2_sa00_h09_01.txt 2>&1

# 14. Pure Uncertainty (1.0/0.0) with SA=0.0 VS Exp 1
echo "Running Exp 14: Pure Uncertainty with StructAlpha 0.0..."
python -m scripts.img2img --snr -15 -r 0.2 --struct_alpha 0.0 --hybrid_alpha 1.0 --hybrid_beta 0.0 > log_g2_sa00_h10_00.txt 2>&1

# 15. Robustness Check 1: Heavy Noise (-20dB) with SA=0.0
# Group 4 (SA=0.3) との比較用
echo "Running Exp 15: Heavy Noise (-20dB) with StructAlpha 0.0 (Proposed Ratio)..."
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.0 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g2_sa00_snr-20.txt 2>&1

# 16. Robustness Check 2: Low Rate (0.1) with SA=0.0
# Group 3 (SA=0.3) との比較用
echo "Running Exp 16: Low Rate (R=0.1) with StructAlpha 0.0 (Proposed Ratio)..."
python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.0 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g2_sa00_r0.1.txt 2>&1


# -----------------------------------------------------------------
# Group 3: Low Bandwidth Efficiency (SNR -15, Rate 0.1) [5 runs]
# 設定: Struct Alpha 0.3 (Baseline)
# -----------------------------------------------------------------
echo "--- Group 3: Low Rate (R 0.1) Efficiency (SA=0.3) ---"

# 17. Uncertainty Focused
echo "Running Exp 17: Low Rate - Alpha 0.9 / Beta 0.1..."
python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.3 --hybrid_alpha 0.9 --hybrid_beta 0.1 > log_g3_h09_01.txt 2>&1

# 18. Proposed
echo "Running Exp 18: Low Rate - Alpha 0.7 / Beta 0.3..."
python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.3 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g3_h07_03.txt 2>&1

# 19. Balanced
echo "Running Exp 19: Low Rate - Alpha 0.5 / Beta 0.5..."
python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.3 --hybrid_alpha 0.5 --hybrid_beta 0.5 > log_g3_h05_05.txt 2>&1

# 20. Edge Focused
echo "Running Exp 20: Low Rate - Alpha 0.3 / Beta 0.7..."
python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.3 --hybrid_alpha 0.3 --hybrid_beta 0.7 > log_g3_h03_07.txt 2>&1

# 21. Strong Edge
echo "Running Exp 21: Low Rate - Alpha 0.1 / Beta 0.9..."
python -m scripts.img2img --snr -15 -r 0.1 --struct_alpha 0.3 --hybrid_alpha 0.1 --hybrid_beta 0.9 > log_g3_h01_09.txt 2>&1


# -----------------------------------------------------------------
# Group 4: Heavy Noise Robustness (SNR -20, Rate 0.2) [5 runs]
# 設定: Struct Alpha 0.3 (Baseline)
# -----------------------------------------------------------------
echo "--- Group 4: Heavy Noise (SNR -20) Robustness (SA=0.3) ---"

# 22. Uncertainty Focused
echo "Running Exp 22: Heavy Noise - Alpha 0.9 / Beta 0.1..."
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.9 --hybrid_beta 0.1 > log_g4_h09_01.txt 2>&1

# 23. Proposed
echo "Running Exp 23: Heavy Noise - Alpha 0.7 / Beta 0.3..."
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g4_h07_03.txt 2>&1

# 24. Balanced
echo "Running Exp 24: Heavy Noise - Alpha 0.5 / Beta 0.5..."
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.5 --hybrid_beta 0.5 > log_g4_h05_05.txt 2>&1

# 25. Edge Focused
echo "Running Exp 25: Heavy Noise - Alpha 0.3 / Beta 0.7..."
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.3 --hybrid_beta 0.7 > log_g4_h03_07.txt 2>&1

# 26. Strong Edge
echo "Running Exp 26: Heavy Noise - Alpha 0.1 / Beta 0.9..."
python -m scripts.img2img --snr -20 -r 0.2 --struct_alpha 0.3 --hybrid_alpha 0.1 --hybrid_beta 0.9 > log_g4_h01_09.txt 2>&1


# -----------------------------------------------------------------
# Group 5: Light Noise / High Quality (SNR -10, Rate 0.05) [4 runs]
# 設定: Struct Alpha 0.3 (Baseline)
# -----------------------------------------------------------------
echo "--- Group 5: Light Noise (SNR -10, R 0.05) (SA=0.3) ---"

# 27. High U
echo "Running Exp 27: Light Noise - Alpha 0.8 / Beta 0.2..."
python -m scripts.img2img --snr -10 -r 0.05 --struct_alpha 0.3 --hybrid_alpha 0.8 --hybrid_beta 0.2 > log_g5_h08_02.txt 2>&1

# 28. Proposed
echo "Running Exp 28: Light Noise - Alpha 0.7 / Beta 0.3..."
python -m scripts.img2img --snr -10 -r 0.05 --struct_alpha 0.3 --hybrid_alpha 0.7 --hybrid_beta 0.3 > log_g5_h07_03.txt 2>&1

# 29. Balanced
echo "Running Exp 29: Light Noise - Alpha 0.5 / Beta 0.5..."
python -m scripts.img2img --snr -10 -r 0.05 --struct_alpha 0.3 --hybrid_alpha 0.5 --hybrid_beta 0.5 > log_g5_h05_05.txt 2>&1

# 30. Edge Heavy
echo "Running Exp 30: Light Noise - Alpha 0.2 / Beta 0.8..."
python -m scripts.img2img --snr -10 -r 0.05 --struct_alpha 0.3 --hybrid_alpha 0.2 --hybrid_beta 0.8 > log_g5_h02_08.txt 2>&1

echo "All 30 experiments completed."

#tail -f log_g5_h07_03.txt
# nohup ./run.sh > main_process.log 2>&1 &
#python -m scripts.img2img --snr -13 -r 0.1