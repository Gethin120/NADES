#!/usr/bin/env bash

# set -euo pipefail

ROOT_DIR="/home/workspace/gethin_learn/DPHSF/experiment/NADES"
MAIN_PY="${ROOT_DIR}/main.py"

# 创建临时目录并设置 TMPDIR 环境变量
TMP_DIR="${ROOT_DIR}/tmp"
mkdir -p "$TMP_DIR"
export TMPDIR="$TMP_DIR"
export TEMP="$TMP_DIR"
export TMP="$TMP_DIR"


# base

# baseline
# python3 "$MAIN_PY" --model "PATE" --dataset "yelp" --epsilon 20
# python3 "$MAIN_PY" --model "ScalePATE" --dataset "yelp" --epsilon 20
# python3 "$MAIN_PY" --model "PrivGNN" --dataset "yelp" --epsilon 20







# === epsilon=0.5 最佳配置（修复-1标签+减少教师数量到10）===
# 配置：confidence_threshold=4, gaussian_noise_scale=70.0, num_teachers=10
# 性能：AUC=0.6429, Recall=0.395, Gmean=0.5559, F1_macro=0.516
# 说明：减少教师数量后，生成7个子图（vs之前18个教师时只有9个），子图更大，更多节点参与训练
# python3 "$MAIN_PY" --model "ScalePATE" --dataset "elliptic" --epsilon 0.5 --confidence_threshold 4 --gaussian_noise_scale 70.0 --num_teachers 10 --detection_epsilon 0.001

# === epsilon=1.0 最佳配置（修复-1标签+减少教师数量到10）===
# 配置：confidence_threshold=5, gaussian_noise_scale=70.0, num_teachers=10
# 性能：AUC=0.5869, Recall=0.4202, Gmean=0.5517, F1_macro=0.4883
# 说明：减少教师数量后性能大幅提升（recall从0.2605提升到0.4202）
python3 "$MAIN_PY" --model "ScalePATE" --dataset "elliptic" --epsilon 1.0 --confidence_threshold 5 --gaussian_noise_scale 70.0 --num_teachers 10 --detection_epsilon 0.001

# === epsilon=2.0 最佳配置（修复-1标签+减少教师数量到10）===
# 配置：confidence_threshold=4, gaussian_noise_scale=60.0, num_teachers=10
# 性能：AUC=0.561, Recall=0.1176, Gmean=0.3293, F1_macro=0.517
# 说明：降低confidence_threshold到4以适应7个教师（需要4/7=57%一致）
# python3 "$MAIN_PY" --model "ScalePATE" --dataset "elliptic" --epsilon 2.0 --confidence_threshold 4 --gaussian_noise_scale 60.0 --num_teachers 10 --detection_epsilon 0.001

# === epsilon=5.0 配置（减少教师数量到10，需要进一步优化）===
# 配置：confidence_threshold=4, gaussian_noise_scale=60.0, num_teachers=10
# 性能：待测试
# python3 "$MAIN_PY" --model "ScalePATE" --dataset "elliptic" --epsilon 5.0 --confidence_threshold 4 --gaussian_noise_scale 60.0 --num_teachers 10 --detection_epsilon 0.001

# === epsilon=10.0 配置（减少教师数量到10，需要进一步优化）===
# 配置：confidence_threshold=4, gaussian_noise_scale=60.0, num_teachers=10
# 性能：待测试
# python3 "$MAIN_PY" --model "ScalePATE" --dataset "elliptic" --epsilon 10.0 --confidence_threshold 4 --gaussian_noise_scale 60.0 --num_teachers 10 --detection_epsilon 0.001

# === epsilon=30.0 配置（减少教师数量到10，需要进一步优化）===
# 配置：confidence_threshold=4, gaussian_noise_scale=50.0, num_teachers=10
# 性能：待测试
# python3 "$MAIN_PY" --model "ScalePATE" --dataset "elliptic" --epsilon 30.0 --confidence_threshold 4 --gaussian_noise_scale 50.0 --num_teachers 10 --detection_epsilon 0.001








