#!/usr/bin/env bash

# set -euo pipefail

ROOT_DIR="/home/workspace/gethin_learn/DPHSF/experiment/NADES"
MAIN_PY="${ROOT_DIR}/main.py"

TMP_DIR="${ROOT_DIR}/tmp"
mkdir -p "$TMP_DIR"
export TMPDIR="$TMP_DIR"
export TEMP="$TMP_DIR"
export TMP="$TMP_DIR"

python3 "$MAIN_PY" --model "NADES" --dataset "elliptic" --epsilon 1.0 --confidence_threshold 5 --gaussian_noise_scale 70.0 --num_teachers 10 --detection_epsilon 0.001








