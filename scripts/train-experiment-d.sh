#!/usr/bin/env bash
# Manual entry point for this server. Never invoked by setup or tests.
set -euo pipefail
captcha_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
captcha_resume=()
if [[ $# == 1 && "$1" == --resume ]]; then
    captcha_resume=(--resume outputs/runs/crnn-v2-20k-cuda-plateau/last.pt)
elif [[ $# != 0 ]]; then
    printf '%s\n' 'Uso: ./scripts/train-experiment-d.sh [--resume]' >&2
    exit 2
fi
cd -- "$captcha_root"
# Physical GPU 0 verified on this server; UUID keeps selection explicit.
export CUDA_VISIBLE_DEVICES=GPU-bc252296-7ce0-02fb-acbd-4f371c943715
./.venv-ml/bin/python generator/prepare_experiment_d.py --verify-only
exec ./.venv-ml/bin/python -m ml.train \
    --data dataset/synthetic/v2_20k_train \
    --device cuda \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --seed 42 \
    --scheduler plateau \
    --plateau-start-epoch 10 \
    --plateau-patience 8 \
    --plateau-factor 0.5 \
    --plateau-min-delta 0.002 \
    --plateau-cooldown 2 \
    --plateau-min-lr 0.00001 \
    --early-stopping-patience 25 \
    --early-stopping-min-delta 0.002 \
    --early-stopping-start-epoch 11 \
    --run-dir outputs/runs/crnn-v2-20k-cuda-plateau \
    "${captcha_resume[@]}"
