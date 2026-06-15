#!/usr/bin/env bash
set -euo pipefail

# Small pareto_variance grid:
# - model: gpt2
# - task: sva
# - metric: prob_diff
# - attribution styles: ceap, eap-ig
# - div-for-scoring: false
# - seeds: 0, 1, 2, 3, run sequentially on one GPU
# - visualization: resampling-variance PJI and unfaithfulness plots

MODEL="gpt2"
TASK="sva"
METRIC="prob_diff"
BATCH_SIZE=256
IG_STYLES=("ceap" "eap-ig")
SEEDS=(0 1 2 3)
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
RESULT_FOLDER="results_resampling_variance"

for ig_style in "${IG_STYLES[@]}"; do
  echo "Running ${MODEL} ${TASK} ${METRIC} ${ig_style} for seeds sequentially on GPU ${GPU}: ${SEEDS[*]}"
  for seed in "${SEEDS[@]}"; do
    echo "  seed=${seed}"
    CUDA_VISIBLE_DEVICES="${GPU}" python pareto_variance.py \
      --model="${MODEL}" \
      --task="${TASK}" \
      --metric="${METRIC}" \
      --batch-size="${BATCH_SIZE}" \
      --ig-style="${ig_style}" \
      --seed="${seed}" \
      --result-folder="${RESULT_FOLDER}"
  done
done

python visualization/plot_resampling_variance.py \
  --results-root="${RESULT_FOLDER}" \
  --model="${MODEL}" \
  --task="${TASK}" \
  --eval-dir="graph_eval_with_${METRIC}" \
  --setups ceap_20 eap-ig_20
