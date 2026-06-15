#!/usr/bin/env bash
set -euo pipefail

# Small single-sample workflow:
# - model: gpt2
# - task/metric: sva + prob_diff
# - attribution style: ceap
# - div-for-scoring: false
# - all-averaging: false
# - parallel batch size: 20
# - visualization: template UMAP/PJI and plural_1 diagnostics

MODEL="gpt2"
TASK="sva"
METRIC="prob_diff"
IG_STYLE="ceap"
INTEGRATED_STEPS=20
SEED=4
PARALLEL_BATCH_SIZE=20
RESULT_FOLDER="results_rephrasing_samplewise_variance"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
TARGET_EDGES=1000
TEMPLATE_LABEL="plural_1"

CUDA_VISIBLE_DEVICES="${GPU}" python pareto_single_sample_analysis.py \
  --model="${MODEL}" \
  --task="${TASK}" \
  --metric="${METRIC}" \
  --ig-style="${IG_STYLE}" \
  --integrated-steps="${INTEGRATED_STEPS}" \
  --seed="${SEED}" \
  --parallel-batch-size="${PARALLEL_BATCH_SIZE}" \
  --result-folder="${RESULT_FOLDER}" \
  --no-all-averaging

python visualization/plot_rephrasing_samplewise_variance.py \
  --results-root="${RESULT_FOLDER}" \
  --model="${MODEL}" \
  --model-family="${MODEL}" \
  --task="${TASK}" \
  --graph-eval="graph_eval_with_${METRIC}" \
  --circuit="${IG_STYLE}_${INTEGRATED_STEPS}" \
  --seed="${SEED}" \
  --target-edges="${TARGET_EDGES}" \
  --template-label="${TEMPLATE_LABEL}"
