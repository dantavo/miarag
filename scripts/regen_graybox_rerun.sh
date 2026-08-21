#!/usr/bin/env bash
# scripts/regen_graybox_rerun.sh — SOLO azure-graybox (fallito per inf in XGBoost).
# Dopo il fix (_xgb_proba sanitizza inf/nan + exp sicuro nella ppl nativa).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=1
export MIARAG_S2MIA_WORKERS=4     # azure rete: 4 concorrenti (retry/backoff gestisce 429)

echo "=== graybox rerun start: $(date '+%F %T') (workers=$MIARAG_S2MIA_WORKERS) ==="
uv run python scripts/run_attack.py --split doc --graybox --llm azure_openai \
  --results-dir results_azure --attacks s2mia
echo ">>> attack exit=$?"
if [ -f results_azure/scores_s2mia_graybox.csv ]; then
  cp results_azure/scores_s2mia_graybox.csv results/scores_s2mia_graybox.csv
  echo ">>> copiato results_azure/scores_s2mia_graybox.csv → results/scores_s2mia_graybox.csv"
fi
echo "=== graybox rerun done: $(date '+%F %T') ==="
