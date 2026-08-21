#!/usr/bin/env bash
# scripts/regen_s2mia_azure.sh
# Rigenera gli score S2MIA per il target AZURE (GPT-4o-mini) col fix di scoring.
# Gira IN PARALLELO al job ollama: usa la rete (API Azure), non la GPU locale →
# nessuna contesa col target locale.
#
# PARALLELISMO: MIARAG_S2MIA_WORKERS=2 → concorrenza moderata amichevole coi
# rate-limit Azure (il provider ha già retry/backoff su 429). Azure non è il
# collo di bottiglia (finisce ampiamente dentro la finestra del job ollama).
#
# Config → results_azure/ :  none (black-box) + graybox (logprob)
# Poi copia i CSV nei nomi attesi da Cap06 in results/.
#
# Usage:  nohup caffeinate -dimsu ./scripts/regen_s2mia_azure.sh > logs/regen_s2mia_azure.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1              # black-box usa gpt2 (torch) + xgboost → stesso fix segfault
export MIARAG_S2MIA_WORKERS=2        # rete: concorrenza moderata (rate-limit friendly)

echo "=== regen S2MIA AZURE start: $(date '+%F %T') (workers=$MIARAG_S2MIA_WORKERS) ==="

run() {
  local desc="$1"; shift
  echo ">>> [$desc] start $(date '+%T'): run_attack.py $*"
  uv run python scripts/run_attack.py "$@" --attacks s2mia
  echo ">>> [$desc] done  $(date '+%T') (exit=$?)"
}

run "azure-none"    --split doc --defense none    --llm azure_openai --results-dir results_azure
run "azure-graybox" --split doc --graybox         --llm azure_openai --results-dir results_azure

# --- Allinea i nomi Azure a quelli attesi da Cap06 in results/ ---
if [ -f results_azure/scores_s2mia.csv ]; then
  cp results_azure/scores_s2mia.csv results/scores_s2mia_azure44.csv
  echo ">>> copiato results_azure/scores_s2mia.csv → results/scores_s2mia_azure44.csv"
fi
if [ -f results_azure/scores_s2mia_graybox.csv ]; then
  cp results_azure/scores_s2mia_graybox.csv results/scores_s2mia_graybox.csv
  echo ">>> copiato results_azure/scores_s2mia_graybox.csv → results/scores_s2mia_graybox.csv"
fi

echo "=== regen S2MIA AZURE done: $(date '+%F %T') ==="
