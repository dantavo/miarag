#!/usr/bin/env bash
# scripts/regen_s2mia.sh
# Rigenera SOLO gli score S2MIA dopo il fix del metodo di scoring (S²MIA-M/XGBoost,
# feature grezze bleu+ppl salvate). Non tocca budgetleak/rag_mia.
#
# Config rigenerate:
#   Ollama (default provider):  none, paraphrase, prompt_hardening  → results/
#   Azure OpenAI:               none (black-box) + graybox (logprob) → results_azure/
#   poi copia i CSV Azure nei nomi attesi da Cap06 in results/
#
# Usage:  nohup ./scripts/regen_s2mia.sh > logs/regen_s2mia.log 2>&1 &
set -uo pipefail   # NON -e: un run fallito non deve abortire gli altri
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false   # evita semaphore-leak da fork tokenizers HF
export HF_HUB_DISABLE_XET=1            # download HF stabile (come run_ingestion)
export OMP_NUM_THREADS=1               # CRITICO: torch+xgboost = 2 runtime OpenMP → segfault macOS ARM

echo "=== regen S2MIA start: $(date '+%F %T') ==="

run() {  # $1=descrizione, resto=args
  local desc="$1"; shift
  echo ">>> [$desc] start $(date '+%T'): run_attack.py $*"
  uv run python scripts/run_attack.py "$@" --attacks s2mia
  echo ">>> [$desc] done  $(date '+%T') (exit=$?)"
}

# --- Ollama (default) → results/ ---
run "ollama-none"       --split doc --defense none
run "ollama-paraphrase" --split doc --defense paraphrase
run "ollama-harden"     --split doc --defense prompt_hardening

# --- Azure black-box + graybox → results_azure/ ---
run "azure-none"    --split doc --defense none    --llm azure_openai --results-dir results_azure
run "azure-graybox" --split doc --graybox         --llm azure_openai --results-dir results_azure

# --- Allinea i nomi Azure a quelli attesi da Cap06 in results/ ---
# La Tabella 6.1 usa la colonna Azure da scores_s2mia_azure44.csv;
# graybox continua della §6.2.6 da scores_s2mia_graybox.csv.
if [ -f results_azure/scores_s2mia.csv ]; then
  cp results_azure/scores_s2mia.csv results/scores_s2mia_azure44.csv
  echo ">>> copiato results_azure/scores_s2mia.csv → results/scores_s2mia_azure44.csv"
fi
if [ -f results_azure/scores_s2mia_graybox.csv ]; then
  cp results_azure/scores_s2mia_graybox.csv results/scores_s2mia_graybox.csv
  echo ">>> copiato results_azure/scores_s2mia_graybox.csv → results/scores_s2mia_graybox.csv"
fi

echo "=== regen S2MIA done: $(date '+%F %T') ==="
echo "CSV S2MIA prodotti:"
ls -1 results/scores_s2mia*.csv
echo ""
echo "NB: rilancia poi  uv run python scripts/run_eval.py --prior 0.1  per il summary."
