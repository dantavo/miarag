#!/usr/bin/env bash
# scripts/regen_s2mia_ollama.sh
# Rigenera gli score S2MIA per il target OLLAMA (llama3.1:8b) con il fix di
# scoring (S²MIA-M/XGBoost, feature grezze bleu+ppl salvate).
#
# PARALLELISMO: MIARAG_S2MIA_WORKERS=4 → 4 query concorrenti a Ollama, che le
# batcha (continuous-batching). RICHIEDE un server Ollama con OLLAMA_NUM_PARALLEL>=4
# (avviato con `OLLAMA_NUM_PARALLEL=4 ollama serve`). Benchmark: ~2.7x.
#
# Config → results/ :  none, paraphrase, prompt_hardening
#
# Usage:  nohup caffeinate -dimsu ./scripts/regen_s2mia_ollama.sh > logs/regen_s2mia_ollama.log 2>&1 &
set -uo pipefail   # NON -e: un run fallito non abortisce gli altri
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false   # evita semaphore-leak da fork tokenizers HF
export HF_HUB_DISABLE_XET=1           # download HF stabile
export OMP_NUM_THREADS=1              # CRITICO: torch+xgboost = 2 runtime OpenMP → segfault macOS ARM
export MIARAG_S2MIA_WORKERS=4         # continuous-batching su Ollama -np 4 (~2.7x)

echo "=== regen S2MIA OLLAMA start: $(date '+%F %T') (workers=$MIARAG_S2MIA_WORKERS) ==="

# sanity: verifica che il server Ollama sia -np>=4 (altrimenti serializza)
NP=$(pgrep -fl llama-server | grep -oE '\-np [0-9]+' | grep -oE '[0-9]+' | head -1 || echo "?")
echo ">>> Ollama runner -np=$NP (atteso >=4 per il batching)"

run() {  # $1=descrizione, resto=args
  local desc="$1"; shift
  echo ">>> [$desc] start $(date '+%T'): run_attack.py $*"
  uv run python scripts/run_attack.py "$@" --attacks s2mia
  echo ">>> [$desc] done  $(date '+%T') (exit=$?)"
}

run "ollama-none"       --split doc --defense none
run "ollama-paraphrase" --split doc --defense paraphrase
run "ollama-harden"     --split doc --defense prompt_hardening

echo "=== regen S2MIA OLLAMA done: $(date '+%F %T') ==="
echo "CSV S2MIA ollama prodotti:"
ls -1 results/scores_s2mia.csv results/scores_s2mia_paraphrase.csv results/scores_s2mia_prompt_hardening.csv 2>/dev/null
