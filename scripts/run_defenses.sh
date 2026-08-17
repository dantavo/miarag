#!/usr/bin/env bash
# scripts/run_defenses.sh
# Unattended defense runner (document-level split) for the security/utility trade-off.
#
# Runs the full 3-attack pipeline under each text-based defense, then a defense
# evaluation. Designed to be launched when the Mac is COOL (Ollama generation is
# ~2x slower under thermal throttling), ideally overnight.
#
# Usage:
#   ./scripts/run_defenses.sh              # runs paraphrase + prompt_hardening
#   nohup ./scripts/run_defenses.sh > logs/defenses.log 2>&1 &   # background
#
# Prereqs: Ollama running (ollama serve), data/processed/reports.jsonl ingested.
# Outputs: results/scores_{attack}_{defense}.csv  (git-ignored)
#
# NOTE: baseline (no defense) doc-split scores must already exist in results/
# (scores_s2mia.csv, scores_budgetleak.csv, scores_rag_mia.csv) for the
# comparison. Regenerate them with:  run_attack.py --split doc
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

SPLIT="${SPLIT:-doc}"   # override with SPLIT=chunk if needed

echo "=== defense run start: $(date '+%F %T') (split=$SPLIT) ==="

for DEF in paraphrase prompt_hardening; do
  echo ">>> defense=$DEF starting $(date '+%T')"
  uv run python scripts/run_attack.py --split "$SPLIT" --defense "$DEF"
  echo ">>> defense=$DEF done $(date '+%T')"
done

echo "=== defense run done: $(date '+%F %T') ==="
echo "CSV prodotti:"
ls -1 results/scores_*_paraphrase.csv results/scores_*_prompt_hardening.csv 2>/dev/null || true
echo ""
echo "Per valutare il trade-off:  uv run python scripts/eval_defenses.py"
