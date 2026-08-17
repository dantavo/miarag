# scripts/eval_defenses.py
"""Evaluate the security/utility trade-off: baseline vs each text-based defense.

run_eval.py only evaluates baseline scores (scores_{attack}.csv). This script
also reads the defense-suffixed CSVs (scores_{attack}_{defense}.csv) and builds
a comparison table + results/summary_defenses.csv.

Guards against NaN/inf scores (perplexity failures under paraphrase) by dropping
non-finite rows before scoring.

Usage:
    PYTHONPATH=src uv run python scripts/eval_defenses.py --prior 0.1
"""
import argparse
import csv
import math
from pathlib import Path

from miarag.config import get_settings
from miarag.metrics import evaluate

ATTACKS = ["s2mia", "budgetleak", "rag_mia"]
SCENARIOS = [
    ("baseline", ""),
    ("paraphrase", "_paraphrase"),
    ("prompt_hardening", "_prompt_hardening"),
]


def _load_finite(path: Path):
    scores, labels = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                sc = float(row["score"])
            except (ValueError, KeyError):
                continue
            if not math.isfinite(sc):  # drop NaN/inf (e.g. perplexity failure)
                continue
            scores.append(sc)
            labels.append(int(row["label"]))
    return scores, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", type=float, default=0.1)
    args = ap.parse_args()
    s = get_settings()

    rows = []
    print(f"{'attack':<11}{'scenario':<18}{'AUC':>7}{'TPR@1%FPR':>11}{'PPV':>7}{'adv':>7}{'  n_valid':>9}")
    for attack in ATTACKS:
        for scen_name, suffix in SCENARIOS:
            path = s.results_dir / f"scores_{attack}{suffix}.csv"
            if not path.exists():
                print(f"{attack:<11}{scen_name:<18}{'--':>7}   (file mancante: {path.name})")
                continue
            scores, labels = _load_finite(path)
            if len(set(labels)) < 2:
                print(f"{attack:<11}{scen_name:<18}{'N/A':>7}   (una sola classe)")
                continue
            rep = evaluate(scores, labels, prior=args.prior)
            rows.append({
                "attack": attack, "scenario": scen_name,
                "auc": rep.auc, "tpr_at_1fpr": rep.tpr_at_1fpr,
                "ppv": rep.ppv, "advantage": rep.advantage,
                "n_valid": len(labels),
            })
            print(f"{attack:<11}{scen_name:<18}{rep.auc:>7.3f}{rep.tpr_at_1fpr:>11.3f}"
                  f"{rep.ppv:>7.3f}{rep.advantage:>7.3f}{len(labels):>9}")

    out = s.results_dir / "summary_defenses.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "scenario", "auc", "tpr_at_1fpr", "ppv", "advantage", "n_valid"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
