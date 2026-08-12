# scripts/run_eval.py
import argparse, csv
from pathlib import Path
from miarag.config import get_settings
from miarag.metrics import evaluate
from miarag.plots import plot_roc

def _load(path: Path):
    scores, labels, has_person = [], [], []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
            hp_val = int(row["has_person"])
            if hp_val not in {0, 1}:
                raise ValueError(f"has_person deve essere 0/1, trovato {hp_val}")
            has_person.append(hp_val)
    return scores, labels, has_person

def build_rows(name: str, scores: list[float], labels: list[int], has_person: list[int], prior: float) -> list[dict]:
    """Build summary rows for one attack: overall + disaggregated subgroups.

    Returns rows passing all guards (both classes present AND members >= 10 for subgroups).
    """
    rows = []

    # Overall evaluation
    rep = evaluate(scores, labels, prior=prior)
    row = {"attack": name, "subgroup": "all", **rep.to_row()}
    rows.append(row)
    print(row)

    # Disaggregated evaluation by has_person
    for has_p_val, subgroup_name in [(1, "has_person"), (0, "no_person")]:
        indices = [i for i, hp in enumerate(has_person) if hp == has_p_val]
        if not indices:
            print(f"{name}/{subgroup_name}: no data, skipping")
            continue
        sub_scores = [scores[i] for i in indices]
        sub_labels = [labels[i] for i in indices]
        # Check if subgroup has both classes
        if len(set(sub_labels)) < 2:
            print(f"{name}/{subgroup_name}: only one class present, skipping AUC")
            continue
        # Reliability threshold: skip if members < 10
        n_members = sum(sub_labels)
        if n_members < 10:
            print(f"{name}/{subgroup_name}: solo {n_members} membri (<10), AUC non affidabile, skip")
            continue
        sub_rep = evaluate(sub_scores, sub_labels, prior=prior)
        sub_row = {"attack": name, "subgroup": subgroup_name, **sub_rep.to_row()}
        rows.append(sub_row)
        print(sub_row)

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", type=float, default=0.1)
    args = ap.parse_args()
    s = get_settings()
    by_attack, rows = {}, []

    for name in ["s2mia", "budgetleak"]:
        path = s.results_dir / f"scores_{name}.csv"
        if not path.exists():
            continue
        scores, labels, has_person = _load(path)
        by_attack[name] = (scores, labels)

        # Build all rows for this attack (overall + disaggregated subgroups)
        attack_rows = build_rows(name, scores, labels, has_person, args.prior)
        rows.extend(attack_rows)

    with (s.results_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "subgroup", "auc", "tpr_at_1fpr", "ppv", "advantage"])
        w.writeheader(); w.writerows(rows)

    if by_attack:
        plot_roc(by_attack, s.results_dir / "roc.png")

if __name__ == "__main__":
    main()
