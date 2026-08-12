# tests/test_run_eval_hardening.py
import pytest, csv
from pathlib import Path

def test_has_person_validation_rejects_invalid_value(tmp_path):
    from scripts.run_eval import _load
    csv_path = tmp_path / "scores.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["score", "label", "has_person"])
        w.writeheader()
        w.writerow({"score": "0.5", "label": "1", "has_person": "2"})  # invalid
    with pytest.raises(ValueError, match=r"has_person deve essere 0/1, trovato 2"):
        _load(csv_path)

def test_subgroup_with_few_members_is_skipped():
    """Subgroup with <10 members should be skipped: no row written for that subgroup."""
    from scripts.run_eval import build_rows

    # Dataset: has_person=1 has exactly 9 members (both classes present, triggers <10 guard)
    # has_person=0 has 15 members (passes all guards)
    scores = (
        [0.6 + i * 0.01 for i in range(9)]      # 9 members, has_person=1
        + [0.3 + i * 0.01 for i in range(20)]   # 20 non-members, has_person=1
        + [0.7 + i * 0.01 for i in range(15)]   # 15 members, has_person=0
        + [0.4 + i * 0.01 for i in range(15)]   # 15 non-members, has_person=0
    )
    labels = [1]*9 + [0]*20 + [1]*15 + [0]*15
    has_person = [1]*29 + [0]*30

    rows = build_rows("test_attack", scores, labels, has_person, prior=0.1)

    # Extract subgroup names from returned rows
    subgroups = [r["subgroup"] for r in rows]

    # Assert on actual output behavior (not preconditions)
    assert "all" in subgroups, "Should have overall 'all' row"
    assert "no_person" in subgroups, "no_person has 15 members, should be included"
    assert "has_person" not in subgroups, "has_person has 9 members (<10), should be skipped"
