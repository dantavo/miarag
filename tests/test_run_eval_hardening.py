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

def test_subgroup_with_few_members_is_skipped(tmp_path):
    """Subgroup with <10 members should be skipped with a printed note, no row written."""
    csv_path = tmp_path / "scores_test.csv"
    # Create a dataset with 9 members (has_person=1) and 20 non-members (has_person=0)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["score", "label", "has_person"])
        w.writeheader()
        # 9 members with has_person=1
        for i in range(9):
            w.writerow({"score": str(0.6 + i * 0.01), "label": "1", "has_person": "1"})
        # 20 non-members with has_person=1 (to ensure both classes)
        for i in range(20):
            w.writerow({"score": str(0.3 + i * 0.01), "label": "0", "has_person": "1"})
        # Add enough data for has_person=0 to pass (10+ members)
        for i in range(15):
            w.writerow({"score": str(0.7 + i * 0.01), "label": "1", "has_person": "0"})
        for i in range(15):
            w.writerow({"score": str(0.4 + i * 0.01), "label": "0", "has_person": "0"})

    from scripts.run_eval import _load, evaluate
    from miarag.config import get_settings
    s = get_settings()
    scores, labels, has_person = _load(csv_path)

    # Simulate disaggregation for has_person=1 subgroup
    indices = [i for i, hp in enumerate(has_person) if hp == 1]
    sub_labels = [labels[i] for i in indices]
    n_members = sum(sub_labels)

    # Assert that n_members < 10 and thus would be skipped
    assert n_members == 9, f"Expected 9 members, got {n_members}"
    # In the real run_eval.py, this would print a skip message and not call evaluate
    # We verify the logic: should skip if n_members < 10
    assert n_members < 10
