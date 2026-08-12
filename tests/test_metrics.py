# tests/test_metrics.py
from miarag.metrics import roc_auc, tpr_at_fpr, ppv_with_prior, evaluate, tpr_fpr_at_fpr

def test_auc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert roc_auc(scores, labels) == 1.0

def test_auc_random_is_half():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert abs(roc_auc(scores, labels) - 0.5) < 1e-6

def test_ppv_prior_lowers_confidence():
    # con prior basso il PPV crolla anche con TPR alto
    assert ppv_with_prior(tpr=0.9, fpr=0.1, prior=0.1) < 0.6

def test_evaluate_returns_report():
    scores = [0.1, 0.2, 0.8, 0.9]; labels = [0, 0, 1, 1]
    rep = evaluate(scores, labels, prior=0.5, target_fpr=0.5)
    assert rep.auc == 1.0
    assert 0.0 <= rep.ppv <= 1.0
    assert "auc" in rep.to_row()

def test_tpr_fpr_at_fpr_returns_achieved_fpr():
    # Perfect separation: at target_fpr=0.01, the best operating point has FPR=0.0
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    tpr, achieved_fpr = tpr_fpr_at_fpr(scores, labels, target_fpr=0.01)
    assert achieved_fpr <= 0.01  # must be <= target
    assert tpr > 0.0
    # For perfect separation at low FPR, achieved_fpr should be 0.0
    assert achieved_fpr == 0.0

def test_ppv_with_achieved_fpr_zero_is_one():
    # When achieved FPR = 0.0, PPV should be 1.0 (no false positives)
    # regardless of prior (as long as TPR > 0)
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    tpr, achieved_fpr = tpr_fpr_at_fpr(scores, labels, target_fpr=0.01)
    assert achieved_fpr == 0.0
    ppv = ppv_with_prior(tpr, achieved_fpr, prior=0.1)
    assert ppv == 1.0  # no false positives → perfect precision

def test_tpr_at_fpr_unchanged_by_refactor():
    # Ensure tpr_at_fpr still returns the same float value (DRY refactor check)
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    tpr_old_api = tpr_at_fpr(scores, labels, target_fpr=0.01)
    tpr_new_api, _ = tpr_fpr_at_fpr(scores, labels, target_fpr=0.01)
    assert tpr_old_api == tpr_new_api
