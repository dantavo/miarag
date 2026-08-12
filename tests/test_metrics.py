# tests/test_metrics.py
from miarag.metrics import roc_auc, tpr_at_fpr, ppv_with_prior, evaluate

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
