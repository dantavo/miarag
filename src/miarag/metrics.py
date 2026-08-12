# src/miarag/metrics.py
"""Evaluation metrics for Membership Inference Attacks.

Convention: scores are membership scores where HIGHER = more likely member.
If an attack's raw signal is inverted (e.g., perplexity: higher = non-member),
the caller must orient the scores before passing them here.
"""
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def roc_auc(scores, labels) -> float:
    """Compute AUC-ROC for membership scores.

    Args:
        scores: Membership scores (higher = more likely member)
        labels: Ground truth (1 = member, 0 = non-member)

    Returns:
        AUC value in [0, 1]
    """
    return float(roc_auc_score(labels, scores))

def tpr_at_fpr(scores, labels, target_fpr: float = 0.01) -> float:
    """Compute TPR at a given FPR threshold (interpolated).

    This is a key MIA metric: TPR at low FPR (e.g., 1%) measures the attacker's
    true positive rate when constrained to a low false positive rate.

    Implementation: find the maximum TPR among all ROC points where FPR <= target_fpr.
    If no such point exists (all FPR > target), return 0.0.

    Args:
        scores: Membership scores (higher = more likely member)
        labels: Ground truth (1 = member, 0 = non-member)
        target_fpr: Target false positive rate (default: 0.01)

    Returns:
        TPR value at the given FPR threshold
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0

def ppv_with_prior(tpr: float, fpr: float, prior: float) -> float:
    """Compute Positive Predictive Value (precision) incorporating membership prior.

    Formula: PPV = (π·TPR) / (π·TPR + (1−π)·FPR)
    where π is the membership prior (base rate).

    The naive precision from a balanced test set overstates real-world attacker success.
    This formula adjusts for the true membership rate in the wild.

    Args:
        tpr: True positive rate at a given threshold
        fpr: False positive rate at the same threshold
        prior: Membership prior π (e.g., 0.1 = 10% of candidates are members)

    Returns:
        PPV value, or 0.0 if denominator is zero
    """
    denom = prior * tpr + (1 - prior) * fpr
    return float(prior * tpr / denom) if denom > 0 else 0.0

def membership_advantage(tpr: float, fpr: float) -> float:
    """Compute membership advantage: TPR − FPR.

    This measures the gain over random guessing at a given operating point.

    Args:
        tpr: True positive rate
        fpr: False positive rate

    Returns:
        Advantage value (tpr - fpr)
    """
    return float(tpr - fpr)

@dataclass
class AttackReport:
    """Consolidated report of an attack's performance."""
    auc: float
    tpr_at_1fpr: float
    ppv: float
    advantage: float

    def to_row(self) -> dict:
        """Convert to a flat dict for DataFrame export."""
        return asdict(self)

def evaluate(scores, labels, prior: float = 0.1, target_fpr: float = 0.01) -> AttackReport:
    """Evaluate attack performance across all metrics.

    Args:
        scores: Membership scores (higher = more likely member)
        labels: Ground truth (1 = member, 0 = non-member)
        prior: Membership prior for PPV calculation (default: 0.1)
        target_fpr: Target FPR for TPR@FPR metric (default: 0.01)

    Returns:
        AttackReport with AUC, TPR@1%FPR, PPV, and advantage
    """
    auc = roc_auc(scores, labels)
    tpr = tpr_at_fpr(scores, labels, target_fpr)

    # Compute advantage at the Youden's J statistic point (max TPR - FPR)
    fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
    j = int(np.argmax(tpr_curve - fpr_curve))
    adv = membership_advantage(float(tpr_curve[j]), float(fpr_curve[j]))

    ppv = ppv_with_prior(tpr, target_fpr, prior)

    return AttackReport(auc=auc, tpr_at_1fpr=tpr, ppv=ppv, advantage=adv)
