"""Metrics utilities for policy/value evaluation.

Provides small, reusable functions to compute common metrics across
training and validation without duplicating logic in multiple modules.

Design principles:
- DRY: single implementation used by train/validate.
- KISS: small, focused functions with simple inputs/outputs.
- SOLID (SRP): keep metric calculations separate from model/data code.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def compute_policy_value_metrics(
    collected_pi,  # List[torch.Tensor] target policy distributions
    collected_model,  # List[torch.Tensor] model probability distributions
    collected_v_pred,  # List[torch.Tensor] predicted value probabilities (0-1)
    collected_v_t,  # List[torch.Tensor] target value labels (0-1)
) -> Dict[str, Optional[float]]:
    """Compute common policy/value metrics.

    Returns a dict with keys:
      - policy_kl
      - policy_top1_match
      - value_acc
      - value_brier
    Values are floats or None when not computable.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - metrics require torch tensors
        return {
            "policy_kl": None,
            "policy_top1_match": None,
            "value_acc": None,
            "value_brier": None,
        }

    if not collected_pi:
        return {
            "policy_kl": None,
            "policy_top1_match": None,
            "value_acc": None,
            "value_brier": None,
        }

    kl_list: List[float] = []
    top1_list: List[float] = []
    v_hit_list: List[float] = []
    brier_list: List[float] = []

    for pi_t, model_p, v_pred_c, v_lab in zip(
        collected_pi, collected_model, collected_v_pred, collected_v_t
    ):
        try:
            kl = (pi_t * (pi_t.add(1e-12).log() - model_p.add(1e-12).log())).sum().item()
            kl_list.append(float(kl))
        except Exception:
            pass
        try:
            if pi_t.numel() > 0 and model_p.numel() > 0:
                top1_list.append(1.0 if int(pi_t.argmax()) == int(model_p.argmax()) else 0.0)
        except Exception:
            pass
        try:
            v_pred_f = float(v_pred_c)
            v_lab_f = float(v_lab)
            v_hit_list.append(1.0 if (v_pred_f > 0.5) == (v_lab_f > 0.5) else 0.0)
            diff = v_pred_f - v_lab_f
            brier_list.append(diff * diff)
        except Exception:
            pass

    def _mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "policy_kl": _mean(kl_list),
        "policy_top1_match": _mean(top1_list),
        "value_acc": _mean(v_hit_list),
        "value_brier": _mean(brier_list),
    }
