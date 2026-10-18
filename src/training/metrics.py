"""Chỉ số đánh giá phân loại nhóm tuổi (có tính thứ tự)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.common.constants import AGE_GROUPS


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(AGE_GROUPS)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    dist = np.abs(y_true - y_pred)
    out = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(n)), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=list(range(n)), zero_division=0)),
        "ordinal_mae": float(dist.mean()) if len(dist) else 0.0,
        "within_one_group_acc": float((dist <= 1).mean()) if len(dist) else 0.0,
        "per_class_f1": dict(zip(AGE_GROUPS, f1_score(y_true, y_pred, average=None, labels=list(range(n)), zero_division=0).tolist())),
        "support": dict(zip(AGE_GROUPS, np.bincount(y_true, minlength=n).tolist())),
        "confusion_matrix": cm.tolist(),
    }
    if probs is not None and len(probs):
        eps = 1e-8
        out["nll"] = float(-np.log(probs[np.arange(len(y_true)), y_true] + eps).mean())
        conf = probs.max(axis=1)
        out["mean_confidence"] = float(conf.mean())
        # Expected Calibration Error (10 bins)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs((y_pred[m] == y_true[m]).mean() - conf[m].mean())
        out["ece"] = float(ece)
    return out


def format_metrics(m: dict, title: str = "") -> str:
    lines = [f"[{title}]" if title else ""]
    lines.append(
        f"  acc={m['accuracy']*100:.1f}%  bal-acc={m['balanced_accuracy']*100:.1f}%  macro-F1={m['macro_f1']:.3f}  "
        f"ordinal-MAE={m['ordinal_mae']:.3f}  within±1={m['within_one_group_acc']*100:.1f}%"
        + (f"  ECE={m['ece']:.3f}" if "ece" in m else "")
    )
    lines.append("  per-class F1: " + "  ".join(f"{k}={v:.2f}" for k, v in m["per_class_f1"].items()))
    lines.append("  support     : " + "  ".join(f"{k}={v}" for k, v in m["support"].items()))
    return "\n".join(lines)
