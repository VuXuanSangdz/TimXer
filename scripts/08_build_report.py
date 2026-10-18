#!/usr/bin/env python3
"""
Tổng hợp kết quả các run trong checkpoints/ -> reports/
    reports/model_comparison.md / .json   (bảng so sánh: DINOv3 vs baseline vs sklearn)
    reports/figures/fig_dataset.png        (phân bố nhóm tuổi theo nguồn / split)
    reports/figures/fig_curves.png         (learning curves)
    reports/figures/fig_confusion.png      (confusion matrix test, chuẩn hoá theo hàng)
    reports/figures/fig_compare.png        (bar chart macro-F1 / acc)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.constants import AGE_GROUPS  # noqa: E402
from src.training.metrics import classification_metrics  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def sklearn_baselines(processed: Path) -> dict[str, dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta = pd.read_csv(processed / "metadata.csv")
    sc = pd.read_csv(processed / "features_scalar.csv")
    df = meta.merge(sc, on="clip_id")
    cols = [c for c in sc.columns if c != "clip_id" and not c.startswith(("hour_", "dow_"))]
    X, y = df[cols].fillna(0).to_numpy(), df["label"].to_numpy()
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    out = {}
    for name, clf in [
        ("LogReg (MFCC/F0/formant stats)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))),
        ("HistGradientBoosting (scalar feats)", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)),
    ]:
        clf.fit(X[tr], y[tr])
        out[name] = classification_metrics(y[te], clf.predict(X[te]), clf.predict_proba(X[te]))
    return out


def main():
    runs = {}
    for d in sorted((ROOT / "checkpoints").glob("*/train_meta.json")):
        meta = json.loads(d.read_text(encoding="utf-8"))
        runs[meta["run_name"]] = meta
    if not runs:
        print("Chưa có run nào trong checkpoints/")
        return
    processed = ROOT / "data" / "processed"
    manifest = json.loads((processed / "dataset_manifest.json").read_text(encoding="utf-8"))
    meta_df = pd.read_csv(processed / "metadata.csv")

    rows = []
    for name, m in sklearn_baselines(processed).items():
        rows.append({"model": name, "type": "classical", **{k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae", "within_one_group_acc")}, "params": "-"})
    for rn, meta in runs.items():
        tm = meta["test_metrics"]
        label = {
            "timexer_baseline_linear_patch_noprosody": "TimeXer gốc (linear patch embedding)",
            "timexer_baseline_linear_patch": "TimeXer gốc + prosody token",
            "timexer_dinov3_vits16_noprosody": "TimeXer + DINOv3 ViT-S/16, temporal_concat (không prosody)",
            "timexer_dinov3_vits16": "TimeXer + DINOv3 ViT-S/16, temporal_concat + prosody token (đề xuất)",
            "timexer_dinov3_vits16_temporal_mean": "TimeXer + DINOv3, token temporal mean (ablation, không prosody)",
            "timexer_dinov3_vits16_grid": "TimeXer + DINOv3, token grid 196 (ablation, không prosody)",
        }.get(rn, rn)
        rows.append({"model": label, "type": meta["config"]["model"]["type"], **{k: tm[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae", "within_one_group_acc")}, "params": f"{meta['trainable_params']/1e6:.2f}M", "best_epoch": meta["best_epoch"]})
    cmp = pd.DataFrame(rows)
    cmp.to_json(REPORTS / "model_comparison.json", orient="records", indent=1, force_ascii=False)
    md = ["| Mô hình | Acc | Bal-Acc | Macro-F1 | Ordinal MAE | Within ±1 | Params học |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['accuracy']*100:.1f}% | {r['balanced_accuracy']*100:.1f}% | **{r['macro_f1']:.3f}** | {r['ordinal_mae']:.3f} | {r['within_one_group_acc']*100:.1f}% | {r['params']} |")
    (REPORTS / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- fig dataset
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    pd.crosstab(meta_df["age_group"], meta_df["source"]).reindex(AGE_GROUPS).plot.bar(stacked=True, ax=ax[0], color=["#3b6cf5", "#f59e0b", "#22c55e"][: meta_df.source.nunique()])
    ax[0].set_title(f"Phân bố nhóm tuổi theo nguồn (N={manifest['n_clips']}, {manifest['n_speakers']} speaker)"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)
    pd.crosstab(meta_df["age_group"], meta_df["split"]).reindex(AGE_GROUPS)[["train", "val", "test"]].plot.bar(ax=ax[1], color=["#1e3a8a", "#60a5fa", "#93c5fd"])
    ax[1].set_title("Split speaker-disjoint"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(FIG / "fig_dataset.png", dpi=150); plt.close()

    # ---- fig curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for rn in runs:
        h = pd.read_json(ROOT / "checkpoints" / rn / "history.json")
        ax[0].plot(h.epoch, h.train_loss, "--", label=f"{rn} train"); ax[0].plot(h.epoch, h.val_loss, label=f"{rn} val")
        ax[1].plot(h.epoch, h.val_macro_f1, label=rn)
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7)
    ax[1].set_title("Val macro-F1"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig_curves.png", dpi=150); plt.close()

    # ---- fig confusion
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.4), squeeze=False)
    for ax, (rn, meta) in zip(axes[0], runs.items()):
        cm = np.array(meta["test_metrics"]["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(AGE_GROUPS)):
            for j in range(len(AGE_GROUPS)):
                ax.text(j, i, f"{cmn[i,j]*100:.0f}%\n({int(cm[i,j])})", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(5), AGE_GROUPS); ax.set_yticks(range(5), AGE_GROUPS)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thật"); ax.set_title(f"{rn}\nacc={meta['test_metrics']['accuracy']*100:.1f}%  F1={meta['test_metrics']['macro_f1']:.3f}", fontsize=9)
    plt.tight_layout(); plt.savefig(FIG / "fig_confusion.png", dpi=150); plt.close()

    # ---- fig compare
    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(cmp)), 4.2))
    x = np.arange(len(cmp)); w = 0.38
    ax.bar(x - w / 2, cmp.macro_f1, w, label="Macro-F1", color="#3b6cf5")
    ax.bar(x + w / 2, cmp.accuracy, w, label="Accuracy", color="#93c5fd")
    ax.set_xticks(x, [m.replace(", ", "\n").replace(" (", "\n(") for m in cmp.model], fontsize=7.5); ax.legend(); ax.set_ylim(0, 0.7); ax.set_title("So sánh trên tập test (speaker-disjoint)")
    for i, (f, a) in enumerate(zip(cmp.macro_f1, cmp.accuracy)):
        ax.text(i - w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8); ax.text(i + w / 2, a + 0.01, f"{a*100:.0f}%", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig_compare.png", dpi=150); plt.close()
    print(f"\nHình -> {FIG}")


if __name__ == "__main__":
    main()
