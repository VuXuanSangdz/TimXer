#!/usr/bin/env python3
"""Vẽ sơ đồ kiến trúc TimeXer-DINOv3 (đánh dấu khối (a) đã thay) -> reports/figures/fig_architecture.png"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "fig_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#ffffff", ec="#1a2240", lw=1.2, fs=9, bold=False, style="round,pad=0.02,rounding_size=0.08"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#1a2240", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")

    # ---- cột trái: (a) DINOv3 (thay thế) --------------------------------------------
    ax.add_patch(FancyBboxPatch((0.25, 0.35), 4.3, 6.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fff4e5", ec="#f59e0b", lw=2, ls="--"))
    ax.text(2.4, 6.6, "(a) Endogenous Embedding — THAY bằng DINOv3", ha="center", fontsize=9.5, weight="bold", color="#b45309")
    box(ax, 0.6, 5.55, 3.6, 0.6, "Waveform 5 s @16 kHz (đã khử nhiễu, chuẩn hoá)", fc="#f8fafc", fs=8.5)
    arrow(ax, 2.4, 5.55, 2.4, 5.15)
    box(ax, 0.6, 4.55, 3.6, 0.6, "log-Mel spectrogram  y ∈ R^(128×251)", fc="#e0f2fe", fs=9)
    arrow(ax, 2.4, 4.55, 2.4, 4.15)
    box(ax, 0.6, 3.55, 3.6, 0.6, "→ ảnh 3×224×224 (min-max, flip tần số, ImageNet norm)", fc="#f8fafc", fs=8.5)
    arrow(ax, 2.4, 3.55, 2.4, 3.15)
    box(ax, 0.6, 2.25, 3.6, 0.9, "DINOv3 ViT-S/16 (timm, đóng băng)\n21.6M tham số · 14×14 patch token + CLS", fc="#fde68a", ec="#b45309", lw=1.6, fs=9, bold=True)
    arrow(ax, 1.5, 2.25, 1.5, 1.75); arrow(ax, 3.4, 2.25, 3.4, 1.75)
    box(ax, 0.6, 1.1, 1.75, 0.65, "14 token thời gian\n(concat 14 dải tần)\n→ Linear → d_model", fc="#dbeafe", fs=7.8)
    box(ax, 2.5, 1.1, 1.7, 0.65, "CLS token\n= GLOBAL TOKEN\n(thay random-init)", fc="#fecaca", fs=7.8)
    ax.text(2.4, 0.6, "TimeXer gốc: patch 1-D → Linear(patch_len→d) + global token ngẫu nhiên", ha="center", fontsize=7.5, style="italic", color="#6b7280")

    # ---- cột giữa: encoder TimeXer giữ nguyên ---------------------------------------
    ax.add_patch(FancyBboxPatch((5.0, 0.35), 3.6, 6.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f1f5f9", ec="#64748b", lw=1.5))
    ax.text(6.8, 6.6, "Encoder TimeXer (giữ nguyên) × L=2", ha="center", fontsize=10, weight="bold", color="#334155")
    box(ax, 5.3, 5.55, 3.0, 0.55, "Output head: [global ‖ mean(patch)] → MLP → 5 lớp", fc="#f5d0fe", fs=8.5)
    arrow(ax, 6.8, 5.15, 6.8, 5.55)
    box(ax, 5.3, 4.6, 3.0, 0.5, "LayerNorm", fc="#fef08a", fs=9)
    box(ax, 5.3, 3.95, 3.0, 0.5, "Feed-Forward", fc="#bfdbfe", fs=9)
    box(ax, 5.3, 3.3, 3.0, 0.5, "LayerNorm", fc="#fef08a", fs=9)
    box(ax, 5.3, 2.65, 3.0, 0.5, "(d) Cross-Attention  Q=global, K/V=variate", fc="#fdba74", fs=8.5)
    box(ax, 5.3, 2.0, 3.0, 0.5, "LayerNorm", fc="#fef08a", fs=9)
    box(ax, 5.3, 1.35, 3.0, 0.5, "(c) Self-Attention (temporal-wise)", fc="#fdba74", fs=8.5)
    for y in (1.85, 2.5, 3.15, 3.8, 4.45):
        arrow(ax, 6.8, y, 6.8, y + 0.15)
    arrow(ax, 4.2, 1.42, 5.3, 1.42, color="#b45309", lw=2)  # tokens vào self-attn
    arrow(ax, 4.2, 1.6, 5.3, 1.6, color="#b45309", lw=2)
    ax.text(4.7, 1.0, "15 token\n(14 + global)", ha="center", fontsize=7.5, color="#b45309")

    # ---- cột phải: (b) exogenous giữ nguyên -----------------------------------------
    ax.add_patch(FancyBboxPatch((9.0, 0.35), 3.75, 6.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f0fdf4", ec="#16a34a", lw=1.5))
    ax.text(10.9, 6.6, "(b) Exogenous Embedding (giữ nguyên)", ha="center", fontsize=10, weight="bold", color="#166534")
    box(ax, 9.3, 4.75, 3.15, 1.35, "x ∈ R^(251×13)\nThời gian: hour/dow sin-cos\nNgữ cảnh ghi: duration, SNR, RMS,\nmic/upload, sample-rate\nHỗ trợ theo khung: F0, energy, ZCR, centroid", fc="#dcfce7", fs=8)
    arrow(ax, 10.9, 4.75, 10.9, 4.3)
    box(ax, 9.3, 3.7, 3.15, 0.6, "Inverted embedding\nLinear(T=251 → d_model) mỗi biến", fc="#bbf7d0", fs=8.5)
    arrow(ax, 10.9, 3.7, 10.9, 3.25)
    box(ax, 9.3, 2.65, 3.15, 0.6, "13 variate token  [B, 13, d_model]", fc="#86efac", fs=9)
    arrow(ax, 9.3, 2.9, 8.3, 2.9, color="#166534", lw=2)

    ax.text(6.5, 0.1, "TimeXer-DINOv3 cho dự đoán nhóm tuổi từ giọng nói — cam: khối thay thế; xanh lá/xám: giữ nguyên từ TimeXer (Wang et al., 2024)", ha="center", fontsize=8.5, color="#334155")
    plt.tight_layout(); plt.savefig(OUT, dpi=160, bbox_inches="tight"); plt.close()
    print("->", OUT)


if __name__ == "__main__":
    main()
