#!/usr/bin/env python3
"""Huấn luyện + đánh giá TimeXer-DINOv3 (hoặc baseline) — ghi checkpoints/<run_name>/{best_model.pt, train_meta.json}."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.constants import AGE_GROUPS  # noqa: E402
from src.data.dataset import VoiceAgeArrays, make_loaders  # noqa: E402
from src.models.timexer_dinov3 import TimeXerDINOv3, TimeXerVoiceBaseline  # noqa: E402
from src.training.metrics import format_metrics  # noqa: E402
from src.training.trainer import Trainer, get_device  # noqa: E402


def build_model(cfg: dict, seq_len: int, n_exog: int, n_prosody: int = 0, pretrained: bool = True) -> torch.nn.Module:
    m = cfg["model"]
    n_pro = n_prosody if m.get("use_prosody", False) else 0
    common = dict(seq_len=seq_len, n_exog=n_exog, n_classes=len(AGE_GROUPS), d_model=m["d_model"], n_heads=m["n_heads"], e_layers=m["e_layers"], d_ff=m["d_ff"], dropout=m["dropout"], n_prosody=n_pro)
    if m["type"] == "timexer_dinov3":
        return TimeXerDINOv3(**common, dino_model=m["dino_model"], dino_pretrained=pretrained, dino_img_size=m["dino_img_size"], dino_freeze=m["dino_freeze"], dino_token_mode=m["dino_token_mode"])
    if m["type"] == "timexer_baseline":
        return TimeXerVoiceBaseline(**common, patch_len=m["patch_len"])
    raise ValueError(m["type"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "voice_age_dinov3.yaml"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed = cfg["train"].get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    d = cfg["data"]
    arrays = VoiceAgeArrays.load(ROOT / d["processed_dir"], ROOT / d["dino_cache"] if d.get("dino_cache") else None)
    use_cache = bool(d.get("use_cache")) and arrays.dino_tokens is not None
    loaders = make_loaders(arrays, cfg["train"]["batch_size"], use_cache, d.get("balanced_sampling", True), token_noise=d.get("token_noise", 0.0))
    print("=" * 70)
    print(f"Run: {cfg['run_name']} | model: {cfg['model']['type']} | cache tokens: {use_cache}")
    print(f"Samples train/val/test: {[len(arrays.indices(s)) for s in ('train','val','test')]} | seq_len={arrays.seq_len} | n_exog={arrays.n_exog}")
    print("Label dist (train):", arrays.meta.loc[arrays.indices('train'), 'age_group'].value_counts().reindex(AGE_GROUPS).to_dict())
    print("=" * 70)

    model = build_model(cfg, arrays.seq_len, arrays.n_exog, arrays.n_prosody, pretrained=not use_cache)
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    print(f"Tham số học được: {n_tr/1e6:.2f}M | tổng: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | prosody token: {getattr(model, 'prosody_embedding', None) is not None}")

    device = get_device(cfg["train"].get("device", "auto"))
    ckpt_dir = ROOT / "checkpoints" / cfg["run_name"]
    trainer = Trainer(model, cfg, device, ckpt_dir, class_weights=arrays.class_weights("train", cfg["train"].get("class_weight_power", 0.5)))
    trainer.extra = {"seq_len": arrays.seq_len, "n_exog": arrays.n_exog}
    if getattr(model, "prosody_embedding", None) is not None:
        from src.common.constants import PROSODY_COLS

        trainer.extra.update({"prosody_cols": PROSODY_COLS, "prosody_mean": arrays.prosody_mean.tolist(), "prosody_std": arrays.prosody_std.tolist()})
    fit = trainer.fit(loaders["train"], loaders["val"], cfg["train"]["epochs"])
    best_epoch = trainer.load_best()

    val_m = trainer.evaluate(loaders["val"], collect=True)
    test_m = trainer.evaluate(loaders["test"], collect=True)
    print(format_metrics(val_m, "VAL (best ckpt)"))
    print(format_metrics(test_m, "TEST (best ckpt)"))
    print("Confusion (test, hàng=thật, cột=dự đoán):")
    for g, row in zip(AGE_GROUPS, test_m["confusion_matrix"]):
        print(f"  {g:>6}: {row}")

    # lưu dự đoán test để phân tích
    import pandas as pd

    pred_df = arrays.meta.loc[test_m["_index"], ["clip_id", "speaker_id", "age_group", "source", "locale"]].copy()
    pred_df["pred_group"] = [AGE_GROUPS[i] for i in test_m["_y_pred"]]
    for i, g in enumerate(AGE_GROUPS):
        pred_df[f"p_{g}"] = test_m["_probs"][:, i]
    pred_df.to_csv(ckpt_dir / "test_predictions.csv", index=False, encoding="utf-8")

    strip = lambda m: {k: v for k, v in m.items() if not k.startswith("_")}  # noqa: E731
    meta = {"run_name": cfg["run_name"], "config": cfg, "fit": fit, "best_epoch": best_epoch, "trainable_params": n_tr, "val_metrics": strip(val_m), "test_metrics": strip(test_m), "n_samples": {s: int(len(arrays.indices(s))) for s in ("train", "val", "test")}}
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved: {ckpt_dir / 'best_model.pt'}\nSaved: {ckpt_dir / 'train_meta.json'}")


if __name__ == "__main__":
    main()
