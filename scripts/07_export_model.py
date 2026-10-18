#!/usr/bin/env python3
"""
Xuất mô hình đã huấn luyện cho hệ thống triển khai (app/model/):
    timexer_dinov3.onnx       — toàn bộ pipeline mel+exog -> logits (DINOv3 + TimeXer), opset 17
    timexer_dinov3_head.pt    — chỉ trọng số phần TimeXer (fallback PyTorch; backbone tải từ timm)
    model_card.json           — thông tin mô hình, metrics test, cấu hình tiền xử lý
Kiểm tra: so sánh đầu ra ONNX vs PyTorch trên vài mẫu thật.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.constants import (  # noqa: E402
    AGE_GROUPS, CLIP_SECONDS, EXOG_COLS, F_MAX, F_MIN, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE,
)
from src.data.dataset import VoiceAgeArrays  # noqa: E402
from src.models.timexer_dinov3 import TimeXerDINOv3  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="timexer_dinov3_vits16")
    p.add_argument("--out", default=str(ROOT / "app" / "model"))
    p.add_argument("--no-onnx", action="store_true")
    args = p.parse_args()

    ckpt_dir = ROOT / "checkpoints" / args.run
    ck = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
    meta = json.loads((ckpt_dir / "train_meta.json").read_text(encoding="utf-8"))
    cfg = ck["cfg"]
    m = cfg["model"]
    arrays = VoiceAgeArrays.load(ROOT / cfg["data"]["processed_dir"])
    seq_len, n_exog = arrays.seq_len, arrays.n_exog
    use_prosody = bool(m.get("use_prosody", False)) and "prosody_mean" in ck
    n_prosody = len(ck["prosody_cols"]) if use_prosody else 0

    model = TimeXerDINOv3(
        seq_len=seq_len, n_exog=n_exog, n_classes=len(AGE_GROUPS), d_model=m["d_model"], n_heads=m["n_heads"],
        e_layers=m["e_layers"], d_ff=m["d_ff"], dropout=m["dropout"], dino_model=m["dino_model"], dino_pretrained=True,
        dino_img_size=m["dino_img_size"], dino_freeze=True, dino_token_mode=m["dino_token_mode"], n_prosody=n_prosody,
    )
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and all(k.startswith("en_encoder.backbone.") for k in missing), (missing, unexpected)
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    head_ck = {"state_dict": ck["state_dict"], "cfg": cfg, "epoch": ck["epoch"], "seq_len": seq_len, "n_exog": n_exog, "n_prosody": n_prosody}
    if use_prosody:
        head_ck.update({"prosody_cols": ck["prosody_cols"], "prosody_mean": ck["prosody_mean"], "prosody_std": ck["prosody_std"]})
    torch.save(head_ck, out / "timexer_dinov3_head.pt")

    idx = arrays.indices("test")[:8]
    mel = torch.from_numpy(np.asarray(arrays.mel[idx], dtype=np.float32))
    x_ex = torch.from_numpy(np.asarray(arrays.exog[idx], dtype=np.float32)).permute(0, 2, 1)
    pro = torch.from_numpy(np.asarray(arrays.prosody[idx], dtype=np.float32)) if use_prosody else None
    inputs = (mel, x_ex, pro) if use_prosody else (mel, x_ex)
    names = ["mel", "x_ex", "prosody"] if use_prosody else ["mel", "x_ex"]
    with torch.no_grad():
        ref = model(*inputs).numpy()

    onnx_ok = False
    if not args.no_onnx:
        onnx_path = out / "timexer_dinov3.onnx"
        try:
            torch.onnx.export(
                model, tuple(t[:1] for t in inputs), str(onnx_path), input_names=names, output_names=["logits"],
                dynamic_axes={**{n: {0: "batch"} for n in names}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
            )
            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            got = sess.run(None, {n: t.numpy() for n, t in zip(names, inputs)})[0]
            err = float(np.abs(got - ref).max())
            print(f"ONNX ok — max |Δlogits| = {err:.2e} — {onnx_path.stat().st_size/1e6:.1f} MB")
            onnx_ok = err < 1e-2
            if not onnx_ok:
                print("CẢNH BÁO: sai lệch ONNX lớn, app sẽ dùng fallback PyTorch")
                onnx_path.unlink()
        except Exception as e:  # noqa: BLE001
            print("Xuất ONNX thất bại (app dùng fallback PyTorch):", repr(e)[:400])
            if (out / "timexer_dinov3.onnx").exists():
                (out / "timexer_dinov3.onnx").unlink()

    tm = meta["test_metrics"]
    card = {
        "run_name": cfg["run_name"], "exported_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "TimeXer (self-attn + exog cross-attn) with DINOv3 ViT-S/16 endogenous encoder",
        "dino_model": m["dino_model"], "dino_token_mode": m["dino_token_mode"], "d_model": m["d_model"], "e_layers": m["e_layers"],
        "trainable_params": meta["trainable_params"], "best_epoch": meta["best_epoch"], "n_train": meta["n_samples"]["train"],
        "test_accuracy": tm["accuracy"], "test_balanced_accuracy": tm["balanced_accuracy"], "test_macro_f1": tm["macro_f1"],
        "test_ordinal_mae": tm["ordinal_mae"], "test_within_one": tm["within_one_group_acc"], "test_per_class_f1": tm["per_class_f1"],
        "age_groups": AGE_GROUPS, "onnx": onnx_ok, "onnx_inputs": names,
        "preprocessing": {"sample_rate": SAMPLE_RATE, "clip_seconds": CLIP_SECONDS, "n_mels": N_MELS, "n_fft": N_FFT, "hop_length": HOP_LENGTH, "fmin": F_MIN, "fmax": F_MAX, "denoise": "noisereduce stationary 0.75", "exog_cols": EXOG_COLS, "seq_len": seq_len},
        "prosody": {"cols": ck["prosody_cols"], "mean": ck["prosody_mean"], "std": ck["prosody_std"], "clip": 6.0} if use_prosody else None,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã xuất -> {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size/1e6:8.2f} MB")


if __name__ == "__main__":
    main()
