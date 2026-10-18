#!/usr/bin/env python3
"""
Chạy backbone DINOv3 (đóng băng) một lần trên toàn bộ log-Mel và cache token thô:
    data/processed/dino_tokens_<tag>.npy  — [N, 1 + N_tok, embed_dim] float16 (token 0 = CLS)
Cho phép huấn luyện TimeXer (phần (b)(c)(d) + head) rất nhanh trên CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import VoiceAgeArrays  # noqa: E402
from src.models.dinov3_encoder import DINOv3Encoder  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "voice_age_dinov3.yaml"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.threads:
        torch.set_num_threads(args.threads)

    arrays = VoiceAgeArrays.load(ROOT / cfg["data"]["processed_dir"])
    m = cfg["model"]
    enc = DINOv3Encoder(
        d_model=m["d_model"], model_name=m["dino_model"], pretrained=True, img_size=m["dino_img_size"],
        freeze_backbone=True, token_mode=m["dino_token_mode"],
    ).eval()
    N = len(arrays.meta)
    out_path = ROOT / cfg["data"]["dino_cache"]
    first = enc.encode_backbone(torch.from_numpy(np.asarray(arrays.mel[:1], dtype=np.float32)))
    cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(N, first.shape[1], first.shape[2]))
    print(f"Model {m['dino_model']} | token shape {tuple(first.shape[1:])} | N={N} | threads={torch.get_num_threads()}")
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            mel = torch.from_numpy(np.asarray(arrays.mel[s : s + args.batch_size], dtype=np.float32))
            cache[s : s + len(mel)] = enc.encode_backbone(mel).numpy().astype(np.float16)
            if (s // args.batch_size) % 10 == 0:
                done = s + len(mel)
                print(f"  {done}/{N}  {time.time()-t0:.0f}s  (~{(time.time()-t0)/done*(N-done):.0f}s còn lại)")
    cache.flush()
    print(f"Đã lưu {out_path}  ({out_path.stat().st_size/1e6:.1f} MB) trong {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
