#!/usr/bin/env python3
"""
Bước "Thu thập" (seed): tải clip có nhãn tuổi từ Common Voice 17 (mirror fixie-ai, không gate)
và ghi vào data/raw/ theo CÙNG schema với bản ghi private (xem scripts/02_ingest_private.py):

    data/raw/<source>/<clip_id>.mp3|wav
    data/raw/raw_manifest.csv   (clip_id, path, source, speaker_id, age_group, age_raw, gender, locale,
                                 recorded_at, device, environment, sentence)

Mặc định: tiếng Việt (validated + other) + vài shard tiếng Anh (validated) với giới hạn số clip MỖI nhóm tuổi
(--sup-cap) để bổ sung nhóm 50-64 / 65+ vốn rất hiếm trong Common Voice tiếng Việt mà vẫn có đủ mẫu tiếng Anh ở
các nhóm khác (tránh mô hình học "ngôn ngữ = tuổi").
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.constants import COMMON_VOICE_AGE_MAP  # noqa: E402

REPO = "fixie-ai/common_voice_17_0"
MANIFEST_COLS = [
    "clip_id", "path", "source", "speaker_id", "age_group", "age_raw", "gender",
    "locale", "recorded_at", "device", "environment", "sentence",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "raw"))
    p.add_argument("--vi-splits", nargs="*", default=["validated", "other"])
    p.add_argument("--vi-max-shards", type=int, default=16)
    # Bổ sung (supplement) từ locale khác cho MỌI nhóm tuổi (có giới hạn/nhóm) để tránh "ngôn ngữ = tuổi"
    p.add_argument("--sup-locale", default="en")
    p.add_argument("--sup-split", default="validated")
    p.add_argument("--sup-shards", type=int, default=2, help="0 = không bổ sung")
    p.add_argument(
        "--sup-cap",
        default="<18:400,18-29:500,30-49:600,50-64:900,65+:900",
        help="giới hạn clip mỗi nhóm tuổi cho phần bổ sung",
    )
    p.add_argument("--max-per-speaker", type=int, default=40, help="giới hạn clip/speaker để tránh lệch")
    p.add_argument("--max-clips", type=int, default=0, help="0 = không giới hạn")
    return p.parse_args()


def iter_shards(cfg: str, split: str):
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    if cfg == "vi":
        prefix = f"{cfg}/{split}/"
        return sorted(f for f in files if f.startswith(prefix))
    return sorted(f for f in files if f.startswith(f"{cfg}/{split}-"))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "raw_manifest.csv"
    existing = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8", newline="") as f:
            existing = {r["clip_id"] for r in csv.DictReader(f)}
    fh = open(manifest_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
    if not existing:
        writer.writeheader()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    cache = ROOT / "data" / "hf_cache"
    per_speaker: dict[str, int] = {}
    total = 0

    jobs: list[tuple[str, str, int, dict[str, int] | None]] = [
        ("vi", s, args.vi_max_shards, None) for s in args.vi_splits
    ]
    if args.sup_shards > 0:
        caps = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in args.sup_cap.split(",") if kv}
        jobs.append((args.sup_locale, args.sup_split, args.sup_shards, caps))

    for cfg, split, max_shards, caps in jobs:
        shards = iter_shards(cfg, split)[:max_shards]
        print(f"[{cfg}/{split}] {len(shards)} shard")
        src_dir = out / f"common_voice_{cfg}"
        src_dir.mkdir(exist_ok=True)
        group_count: dict[str, int] = {}
        for shard in shards:
            local = hf_hub_download(REPO, shard, repo_type="dataset", cache_dir=str(cache))
            pf = pq.ParquetFile(local)
            kept = 0
            for batch in pf.iter_batches(batch_size=64, columns=["client_id", "path", "audio", "sentence", "age", "gender", "locale"]):
                for row in batch.to_pylist():
                    age = (row.get("age") or "").strip()
                    group = COMMON_VOICE_AGE_MAP.get(age)
                    if group is None:
                        continue
                    if caps is not None and group_count.get(group, 0) >= caps.get(group, 10**9):
                        continue
                    spk = row["client_id"][:24]
                    if per_speaker.get(spk, 0) >= args.max_per_speaker:
                        continue
                    audio = row["audio"]
                    blob = audio.get("bytes") if isinstance(audio, dict) else None
                    if not blob:
                        continue
                    name = Path(audio.get("path") or row["path"] or f"{spk}_{total}.mp3").name
                    clip_id = f"cv{cfg}_{Path(name).stem}"
                    if clip_id in existing:
                        continue
                    ext = Path(name).suffix or ".mp3"
                    dst = src_dir / f"{clip_id}{ext}"
                    dst.write_bytes(blob)
                    writer.writerow(
                        {
                            "clip_id": clip_id,
                            "path": str(dst.relative_to(out)).replace("\\", "/"),
                            "source": f"common_voice_17_{cfg}",
                            "speaker_id": spk,
                            "age_group": group,
                            "age_raw": age,
                            "gender": (row.get("gender") or "").split("_")[0],
                            "locale": row.get("locale") or cfg,
                            "recorded_at": "",
                            "device": "unknown",
                            "environment": "unknown",
                            "sentence": (row.get("sentence") or "").replace("\n", " "),
                        }
                    )
                    existing.add(clip_id)
                    per_speaker[spk] = per_speaker.get(spk, 0) + 1
                    group_count[group] = group_count.get(group, 0) + 1
                    kept += 1
                    total += 1
                    if args.max_clips and total >= args.max_clips:
                        break
                if args.max_clips and total >= args.max_clips:
                    break
            fh.flush()
            print(f"   {Path(shard).name}: giữ {kept} clip (tổng {total})")
            if args.max_clips and total >= args.max_clips:
                break
    fh.close()
    print(f"\nXong. Manifest: {manifest_path} — {total} clip mới, {len(per_speaker)} speaker.")


if __name__ == "__main__":
    main()
