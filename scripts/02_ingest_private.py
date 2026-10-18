#!/usr/bin/env python3
"""
Bước "Thu thập -> Truyền dữ liệu -> Lưu trữ" cho DỮ LIỆU PRIVATE.

Nguồn 1: thư mục do app thu thập (app/backend, trang /collect) ghi ra:
            data/raw/private/<clip_id>.webm|wav  +  <clip_id>.json  (metadata người đóng góp)
Nguồn 2: thư mục bất kỳ chứa file âm thanh + file CSV mô tả (clip, age hoặc age_group, gender, ...).

Script hợp nhất vào data/raw/raw_manifest.csv (cùng schema với seed Common Voice).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.constants import AGE_GROUPS, age_years_to_group  # noqa: E402

MANIFEST_COLS = [
    "clip_id", "path", "source", "speaker_id", "age_group", "age_raw", "gender",
    "locale", "recorded_at", "device", "environment", "sentence",
]
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--private-dir", default=str(ROOT / "data" / "raw" / "private"))
    p.add_argument("--csv", default=None, help="CSV mô tả (tuỳ chọn) cho thư mục ngoài")
    p.add_argument("--audio-dir", default=None, help="thư mục âm thanh đi kèm --csv")
    return p.parse_args()


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def save_manifest(path: Path, rows: dict[str, dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for r in rows.values():
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLS})


def group_from(meta: dict) -> str | None:
    g = (meta.get("age_group") or "").strip()
    if g in AGE_GROUPS:
        return g
    age = meta.get("age")
    try:
        return age_years_to_group(float(age)) if age not in (None, "") else None
    except ValueError:
        return None


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    manifest_path = raw_dir / "raw_manifest.csv"
    rows = load_manifest(manifest_path)
    added = 0

    # Nguồn 1: app thu thập
    pdir = Path(args.private_dir)
    if pdir.exists():
        for meta_file in sorted(pdir.glob("*.json")):
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            audio = next((pdir / f"{meta_file.stem}{e}" for e in AUDIO_EXT if (pdir / f"{meta_file.stem}{e}").exists()), None)
            if audio is None:
                continue
            clip_id = f"priv_{meta_file.stem}"
            group = group_from(meta)
            if group is None or clip_id in rows:
                continue
            rows[clip_id] = {
                "clip_id": clip_id,
                "path": str(audio.relative_to(raw_dir)).replace("\\", "/"),
                "source": "private_app",
                "speaker_id": meta.get("speaker_id") or f"anon_{abs(hash(meta.get('user_agent', ''))) % 10**8}",
                "age_group": group,
                "age_raw": str(meta.get("age", "")),
                "gender": meta.get("gender", ""),
                "locale": meta.get("locale", "vi"),
                "recorded_at": meta.get("recorded_at") or datetime.fromtimestamp(audio.stat().st_mtime).isoformat(),
                "device": meta.get("device", "unknown"),
                "environment": meta.get("environment", "unknown"),
                "sentence": meta.get("sentence", ""),
            }
            added += 1

    # Nguồn 2: CSV ngoài
    if args.csv:
        adir = Path(args.audio_dir or Path(args.csv).parent)
        with open(args.csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                audio = adir / r.get("path", r.get("clip", ""))
                if not audio.exists():
                    continue
                clip_id = f"ext_{audio.stem}"
                group = group_from(r)
                if group is None or clip_id in rows:
                    continue
                rows[clip_id] = {
                    "clip_id": clip_id,
                    "path": str(audio.resolve()),
                    "source": r.get("source", "external_csv"),
                    "speaker_id": r.get("speaker_id", audio.stem),
                    "age_group": group,
                    "age_raw": r.get("age", ""),
                    "gender": r.get("gender", ""),
                    "locale": r.get("locale", "vi"),
                    "recorded_at": r.get("recorded_at", ""),
                    "device": r.get("device", "unknown"),
                    "environment": r.get("environment", "unknown"),
                    "sentence": r.get("sentence", ""),
                }
                added += 1

    save_manifest(manifest_path, rows)
    print(f"Đã thêm {added} clip private. Tổng manifest: {len(rows)} clip -> {manifest_path}")


if __name__ == "__main__":
    main()
