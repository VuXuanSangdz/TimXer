#!/usr/bin/env python3
"""
Đóng gói data/processed thành thư mục upload Kaggle Dataset (public):

    data/kaggle_dataset/
      dataset-metadata.json      (Kaggle API)
      README.md                  (mô tả luồng thu thập, schema, license)
      metadata.csv, features_scalar.csv, dataset_manifest.json
      mel_logspec.npy, exog_series.npy
      audio_16k_flac/<clip_id>.flac   (--with-audio)

Upload (cần ~/.kaggle/kaggle.json của bạn):
    kaggle datasets create -p data/kaggle_dataset --dir-mode zip --public
    kaggle datasets version -p data/kaggle_dataset -m "update" --dir-mode zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def kaggle_username(default: str = "YOUR_KAGGLE_USERNAME") -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    for p in (Path.home() / ".kaggle" / "kaggle.json", Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text())["username"]
            except Exception:  # noqa: BLE001
                pass
    return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    p.add_argument("--out", default=str(ROOT / "data" / "kaggle_dataset"))
    p.add_argument("--slug", default="voice-age-timexer-dinov3-vi")
    p.add_argument("--title", default="Voice Age (TimeXer-DINOv3) — Vietnamese + English, 5 age groups")
    p.add_argument("--with-audio", action="store_true", help="kèm audio 16 kHz (FLAC) đã chuẩn hoá 5 s")
    p.add_argument("--username", default=None)
    args = p.parse_args()

    src, out = Path(args.processed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ["metadata.csv", "features_scalar.csv", "dataset_manifest.json", "mel_logspec.npy", "exog_series.npy"]:
        shutil.copy2(src / f, out / f)

    if args.with_audio:
        import soundfile as sf

        wav_dir, flac_dir = src / "audio_16k", out / "audio_16k_flac"
        flac_dir.mkdir(exist_ok=True)

        def conv(w: Path):
            y, sr = sf.read(w, dtype="int16")
            sf.write(flac_dir / (w.stem + ".flac"), y, sr, format="FLAC")

        wavs = sorted(wav_dir.glob("*.wav"))
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(conv, wavs))
        print(f"FLAC: {len(wavs)} file")

    user = args.username or kaggle_username()
    manifest = json.loads((src / "dataset_manifest.json").read_text(encoding="utf-8"))
    meta = {
        "title": args.title,
        "id": f"{user}/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["audio", "speech", "age prediction", "time series", "transformer", "vietnamese"],
        "subtitle": "Voice clips -> log-Mel (endogenous) + recording context (exogenous) for TimeXer age-group classification",
        "description": "See README.md in the dataset.",
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""# Voice Age Dataset for TimeXer-DINOv3 (5 nhóm tuổi)

Dữ liệu được xây dựng theo đúng luồng ở slide thiết kế:

**Thiết bị vật lý (điện thoại / laptop / tai nghe) → Thu thập → Truyền dữ liệu → Lưu trữ → Tiền xử lý
(khử nhiễu → cắt & chuẩn hoá độ dài 5 s → trích xuất đặc trưng → chuẩn hoá chuỗi) → Xây dựng biến đầu vào
(y nội sinh: log-Mel/MFCC/F0/formant/energy; x ngoại sinh: thời gian ghi, ngữ cảnh ghi âm) → Chia train/val/test
(speaker-disjoint) → Huấn luyện TimeXer-DINOv3 → Đánh giá.**

## Nguồn
- `private_app`: bản ghi tự thu bằng app thu thập (trang `/collect` của hệ thống), có timestamp & ngữ cảnh thực.
- `common_voice_17_vi` / `common_voice_17_en`: clip seed có nhãn tuổi từ Mozilla Common Voice 17 (CC0),
  dùng để khởi động mô hình khi dữ liệu private còn ít. Với seed, các cột thời gian (hour/dow) = 0 (không biết).

## Thống kê
- Số clip: **{manifest['n_clips']}**, số người nói: **{manifest['n_speakers']}**
- Phân bố nhóm tuổi: `{manifest['label_counts']}`
- Split (theo speaker): `{manifest['split_counts']}`
- Nguồn: `{manifest['sources']}`

## Files
| File | Mô tả |
|---|---|
| `metadata.csv` | clip_id, speaker_id, age_group (nhãn), age_raw, gender, locale, source, recorded_at, device, environment, split, label (0..4), sample_index |
| `features_scalar.csv` | MFCC(20) mean/std, F0 mean/std/median, voiced_ratio, formant F1–F3, energy, ZCR, spectral centroid, SNR, RMS, duration… |
| `mel_logspec.npy` | float16 `[N, 128, 251]` — log-Mel dB (biến nội sinh **y ∈ R^(T×d_y)**), 16 kHz, hop 20 ms, 5 s |
| `exog_series.npy` | float16 `[N, 13, 251]` — biến ngoại sinh **x ∈ R^(T×d_x)**: {manifest['exogenous_x']['columns']} |
| `audio_16k_flac/` | (tuỳ chọn) waveform 5 s đã khử nhiễu / chuẩn hoá |
| `dataset_manifest.json` | schema + thống kê |

Nhóm tuổi: `<18`, `18-29`, `30-49`, `50-64`, `65+` (ánh xạ từ nhãn thập niên Common Voice: teens→<18, twenties→18-29,
thirties/fourties→30-49, fifties/sixties→50-64, seventies+→65+).

## Code
Pipeline, mô hình TimeXer-DINOv3 và app triển khai: xem GitHub repo `timexer-dinov3-voice-age`.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Đã đóng gói -> {out}  ({size:.0f} MB)")
    print("\nUpload lên Kaggle (public):")
    print(f"  kaggle datasets create -p {out} --dir-mode zip --public")
    print(f"  # cập nhật phiên bản: kaggle datasets version -p {out} -m 'v2' --dir-mode zip")
    print(f"  # link sau khi tạo: https://www.kaggle.com/datasets/{user}/{args.slug}")


if __name__ == "__main__":
    main()
