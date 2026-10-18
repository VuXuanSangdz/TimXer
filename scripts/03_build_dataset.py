#!/usr/bin/env python3
"""
Bước "Tiền xử lý dữ liệu" + "Xây dựng biến đầu vào" + "Chia tập train/val/test" (slide 2).

Đầu vào : data/raw/raw_manifest.csv (+ file âm thanh)
Đầu ra  : data/processed/
            metadata.csv          — nhãn, speaker, split, nguồn, thời điểm ghi ...
            features_scalar.csv   — MFCC stats, F0, formant, energy, SNR ... (dùng cho baseline/EDA)
            mel_logspec.npy       — [N, 128, 251] float16 : biến nội sinh y (log-Mel)
            exog_series.npy       — [N, 13, 251] float16  : biến ngoại sinh x
            audio_16k/<clip_id>.wav (tuỳ chọn --save-wav) : waveform 5 s đã chuẩn hoá
            dataset_manifest.json — mô tả schema, thống kê, ánh xạ nhãn
Chia tập theo speaker (speaker-disjoint) để tránh rò rỉ danh tính người nói.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.constants import AGE_GROUPS, AGE_GROUP_TO_ID, EXOG_COLS, N_MELS, SAMPLE_RATE  # noqa: E402
from src.data.audio_processing import N_FRAMES, process_clip  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--out", default=str(ROOT / "data" / "processed"))
    p.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 2))
    p.add_argument("--save-wav", action="store_true", help="lưu thêm wav 16 kHz đã chuẩn hoá")
    p.add_argument("--no-denoise", action="store_true")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _worker(task: tuple[str, str, str, bool, bool, bool, str]) -> dict | None:
    clip_id, path, recorded_at, source_mic, do_denoise, save_wav, wav_dir = task
    try:
        rec = datetime.fromisoformat(recorded_at) if recorded_at else None
        y, feats = process_clip(path, recorded_at=rec, source_mic=source_mic, do_denoise=do_denoise)
        if save_wav:
            import soundfile as sf

            sf.write(str(Path(wav_dir) / f"{clip_id}.wav"), y, SAMPLE_RATE, subtype="PCM_16")
        return {"clip_id": clip_id, "log_mel": feats.log_mel, "exog": feats.exog, "scalars": feats.scalars}
    except Exception as e:  # noqa: BLE001
        return {"clip_id": clip_id, "error": repr(e)}


def speaker_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """Chia theo speaker, phân tầng xấp xỉ theo nhóm tuổi (nhóm tuổi chủ đạo của speaker)."""
    rng = np.random.default_rng(seed)
    spk_group = df.groupby("speaker_id")["age_group"].agg(lambda s: s.value_counts().index[0])
    split = {}
    for g in AGE_GROUPS:
        spks = spk_group[spk_group == g].index.to_numpy()
        rng.shuffle(spks)
        n = len(spks)
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        for i, s in enumerate(spks):
            split[s] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return df["speaker_id"].map(split).fillna("train")


def main():
    args = parse_args()
    raw_dir, out = Path(args.raw_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = out / "audio_16k"
    if args.save_wav:
        wav_dir.mkdir(exist_ok=True)

    man = pd.read_csv(raw_dir / "raw_manifest.csv", dtype=str, keep_default_na=False)
    man = man[man["age_group"].isin(AGE_GROUPS)].drop_duplicates("clip_id").reset_index(drop=True)
    if args.max_clips:
        man = man.sample(n=min(args.max_clips, len(man)), random_state=args.seed).reset_index(drop=True)
    print(f"Clip có nhãn: {len(man)} | speaker: {man.speaker_id.nunique()}")
    print(man.age_group.value_counts().reindex(AGE_GROUPS).to_string())

    def abs_path(p: str) -> str:
        return p if Path(p).is_absolute() else str(raw_dir / p)

    tasks = [
        (r.clip_id, abs_path(r.path), r.recorded_at, r.source == "private_app", not args.no_denoise, args.save_wav, str(wav_dir))
        for r in man.itertuples()
    ]

    results: dict[str, dict] = {}
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is None or "error" in r:
                errors += 1
                if errors <= 5:
                    print("  lỗi:", r)
                continue
            results[r["clip_id"]] = r
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)} xong ({errors} lỗi)")

    man = man[man.clip_id.isin(results)].reset_index(drop=True)
    N = len(man)
    mel = np.zeros((N, N_MELS, N_FRAMES), dtype=np.float16)
    exog = np.zeros((N, len(EXOG_COLS), N_FRAMES), dtype=np.float16)
    scal_rows = []
    for i, cid in enumerate(man.clip_id):
        r = results[cid]
        mel[i] = r["log_mel"]
        exog[i] = r["exog"]
        scal_rows.append({"clip_id": cid, **r["scalars"]})
    scal = pd.DataFrame(scal_rows)

    man["label"] = man["age_group"].map(AGE_GROUP_TO_ID)
    man["split"] = speaker_split(man, args.val_frac, args.test_frac, args.seed)
    man["sample_index"] = np.arange(N)
    man["clip_hash"] = [hashlib.sha1(c.encode()).hexdigest()[:10] for c in man.clip_id]

    np.save(out / "mel_logspec.npy", mel)
    np.save(out / "exog_series.npy", exog)
    man.to_csv(out / "metadata.csv", index=False, encoding="utf-8")
    scal.to_csv(out / "features_scalar.csv", index=False, encoding="utf-8")

    manifest = {
        "name": "voice-age-timexer-vi",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_clips": int(N),
        "n_speakers": int(man.speaker_id.nunique()),
        "n_errors": int(errors),
        "age_groups": AGE_GROUPS,
        "label_counts": man.age_group.value_counts().reindex(AGE_GROUPS).fillna(0).astype(int).to_dict(),
        "split_counts": man.split.value_counts().to_dict(),
        "split_by_group": man.groupby(["split", "age_group"]).size().unstack(fill_value=0).to_dict(),
        "sources": man.source.value_counts().to_dict(),
        "audio": {"sample_rate": SAMPLE_RATE, "clip_seconds": 5.0, "denoise": not args.no_denoise},
        "endogenous_y": {"file": "mel_logspec.npy", "shape": list(mel.shape), "dtype": "float16", "desc": "log-Mel dB [N_MELS, T]"},
        "exogenous_x": {"file": "exog_series.npy", "shape": list(exog.shape), "dtype": "float16", "columns": EXOG_COLS},
        "scalar_features": [c for c in scal.columns if c != "clip_id"],
        "split_strategy": "speaker-disjoint, stratified by speaker's dominant age group",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_clips", "n_speakers", "label_counts", "split_counts")}, indent=1, ensure_ascii=False))
    print(f"\nĐã ghi: {out}")


if __name__ == "__main__":
    main()
