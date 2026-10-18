"""
Tiền xử lý & trích xuất đặc trưng giọng nói (khối "Tiền xử lý dữ liệu" + "Xây dựng biến đầu vào" ở slide 2).

Luồng:  audio thô -> khử nhiễu -> cắt im lặng & chuẩn hoá độ dài -> chuẩn hoá biên độ
        -> (y) log-Mel spectrogram, MFCC, F0/pitch, formant, energy
        -> (x) biến ngoại sinh: thời gian ghi, ngữ cảnh ghi âm, tín hiệu hỗ trợ theo khung
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.common.constants import (
    CLIP_SECONDS,
    EXOG_FRAME_COLS,
    EXOG_STATIC_COLS,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
)

N_FRAMES = int(CLIP_SECONDS * SAMPLE_RATE) // HOP_LENGTH + 1  # 251


# --------------------------------------------------------------------------------------
# 1) Nạp & khử nhiễu
# --------------------------------------------------------------------------------------
def load_audio(source: str | bytes | io.BytesIO, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Đọc file (wav/mp3/flac/ogg/webm*) -> mono float32 @ target_sr. Trả về (y, sr_gốc)."""
    import librosa
    import soundfile as sf

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        y, sr = sf.read(source, dtype="float32", always_2d=False)
    except Exception:
        # fallback: librosa/audioread (webm/ogg từ trình duyệt)
        if isinstance(source, io.BytesIO):
            source.seek(0)
        y, sr = librosa.load(source, sr=None, mono=True)
    if y.ndim > 1:
        y = y.mean(axis=1)
    orig_sr = int(sr)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), orig_sr


def denoise(y: np.ndarray, sr: int = SAMPLE_RATE, strength: float = 0.75) -> np.ndarray:
    """Khử nhiễu nền bằng spectral gating (noisereduce). strength∈[0,1]."""
    if len(y) < sr // 4:
        return y
    try:
        import noisereduce as nr

        return nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=strength).astype(np.float32)
    except Exception:
        return y


# --------------------------------------------------------------------------------------
# 2) Cắt & chuẩn hoá độ dài, chuẩn hoá biên độ
# --------------------------------------------------------------------------------------
def trim_and_fix_length(
    y: np.ndarray, sr: int = SAMPLE_RATE, seconds: float = CLIP_SECONDS, top_db: float = 30.0
) -> tuple[np.ndarray, float]:
    """Cắt im lặng đầu/cuối rồi crop giữa hoặc pad về đúng `seconds`. Trả về (y, duration_sau_trim)."""
    import librosa

    y_t, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_t) < sr // 10:  # gần như im lặng hoàn toàn -> giữ nguyên
        y_t = y
    duration = len(y_t) / sr
    target = int(seconds * sr)
    if len(y_t) >= target:
        start = (len(y_t) - target) // 2
        y_t = y_t[start : start + target]
    else:
        pad = target - len(y_t)
        y_t = np.pad(y_t, (pad // 2, pad - pad // 2), mode="constant")
    return y_t.astype(np.float32), float(duration)


def normalize_amplitude(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / m * peak).astype(np.float32) if m > 1e-6 else y


def preprocess_waveform(
    y: np.ndarray, sr: int = SAMPLE_RATE, do_denoise: bool = True
) -> tuple[np.ndarray, dict[str, float]]:
    """Toàn bộ khối tiền xử lý. Trả về (waveform 5 s @16 kHz, thống kê)."""
    stats: dict[str, float] = {"raw_duration_s": len(y) / sr}
    if do_denoise:
        y = denoise(y, sr)
    y, dur = trim_and_fix_length(y, sr)
    y = normalize_amplitude(y)
    stats["speech_duration_s"] = dur
    return y, stats


# --------------------------------------------------------------------------------------
# 3) Biến nội sinh y — đặc trưng giọng nói
# --------------------------------------------------------------------------------------
def log_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """[N_MELS, T] log-mel (dB)."""
    import librosa

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def mfcc_features(log_mel: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC).astype(np.float32)  # [N_MFCC, T]


def pitch_contour(y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """F0 theo khung (Hz) bằng YIN + mặt nạ voiced. Nhanh hơn pYIN nhiều, đủ cho thống kê."""
    import librosa

    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    voiced = (rms > np.percentile(rms, 35)) & (f0 > 60) & (f0 < 500)
    f0 = np.where(voiced, f0, 0.0)
    return f0.astype(np.float32), voiced


def formant_estimates(y: np.ndarray, sr: int = SAMPLE_RATE, n_formants: int = 3) -> np.ndarray:
    """Ước lượng F1..F3 (Hz) bằng LPC trên các khung năng lượng cao. Trả về median."""
    import librosa

    order = 2 + sr // 1000
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
    frame_len, hop = 2048, 1024
    rms = librosa.feature.rms(y=y_pre, frame_length=frame_len, hop_length=hop)[0]
    idx = np.argsort(rms)[::-1][:12]
    window = np.hamming(frame_len)
    formants = []
    for i in idx:
        start = i * hop
        frame = y_pre[start : start + frame_len]
        if len(frame) < frame_len or np.abs(frame).max() < 1e-4:
            continue
        try:
            a = librosa.lpc(frame * window, order=order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0.01]
            ang = np.arctan2(np.imag(roots), np.real(roots))
            freqs = np.sort(ang * (sr / (2 * math.pi)))
            bw = -0.5 * (sr / (2 * math.pi)) * np.log(np.abs(roots[np.argsort(ang)]))
            cand = [f for f, b in zip(freqs, bw) if 90 < f < 5000 and b < 400]
            if len(cand) >= n_formants:
                formants.append(cand[:n_formants])
        except Exception:
            continue
    if not formants:
        return np.zeros(n_formants, dtype=np.float32)
    return np.median(np.array(formants), axis=0).astype(np.float32)


def energy_contour(y: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0].astype(np.float32)


def _fit_frames(a: np.ndarray, n: int = N_FRAMES) -> np.ndarray:
    if a.shape[-1] == n:
        return a
    if a.shape[-1] > n:
        return a[..., :n]
    pad = n - a.shape[-1]
    return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)], mode="edge")


@dataclass
class VoiceFeatures:
    """Toàn bộ đặc trưng của một clip (y nội sinh + x ngoại sinh + thống kê scalar)."""

    log_mel: np.ndarray  # [N_MELS, T]
    exog: np.ndarray  # [len(EXOG_COLS), T]  (static được broadcast theo T)
    scalars: dict[str, float] = field(default_factory=dict)


def estimate_snr_db(energy: np.ndarray) -> float:
    e = energy[energy > 0]
    if len(e) < 10:
        return 0.0
    hi = np.mean(np.sort(e)[int(0.8 * len(e)) :])
    lo = np.mean(np.sort(e)[: max(1, int(0.2 * len(e)))])
    return float(20 * np.log10(hi / (lo + 1e-8) + 1e-8))


# --------------------------------------------------------------------------------------
# 4) Biến ngoại sinh x — thời gian ghi, ngữ cảnh ghi âm, tín hiệu hỗ trợ
# --------------------------------------------------------------------------------------
def static_exogenous(
    recorded_at: datetime | None,
    speech_duration_s: float,
    snr_db: float,
    rms_db: float,
    source_mic: bool,
    orig_sr: int,
) -> dict[str, float]:
    if recorded_at is None:
        # Không biết thời điểm ghi (vd. dữ liệu seed) -> vector thời gian = 0 (trung tính)
        time_feats = {"hour_sin": 0.0, "hour_cos": 0.0, "dow_sin": 0.0, "dow_cos": 0.0}
    else:
        h = recorded_at.hour + recorded_at.minute / 60.0
        dow = recorded_at.weekday()
        time_feats = {
            "hour_sin": math.sin(2 * math.pi * h / 24),
            "hour_cos": math.cos(2 * math.pi * h / 24),
            "dow_sin": math.sin(2 * math.pi * dow / 7),
            "dow_cos": math.cos(2 * math.pi * dow / 7),
        }
    return {
        **time_feats,
        "duration_norm": min(speech_duration_s, 15.0) / 15.0,
        "snr_db_norm": float(np.clip(snr_db, 0, 60)) / 60.0,
        "rms_db_norm": float(np.clip(rms_db + 60, 0, 60)) / 60.0,
        "source_mic": 1.0 if source_mic else 0.0,
        "orig_sr_norm": min(orig_sr, 48_000) / 48_000.0,
    }


def extract_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    recorded_at: datetime | None = None,
    source_mic: bool = False,
    orig_sr: int = SAMPLE_RATE,
    speech_duration_s: float | None = None,
    compute_formants: bool = True,
) -> VoiceFeatures:
    """y đã qua preprocess_waveform (5 s @ 16 kHz)."""
    log_mel = _fit_frames(log_mel_spectrogram(y, sr))
    mfcc = mfcc_features(log_mel)
    f0, voiced = pitch_contour(y, sr)
    energy = energy_contour(y)
    import librosa

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

    f0, energy, zcr, centroid = (_fit_frames(a) for a in (f0, energy, zcr, centroid))
    voiced = _fit_frames(voiced.astype(np.float32)) > 0.5

    rms_db = float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-8))
    snr_db = estimate_snr_db(energy)
    if speech_duration_s is None:
        speech_duration_s = len(y) / sr

    stat = static_exogenous(recorded_at, speech_duration_s, snr_db, rms_db, source_mic, orig_sr)

    # tín hiệu hỗ trợ theo khung (chuẩn hoá về ~[0,1])
    frame_series = {
        "f0_contour": f0 / 500.0,
        "energy_contour": energy / (energy.max() + 1e-8),
        "zcr_contour": zcr,
        "centroid_contour": centroid / (sr / 2),
    }
    T = log_mel.shape[1]
    exog = np.stack(
        [np.full(T, stat[c], dtype=np.float32) for c in EXOG_STATIC_COLS]
        + [frame_series[c].astype(np.float32) for c in EXOG_FRAME_COLS],
        axis=0,
    )

    scalars: dict[str, Any] = {**stat, "snr_db": snr_db, "rms_db": rms_db, "speech_duration_s": speech_duration_s}
    f0v = f0[voiced] if voiced.any() else np.array([0.0])
    scalars.update(
        {
            "f0_mean": float(f0v.mean()),
            "f0_std": float(f0v.std()),
            "f0_median": float(np.median(f0v)),
            "voiced_ratio": float(voiced.mean()),
            "energy_mean": float(energy.mean()),
            "energy_std": float(energy.std()),
            "zcr_mean": float(zcr.mean()),
            "centroid_mean": float(centroid.mean()),
        }
    )
    for i in range(N_MFCC):
        scalars[f"mfcc{i:02d}_mean"] = float(mfcc[i].mean())
        scalars[f"mfcc{i:02d}_std"] = float(mfcc[i].std())
    if compute_formants:
        f1, f2, f3 = formant_estimates(y, sr)
        scalars.update({"formant_f1": float(f1), "formant_f2": float(f2), "formant_f3": float(f3)})
    return VoiceFeatures(log_mel=log_mel.astype(np.float16), exog=exog.astype(np.float16), scalars=scalars)


def process_clip(
    source: str | bytes,
    recorded_at: datetime | None = None,
    source_mic: bool = False,
    do_denoise: bool = True,
    compute_formants: bool = True,
) -> tuple[np.ndarray, VoiceFeatures]:
    """Tiện ích end-to-end: file/bytes -> (waveform chuẩn hoá, VoiceFeatures)."""
    y, orig_sr = load_audio(source)
    y, st = preprocess_waveform(y, SAMPLE_RATE, do_denoise=do_denoise)
    feats = extract_features(
        y,
        SAMPLE_RATE,
        recorded_at=recorded_at,
        source_mic=source_mic,
        orig_sr=orig_sr,
        speech_duration_s=st["speech_duration_s"],
        compute_formants=compute_formants,
    )
    feats.scalars["raw_duration_s"] = st["raw_duration_s"]
    feats.scalars["orig_sr"] = orig_sr
    return y, feats
