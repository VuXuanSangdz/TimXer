"""
Suy luận TimeXer-DINOv3: audio bytes -> tiền xử lý -> (mel, exog) -> mô hình -> xác suất 5 nhóm tuổi.

Ưu tiên ONNX (app/model/timexer_dinov3.onnx, xuất bởi scripts/07_export_model.py);
nếu không có thì dựng model PyTorch (backbone DINOv3 từ timm + trọng số TimeXer trong checkpoint).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.common.constants import AGE_GROUP_LABELS_VI, AGE_GROUPS  # noqa: E402
from src.data.audio_processing import process_clip  # noqa: E402

MODEL_DIR = APP_DIR / "model"


class VoiceAgePredictor:
    def __init__(self, model_dir: Path = MODEL_DIR, checkpoint: Path | None = None):
        self.model_dir = Path(model_dir)
        self.backend = None
        self.info: dict = {}
        # Chuẩn hoá prosody token (mean/std train) — lấy từ model_card.json hoặc checkpoint
        self.prosody_cols: list[str] | None = None
        self.prosody_mean: np.ndarray | None = None
        self.prosody_std: np.ndarray | None = None
        onnx_path = self.model_dir / "timexer_dinov3.onnx"
        meta_path = self.model_dir / "model_card.json"
        if meta_path.exists():
            self.info = json.loads(meta_path.read_text(encoding="utf-8"))
            if self.info.get("prosody"):
                self._set_prosody(self.info["prosody"]["cols"], self.info["prosody"]["mean"], self.info["prosody"]["std"])
        if onnx_path.exists():
            self._init_onnx(onnx_path)
        else:
            self._init_torch(checkpoint or self._find_checkpoint())

    def _set_prosody(self, cols, mean, std) -> None:
        self.prosody_cols = list(cols)
        self.prosody_mean = np.asarray(mean, dtype=np.float32)
        self.prosody_std = np.asarray(std, dtype=np.float32)

    def _prosody_vector(self, scalars: dict) -> np.ndarray | None:
        if self.prosody_cols is None:
            return None
        v = np.array([float(scalars.get(c, np.nan)) for c in self.prosody_cols], dtype=np.float32)
        v = np.where(np.isfinite(v), v, self.prosody_mean)  # thiếu -> mean train
        z = (v - self.prosody_mean) / self.prosody_std
        return np.clip(z, -6.0, 6.0).astype(np.float32)[None]  # [1, n_prosody]

    # ------------------------------------------------------------------ init
    def _find_checkpoint(self) -> Path:
        cands = [self.model_dir / "timexer_dinov3_head.pt", *sorted((ROOT / "checkpoints").glob("timexer_dinov3*/best_model.pt"))]
        for c in cands:
            if c.exists():
                return c
        raise FileNotFoundError("Không tìm thấy model ONNX hoặc checkpoint TimeXer-DINOv3.")

    def _init_onnx(self, path: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.backend = "onnxruntime"
        self.info.setdefault("model_file", path.name)

    def _init_torch(self, ckpt_path: Path) -> None:
        import torch
        import yaml

        from src.models.timexer_dinov3 import TimeXerDINOv3

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") or yaml.safe_load(open(ROOT / "configs" / "voice_age_dinov3.yaml", encoding="utf-8"))
        m = cfg["model"]
        if "prosody_cols" in ck and self.prosody_cols is None:
            self._set_prosody(ck["prosody_cols"], ck["prosody_mean"], ck["prosody_std"])
        n_prosody = len(self.prosody_cols) if (self.prosody_cols and m.get("use_prosody", False)) else 0
        if n_prosody == 0:
            self.prosody_cols = None
        self.model = TimeXerDINOv3(
            seq_len=int(ck.get("seq_len", 251)), n_exog=int(ck.get("n_exog", 13)), n_classes=len(AGE_GROUPS),
            d_model=m["d_model"], n_heads=m["n_heads"], e_layers=m["e_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
            dino_model=m["dino_model"], dino_pretrained=True, dino_img_size=m["dino_img_size"],
            dino_freeze=True, dino_token_mode=m["dino_token_mode"], n_prosody=n_prosody,
        )
        self.model.load_state_dict(ck["state_dict"], strict=False)
        self.model.eval()
        torch.set_num_threads(4)
        self.backend = "pytorch"
        self.info.setdefault("run_name", cfg.get("run_name", "timexer_dinov3"))
        self.info.setdefault("model_file", str(ckpt_path.name))

    # ------------------------------------------------------------------ predict
    def predict(self, audio: bytes | str, recorded_at: datetime | None = None, source_mic: bool = False) -> dict:
        t0 = time.time()
        y, feats = process_clip(audio, recorded_at=recorded_at or datetime.now(), source_mic=source_mic, compute_formants=True)
        mel = feats.log_mel.astype(np.float32)[None]  # [1,128,T]
        x_ex = feats.exog.astype(np.float32).T[None]  # [1,T,13]
        prosody = self._prosody_vector(feats.scalars)  # [1,51] hoặc None
        logits = self._run(mel, x_ex, prosody)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        k = int(probs.argmax())
        return {
            "pred_group": AGE_GROUPS[k],
            "pred_label_vi": AGE_GROUP_LABELS_VI[AGE_GROUPS[k]],
            "confidence": float(probs[k]),
            "probs": {g: float(p) for g, p in zip(AGE_GROUPS, probs)},
            "features": {
                "speech_duration_s": round(float(feats.scalars.get("speech_duration_s", 0)), 2),
                "raw_duration_s": round(float(feats.scalars.get("raw_duration_s", 0)), 2),
                "f0_mean_hz": round(float(feats.scalars.get("f0_mean", 0)), 1),
                "formant_f1_hz": round(float(feats.scalars.get("formant_f1", 0)), 0),
                "formant_f2_hz": round(float(feats.scalars.get("formant_f2", 0)), 0),
                "snr_db": round(float(feats.scalars.get("snr_db", 0)), 1),
                "orig_sr": int(feats.scalars.get("orig_sr", 16000)),
            },
            "latency_ms": int((time.time() - t0) * 1000),
            "backend": self.backend,
            "model": self.info.get("run_name", "timexer_dinov3"),
        }

    def _run(self, mel: np.ndarray, x_ex: np.ndarray, prosody: np.ndarray | None) -> np.ndarray:
        if self.backend == "onnxruntime":
            feed = {"mel": mel, "x_ex": x_ex}
            if "prosody" in self.in_names:
                if prosody is None:
                    raise RuntimeError("Model ONNX cần đầu vào 'prosody' nhưng model_card.json thiếu thông tin chuẩn hoá.")
                feed["prosody"] = prosody
            return self.sess.run(None, feed)[0][0]
        import torch

        with torch.no_grad():
            pro = torch.from_numpy(prosody) if (prosody is not None and self.model.prosody_embedding is not None) else None
            return self.model(torch.from_numpy(mel), torch.from_numpy(x_ex), pro)[0].numpy()
