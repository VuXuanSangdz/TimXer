"""
TimeXer AI — API + web UI (slide 3) + endpoint thu thập dữ liệu private (slide 2).

Chạy local:   uvicorn app.backend.main:app --host 0.0.0.0 --port 7860 --reload
Docker/HF:    xem app/Dockerfile
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))

from app.backend.inference import VoiceAgePredictor  # noqa: E402
from src.common.constants import AGE_GROUP_LABELS_VI, AGE_GROUPS, age_years_to_group  # noqa: E402

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", APP_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_RAW_DIR = Path(os.environ.get("PRIVATE_RAW_DIR", ROOT / "data" / "raw" / "private"))
DB_PATH = DATA_DIR / "history.db"
MAX_UPLOAD_MB = 15

app = FastAPI(title="TimeXer AI — Dự đoán độ tuổi từ giọng nói", version="1.0.0")
predictor: VoiceAgePredictor | None = None


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, ts TEXT, filename TEXT, source TEXT, pred_group TEXT, confidence REAL, probs TEXT, features TEXT, latency_ms INTEGER)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS contributions (id TEXT PRIMARY KEY, ts TEXT, age_group TEXT, gender TEXT, locale TEXT, device TEXT, environment TEXT, path TEXT)"
    )
    return con


@app.on_event("startup")
def _startup() -> None:
    global predictor
    predictor = VoiceAgePredictor()
    db().close()


# ----------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "backend": predictor.backend if predictor else None}


@app.get("/api/model")
def model_info():
    return {"age_groups": AGE_GROUPS, "labels_vi": AGE_GROUP_LABELS_VI, **(predictor.info if predictor else {}), "backend": predictor.backend if predictor else None}


async def _read_upload(file: UploadFile) -> bytes:
    blob = await file.read()
    if not blob:
        raise HTTPException(400, "File rỗng")
    if len(blob) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File quá lớn (> {MAX_UPLOAD_MB} MB)")
    return blob


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), source: str = Form("upload"), recorded_at: str | None = Form(None)):
    assert predictor is not None
    blob = await _read_upload(file)
    try:
        rec = datetime.fromisoformat(recorded_at) if recorded_at else datetime.now()
    except ValueError:
        rec = datetime.now()
    try:
        res = predictor.predict(blob, recorded_at=rec, source_mic=(source == "mic"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Không đọc được âm thanh: {e}") from e
    rid = uuid.uuid4().hex[:12]
    res["id"] = rid
    res["timestamp"] = rec.isoformat(timespec="seconds")
    res["filename"] = file.filename or ("recording.wav" if source == "mic" else "upload")
    con = db()
    con.execute(
        "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?)",
        (rid, res["timestamp"], res["filename"], source, res["pred_group"], res["confidence"], json.dumps(res["probs"]), json.dumps(res["features"]), res["latency_ms"]),
    )
    con.commit(); con.close()
    return res


@app.get("/api/history")
def history(limit: int = 20):
    con = db()
    rows = con.execute("SELECT id, ts, filename, source, pred_group, confidence, probs, features, latency_ms FROM history ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [
        {"id": r[0], "timestamp": r[1], "filename": r[2], "source": r[3], "pred_group": r[4], "pred_label_vi": AGE_GROUP_LABELS_VI.get(r[4], r[4]), "confidence": r[5], "probs": json.loads(r[6]), "features": json.loads(r[7]), "latency_ms": r[8]}
        for r in rows
    ]


@app.delete("/api/history")
def clear_history():
    con = db(); con.execute("DELETE FROM history"); con.commit(); con.close()
    return {"ok": True}


@app.post("/api/collect")
async def collect(
    file: UploadFile = File(...),
    age: str | None = Form(None),
    age_group: str | None = Form(None),
    gender: str = Form(""),
    locale: str = Form("vi"),
    device: str = Form("unknown"),
    environment: str = Form("unknown"),
    speaker_id: str = Form(""),
    sentence: str = Form(""),
    consent: str = Form("false"),
):
    """Thu thập dữ liệu private: lưu audio + metadata vào data/raw/private (đưa vào pipeline bằng scripts/02_ingest_private.py)."""
    if consent.lower() not in ("true", "1", "yes", "on"):
        raise HTTPException(400, "Cần đồng ý chia sẻ dữ liệu (consent)")
    group = (age_group or "").strip()
    if group not in AGE_GROUPS:
        raw = (age or "").strip()
        if not raw:
            raise HTTPException(400, "Thiếu tuổi hoặc nhóm tuổi")
        try:
            group = age_years_to_group(float(raw))
        except ValueError as e:
            raise HTTPException(400, f"Tuổi không hợp lệ: {raw}") from e
    blob = await _read_upload(file)
    PRIVATE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cid = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    ext = ".wav"  # frontend luôn gửi WAV PCM 16 kHz sau decodeToWav
    (PRIVATE_RAW_DIR / f"{cid}{ext}").write_bytes(blob)
    meta = {
        "clip_id": cid, "recorded_at": datetime.now().isoformat(timespec="seconds"), "age": age, "age_group": group,
        "gender": gender, "locale": locale, "device": device, "environment": environment,
        "speaker_id": speaker_id or f"anon_{uuid.uuid4().hex[:8]}", "sentence": sentence, "consent": True,
    }
    (PRIVATE_RAW_DIR / f"{cid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    con = db()
    con.execute("INSERT INTO contributions VALUES (?,?,?,?,?,?,?,?)", (cid, meta["recorded_at"], group, gender, locale, device, environment, f"{cid}{ext}"))
    con.commit(); con.close()
    return {"ok": True, "clip_id": cid, "age_group": group}


@app.get("/api/collect/stats")
def collect_stats():
    con = db()
    rows = con.execute("SELECT age_group, COUNT(*) FROM contributions GROUP BY age_group").fetchall()
    con.close()
    return {"total": sum(r[1] for r in rows), "by_group": {g: 0 for g in AGE_GROUPS} | {r[0]: r[1] for r in rows}}


# ----------------------------------------------------------------------------- UI
FRONTEND = APP_DIR / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/collect")
def collect_page():
    return FileResponse(FRONTEND / "collect.html")


@app.exception_handler(404)
async def not_found(_, __):
    return JSONResponse({"detail": "Not found"}, status_code=404)
