"""Hằng số dùng chung cho toàn dự án TimeXer-DINOv3 Voice Age."""

from __future__ import annotations

# 5 nhóm tuổi hiển thị trên UI (slide 3)
AGE_GROUPS: list[str] = ["<18", "18-29", "30-49", "50-64", "65+"]
AGE_GROUP_LABELS_VI: dict[str, str] = {
    "<18": "Dưới 18 tuổi",
    "18-29": "18 – 29 tuổi",
    "30-49": "30 – 49 tuổi",
    "50-64": "50 – 64 tuổi",
    "65+": "Từ 65 tuổi trở lên",
}
AGE_GROUP_TO_ID: dict[str, int] = {g: i for i, g in enumerate(AGE_GROUPS)}

# Ánh xạ nhãn thập niên của Common Voice -> 5 nhóm tuổi của hệ thống.
# Lưu ý xấp xỉ: "teens" (13-19) -> "<18"; "sixties" (60-69) -> "50-64".
COMMON_VOICE_AGE_MAP: dict[str, str] = {
    "teens": "<18",
    "twenties": "18-29",
    "thirties": "30-49",
    "fourties": "30-49",
    "fifties": "50-64",
    "sixties": "50-64",
    "seventies": "65+",
    "eighties": "65+",
    "nineties": "65+",
}


def age_years_to_group(age: float | int) -> str:
    """Tuổi (năm) -> nhóm tuổi. Dùng khi người dùng tự khai tuổi trong app thu thập."""
    age = float(age)
    if age < 18:
        return "<18"
    if age < 30:
        return "18-29"
    if age < 50:
        return "30-49"
    if age < 65:
        return "50-64"
    return "65+"


# Âm thanh chuẩn hoá
SAMPLE_RATE = 16_000
CLIP_SECONDS = 5.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320  # 20 ms @ 16 kHz -> 251 khung / 5 s
N_MFCC = 20
F_MIN = 20.0
F_MAX = 8000.0

# Kích thước ảnh phổ đưa vào DINOv3 (ViT-S/16: 224/16 = 14 patch mỗi chiều)
DINO_IMAGE_SIZE = 224
DINO_PATCH = 16

# Biến ngoại sinh (slide 2: thời gian ngày giờ + ngữ cảnh ghi âm) + tín hiệu hỗ trợ theo khung
EXOG_STATIC_COLS: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "duration_norm",
    "snr_db_norm",
    "rms_db_norm",
    "source_mic",
    "orig_sr_norm",
]
EXOG_FRAME_COLS: list[str] = [
    "f0_contour",
    "energy_contour",
    "zcr_contour",
    "centroid_contour",
]
EXOG_COLS: list[str] = EXOG_STATIC_COLS + EXOG_FRAME_COLS

# Thống kê giọng nói (phần "MFCC / Pitch / Formant / Energy" của biến nội sinh y ở slide 2) — gom thành
# 1 "prosody token" nội sinh đặt cạnh các token DINOv3. Thứ tự cố định để dùng chung train/export/app.
PROSODY_COLS: list[str] = (
    ["f0_mean", "f0_std", "f0_median", "voiced_ratio", "energy_mean", "energy_std", "zcr_mean", "centroid_mean"]
    + ["formant_f1", "formant_f2", "formant_f3"]
    + [f"mfcc{i:02d}_{s}" for i in range(N_MFCC) for s in ("mean", "std")]
)
