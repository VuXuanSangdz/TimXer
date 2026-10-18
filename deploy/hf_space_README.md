---
title: TimeXer AI – Voice Age
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TimeXer AI — Dự đoán độ tuổi từ giọng nói

TimeXer (NeurIPS 2024) với encoder nội sinh **DINOv3 ViT-S/16** + prosody token.
Ghi âm / tải file → xác suất 5 nhóm tuổi. Trang thu thập dữ liệu: [`/collect`](/collect).

API: `GET /api/health` · `POST /api/predict` · `POST /api/collect`
