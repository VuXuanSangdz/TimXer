# Deploy TimeXer AI (Dự đoán độ tuổi từ giọng nói)

Ứng dụng là một container Docker duy nhất (FastAPI + UI tĩnh + ONNX Runtime), cổng **7860**.

## A. Hugging Face Spaces (Docker SDK) — khuyến nghị, miễn phí

1. Xuất model: `python scripts/07_export_model.py` → `app/model/{timexer_dinov3.onnx, timexer_dinov3_head.pt, model_card.json}`.
2. Tạo Space mới: https://huggingface.co/new-space → SDK **Docker**, hardware CPU basic (2 vCPU, 16 GB).
3. Đẩy code lên Space (Space là một git repo; file > 10 MB cần Git LFS):

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                                    # token write
hf repo create timexer-ai-voice-age --type space --space_sdk docker   # tên tuỳ ý

git clone https://huggingface.co/spaces/<hf_username>/timexer-ai-voice-age space && cd space
# copy phần cần thiết của repo
robocopy ..\src src /E ; robocopy ..\configs configs /E ; robocopy ..\app app /E   # (Windows)  — Linux/mac: cp -r ../src ../configs ../app .
copy ..\app\Dockerfile Dockerfile
git lfs install && git lfs track "*.onnx" "*.pt" && git add .gitattributes
git add . && git commit -m "TimeXer-DINOv3 voice age app" && git push
```

4. Thêm vào đầu `README.md` của Space (metadata bắt buộc):

```yaml
---
title: TimeXer AI – Voice Age
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

5. Space build ~5–8 phút → link dạng `https://huggingface.co/spaces/<hf_username>/timexer-ai-voice-age`
   (hoặc `https://<hf_username>-timexer-ai-voice-age.hf.space`). Trang thu thập dữ liệu: `/collect`.

> Nếu không đưa file ONNX lên (tiết kiệm LFS), app tự dựng model PyTorch: tải backbone DINOv3 từ `timm` lúc khởi
> động (cần `torch`, `timm` trong `requirements-deploy.txt`) và nạp `timexer_dinov3_head.pt` (~7 MB, đã có trong
> repo; chứa cả mean/std chuẩn hoá prosody).

## B. Render / Railway / Fly.io

* Render: *New → Web Service → Docker*, Dockerfile path `app/Dockerfile`, port 7860, plan Free/Starter.
* Fly.io: `fly launch --dockerfile app/Dockerfile --internal-port 7860`.
* Railway: deploy từ GitHub repo, đặt `Dockerfile path = app/Dockerfile`.

## C. VPS / máy nội bộ

```bash
docker build -f app/Dockerfile -t timexer-ai .
docker run -d --name timexer-ai -p 7860:7860 -v $PWD/app_data:/tmp/app_data -v $PWD/private_raw:/tmp/private_raw timexer-ai
```

Biến môi trường: `APP_DATA_DIR` (SQLite lịch sử), `PRIVATE_RAW_DIR` (bản ghi đóng góp), `HF_HOME` (cache timm).

## Kiểm tra sau deploy

```bash
curl https://<host>/api/health
curl -F "file=@sample.wav" -F "source=upload" https://<host>/api/predict
```
