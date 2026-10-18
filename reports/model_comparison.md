| Mô hình | Acc | Bal-Acc | Macro-F1 | Ordinal MAE | Within ±1 | Params học |
|---|---|---|---|---|---|---|
| LogReg (MFCC/F0/formant stats) | 34.9% | 32.1% | **0.313** | 1.050 | 72.9% | - |
| HistGradientBoosting (scalar feats) | 40.6% | 33.2% | **0.328** | 0.910 | 76.6% | - |
| TimeXer gốc + prosody token | 36.3% | 32.7% | **0.321** | 1.073 | 69.8% | 0.75M |
| TimeXer gốc (linear patch embedding) | 30.0% | 26.9% | **0.251** | 1.073 | 73.3% | 0.73M |
| TimeXer + DINOv3 ViT-S/16, temporal_concat + prosody token (đề xuất) | 39.0% | 33.6% | **0.335** | 0.944 | 75.5% | 1.22M |
| TimeXer + DINOv3, token grid 196 (ablation, không prosody) | 30.5% | 34.8% | **0.282** | 1.112 | 71.3% | 0.56M |
| TimeXer + DINOv3 ViT-S/16, temporal_concat (không prosody) | 35.5% | 33.3% | **0.307** | 1.039 | 71.8% | 1.20M |
| TimeXer + DINOv3, token temporal mean (ablation, không prosody) | 30.9% | 35.6% | **0.293** | 1.131 | 70.0% | 0.56M |
