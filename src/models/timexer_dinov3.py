"""
TimeXer-DINOv3 cho bài toán phân loại nhóm tuổi từ giọng nói.

So với src/models/timexer_original.py (TimeXer gốc, dự báo chuỗi):
  [THAY]  (a) EnEmbedding          -> DINOv3Encoder (src/models/dinov3_encoder.py)
  [GIỮ]   (b) ExEmbedding           (variate token cho biến ngoại sinh)
  [GIỮ]   (c)(d) EncoderLayer       (Self-Attention + Cross-Attention + FFN, LayerNorm)
  [THAY]  FlattenHead (hồi quy T bước) -> ClassificationHead (5 nhóm tuổi)
  [BỎ]    RevIN (không cần cho ảnh phổ đã chuẩn hoá min-max)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.dinov3_encoder import DINOv3Encoder, MelPatchEmbedding
from src.models.layers import EncoderLayer, ExEmbedding


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        glb = enc[:, -1, :]  # global token (vị trí cuối, như TimeXer gốc)
        mean_patch = enc[:, :-1, :].mean(dim=1)
        return self.net(torch.cat([glb, mean_patch], dim=-1))


class ProsodyTokenEmbedding(nn.Module):
    """
    Thống kê giọng nói (MFCC mean/std, F0, formant, energy, ... — PROSODY_COLS, đã z-score) -> 1 token nội sinh.
    Đây là phần "MFCC / Pitch / Formant / Energy" của biến nội sinh y trong thiết kế; token này đứng cạnh các
    token thời gian của DINOv3 và tham gia self-attention như một patch "tổng hợp".
    """

    def __init__(self, n_prosody: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_prosody, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, prosody: torch.Tensor) -> torch.Tensor:  # [B, P] -> [B, 1, d_model]
        return self.net(prosody.float()).unsqueeze(1) + self.type_embedding


class TimeXerDINOv3(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_exog: int,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        dino_model: str = "vit_small_patch16_dinov3.lvd1689m",
        dino_pretrained: bool = True,
        dino_img_size: int = 224,
        dino_freeze: bool = True,
        dino_token_mode: str = "temporal_concat",
        dino_backbone: nn.Module | None = None,
        n_prosody: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_exog = n_exog
        self.n_classes = n_classes
        self.n_prosody = n_prosody
        self.prosody_embedding = ProsodyTokenEmbedding(n_prosody, d_model, dropout) if n_prosody > 0 else None

        # (a) >>> THAY EnEmbedding bằng DINOv3 <<<
        self.en_encoder = DINOv3Encoder(
            d_model=d_model,
            model_name=dino_model,
            pretrained=dino_pretrained,
            img_size=dino_img_size,
            freeze_backbone=dino_freeze,
            token_mode=dino_token_mode,
            dropout=dropout,
            backbone=dino_backbone,
        )
        # (b) giữ nguyên
        self.ex_embedding = ExEmbedding(seq_len, d_model, dropout)
        # (c)(d) giữ nguyên
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassificationHead(d_model, n_classes, dropout)

    # --- đường đi chung sau khi có token nội sinh -------------------------------------
    def _encode(self, tokens: torch.Tensor, glb: torch.Tensor, x_ex: torch.Tensor, prosody: torch.Tensor | None) -> torch.Tensor:
        parts = [tokens]
        if self.prosody_embedding is not None:
            if prosody is None:
                raise ValueError("Model được cấu hình với prosody token nhưng không nhận được `prosody`.")
            parts.append(self.prosody_embedding(prosody))
        parts.append(glb)  # global token ở cuối (EncoderLayer dùng x[:, -1])
        enc = torch.cat(parts, dim=1)
        ex = self.ex_embedding(x_ex)  # [B, C, d_model] — mỗi biến ngoại sinh = 1 token
        for layer in self.encoder_layers:
            enc = layer(enc, ex)
        enc = self.norm(enc)
        return self.head(enc)

    def forward(self, mel: torch.Tensor, x_ex: torch.Tensor, prosody: torch.Tensor | None = None) -> torch.Tensor:
        """
        mel    : [B, n_mels, T]  log-Mel (biến nội sinh y)
        x_ex   : [B, T, C]       biến ngoại sinh x (thời gian, ngữ cảnh ghi âm, tín hiệu hỗ trợ)
        prosody: [B, P]          thống kê giọng nói đã z-score (tuỳ chọn, PROSODY_COLS)
        -> logits [B, n_classes]
        """
        tokens, glb = self.en_encoder(mel)
        return self._encode(tokens, glb, x_ex, prosody)

    def forward_from_cache(self, raw_tokens: torch.Tensor, x_ex: torch.Tensor, prosody: torch.Tensor | None = None) -> torch.Tensor:
        """Huấn luyện nhanh với backbone đóng băng: raw_tokens = en_encoder.encode_backbone(mel) đã cache."""
        tokens, glb = self.en_encoder.project(raw_tokens)
        return self._encode(tokens, glb, x_ex, prosody)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class TimeXerVoiceBaseline(nn.Module):
    """Ablation: cùng kiến trúc nhưng khối (a) là patch embedding tuyến tính kiểu TimeXer gốc."""

    def __init__(
        self,
        seq_len: int,
        n_exog: int,
        n_mels: int = 128,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        patch_len: int = 16,
        n_prosody: int = 0,
    ):
        super().__init__()
        self.en_encoder = MelPatchEmbedding(n_mels, seq_len, d_model, patch_len, dropout)
        self.prosody_embedding = ProsodyTokenEmbedding(n_prosody, d_model, dropout) if n_prosody > 0 else None
        self.ex_embedding = ExEmbedding(seq_len, d_model, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassificationHead(d_model, n_classes, dropout)

    def forward(self, mel: torch.Tensor, x_ex: torch.Tensor, prosody: torch.Tensor | None = None) -> torch.Tensor:
        tokens, glb = self.en_encoder(mel)
        parts = [tokens]
        if self.prosody_embedding is not None:
            parts.append(self.prosody_embedding(prosody))
        enc = torch.cat(parts + [glb], dim=1)
        ex = self.ex_embedding(x_ex)
        for layer in self.encoder_layers:
            enc = layer(enc, ex)
        return self.head(self.norm(enc))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
