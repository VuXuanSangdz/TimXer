"""
TimeXer cho dự báo nhu cầu gọi xe với biến ngoại sinh.

- X_en: RevIN + patch self-attention
- X_ex: global-scaled trước (per-channel) + variate cross-attention
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.layers import EnEmbedding, ExEmbedding, EncoderLayer, FlattenHead
from src.models.revin import RevIN


class TimeXer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_exog: int,
        n_continuous: int | None = None,
        flag_embed_boost: float = 1.0,
        patch_len: int = 4,
        patch_stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin_en: bool = True,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_exog = n_exog
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = (seq_len - patch_len) // patch_stride + 1
        self.use_revin_en = use_revin_en

        self.revin = RevIN(1) if use_revin_en else None
        self.en_embedding = EnEmbedding(1, d_model, patch_len, patch_stride, dropout)
        self.ex_embedding = ExEmbedding(
            seq_len, d_model, dropout,
            n_continuous=n_continuous,
            flag_embed_boost=flag_embed_boost,
        )
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_nf = d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, pred_len, dropout)

    def forward(self, x_en: torch.Tensor, x_ex: torch.Tensor) -> torch.Tensor:
        """
        x_en: [B, L, 1] — trip_count (raw, RevIN bên trong)
        x_ex: [B, L, C] — đã global-scaled per channel
        Returns: [B, T, 1] dự báo trip_count (scale gốc sau denorm)
        """
        if self.revin is not None:
            x_en = self.revin.norm(x_en)

        # [B, 1, L]
        en_in = x_en.permute(0, 2, 1)
        en_embed, n_vars = self.en_embedding(en_in)
        ex_embed = self.ex_embedding(x_ex)

        enc = en_embed
        for layer in self.encoder_layers:
            enc = layer(enc, ex_embed)
        enc = self.norm(enc)

        enc = enc.reshape(-1, n_vars, enc.shape[-2], enc.shape[-1])
        enc = enc.permute(0, 1, 3, 2)  # [B, 1, d_model, patch_num+1]
        dec = self.head(enc).permute(0, 2, 1)  # [B, T, 1]

        if self.revin is not None:
            dec = self.revin.denorm(dec)
        return dec
