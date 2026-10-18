from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool = False, attention_dropout: float = 0.1):
        super().__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        scale = 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        self.n_heads = n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, _ = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), None


class EnEmbedding(nn.Module):
    """Patch-wise embedding cho chuỗi nội sinh + global token."""

    def __init__(self, n_vars: int, d_model: int, patch_len: int, patch_stride: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, n_vars, L]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat(x.shape[0], 1, 1, 1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.reshape(-1, n_vars, x.shape[-2], x.shape[-1])
        x = torch.cat([x, glb], dim=2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return self.dropout(x), n_vars


class ExEmbedding(nn.Module):
    """Inverted embedding — mỗi biến ngoại sinh = 1 token (variate-level)."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        dropout: float,
        n_continuous: int | None = None,
        flag_embed_boost: float = 1.0,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.flag_scale = (
            nn.Parameter(torch.tensor(flag_embed_boost, dtype=torch.float32))
            if n_continuous is not None and flag_embed_boost != 1.0
            else None
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> [B, C, d_model]
        emb = self.value_embedding(x.permute(0, 2, 1))
        if self.flag_scale is not None and self.n_continuous is not None:
            nc = self.n_continuous
            if nc < emb.shape[1]:
                emb = torch.cat([emb[:, :nc, :], emb[:, nc:, :] * self.flag_scale], dim=1)
        return self.dropout(emb)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout), d_model, n_heads
        )
        self.cross_attention = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout), d_model, n_heads
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        B, _, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = x_glb_ori.reshape(B, -1, D)
        x_glb_attn = self.dropout(self.cross_attention(x_glb, cross, cross)[0])
        x_glb_attn = x_glb_attn.reshape(x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2]).unsqueeze(1)
        x_glb = self.norm2(x_glb_ori + x_glb_attn)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class FlattenHead(nn.Module):
    def __init__(self, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.flatten(x)))
