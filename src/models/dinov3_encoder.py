"""
>>> ĐIỂM THAY ĐỔI CHÍNH SO VỚI TimeXer GỐC <<<

Khối (a) "Endogenous Series Embedding" của TimeXer gốc (src/models/layers.py::EnEmbedding):
    chuỗi nội sinh -> cắt patch không chồng lấn -> Linear(patch_len -> d_model)
    + 1 global token khởi tạo ngẫu nhiên.

Ở đây khối (a) được thay bằng DINOv3 (Meta, 2025) — Vision Transformer tự giám sát:
    log-Mel spectrogram [n_mels, T] -> ảnh 3 kênh 224x224 -> DINOv3 ViT-S/16
      -> 14x14 patch token (mỗi token = 1 "patch" thời gian-tần số, tương đương
         "non-overlap patches" của TimeXer) -> gộp theo trục tần số => 14 token thời gian
      -> CLS token của DINOv3 đóng vai trò GLOBAL TOKEN (thay cho token khởi tạo ngẫu nhiên)
      -> Linear(384 -> d_model) + positional embedding.

Các khối (b) ExEmbedding, (c) Self-Attention, (d) Cross-Attention, FFN giữ NGUYÊN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import PositionalEmbedding

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV3_MODELS = {
    # timm name                                    : embed_dim
    "vit_small_patch16_dinov3.lvd1689m": 384,
    "vit_small_plus_patch16_dinov3.lvd1689m": 384,
    "vit_base_patch16_dinov3.lvd1689m": 768,
    "vit_large_patch16_dinov3.lvd1689m": 1024,
}


class MelToImage(nn.Module):
    """log-Mel [B, n_mels, T] -> ảnh [B, 3, H, W] chuẩn ImageNet (nằm trong graph để export ONNX)."""

    def __init__(self, img_size: int | tuple[int, int] = 224):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.float()
        mn = mel.amin(dim=(1, 2), keepdim=True)
        mx = mel.amax(dim=(1, 2), keepdim=True)
        x = (mel - mn) / (mx - mn + 1e-6)  # [0,1]
        x = torch.flip(x, dims=[1])  # tần số thấp ở dưới như ảnh phổ thông thường
        x = x.unsqueeze(1)  # [B,1,n_mels,T]
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std


class DINOv3Encoder(nn.Module):
    """
    Thay thế EnEmbedding. Trả về (patch_tokens [B, N, d_model], global_token [B, 1, d_model]).

    token_mode:
      - "temporal"        : gộp trung bình theo trục tần số -> N = W (14) token thời gian (rẻ nhất)
      - "temporal_concat" : nối H (14) token tần số của cùng cột thời gian -> N = W token, mỗi token
                            H*embed_dim chiều -> Linear về d_model (giữ cấu trúc tần số; mặc định)
      - "grid"            : giữ toàn bộ N = H*W (196) token thời gian-tần số

    encode_backbone() luôn trả về lưới đầy đủ [B, 1+H*W, D] (CLS + patches) để một cache dùng được
    cho mọi token_mode; việc gộp token diễn ra trong project().
    """

    def __init__(
        self,
        d_model: int = 128,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        pretrained: bool = True,
        img_size: int | tuple[int, int] = 224,
        freeze_backbone: bool = True,
        token_mode: str = "temporal_concat",
        dropout: float = 0.1,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        import timm

        self.model_name = model_name
        self.token_mode = token_mode
        self.to_image = MelToImage(img_size)
        if backbone is None:
            backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.backbone = backbone
        self.embed_dim = int(getattr(backbone, "embed_dim", DINOV3_MODELS.get(model_name, 384)))
        self.n_prefix = int(getattr(backbone, "num_prefix_tokens", 1))
        self.patch = int(backbone.patch_embed.patch_size[0])
        H = self.to_image.img_size[0] // self.patch
        W = self.to_image.img_size[1] // self.patch
        self.grid = (H, W)
        if token_mode not in ("temporal", "temporal_concat", "grid"):
            raise ValueError(token_mode)
        self.n_tokens = H * W if token_mode == "grid" else W
        token_dim = self.embed_dim * H if token_mode == "temporal_concat" else self.embed_dim

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        self.proj = nn.Linear(token_dim, d_model)
        self.proj_glb = nn.Linear(self.embed_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # giữ BN/LN thống kê & tắt dropout của backbone
        return self

    # ---- 1) chạy backbone: mel -> token thô (chưa chiếu về d_model) --------------------
    @torch.no_grad()
    def _frozen_backbone(self, img: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(img)

    def encode_backbone(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, n_mels, T] -> raw tokens [B, 1 + H*W, embed_dim]  (token 0 = CLS, sau đó lưới H×W)."""
        img = self.to_image(mel)
        feats = self._frozen_backbone(img) if self.freeze_backbone else self.backbone.forward_features(img)
        cls = feats[:, :1]
        patches = feats[:, self.n_prefix :]  # bỏ CLS + register tokens
        return torch.cat([cls, patches], dim=1)

    def pool_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        """[B, H*W, D] -> token nội sinh theo token_mode."""
        B, _, D = patches.shape
        H, W = self.grid
        if self.token_mode == "grid":
            return patches
        g = patches.reshape(B, H, W, D)
        if self.token_mode == "temporal":
            return g.mean(dim=1)  # [B, W, D]
        return g.permute(0, 2, 1, 3).reshape(B, W, H * D)  # temporal_concat: [B, W, H*D]

    # ---- 2) chiếu token thô về d_model (phần học được) --------------------------------
    def project(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls, patches = raw[:, :1].float(), raw[:, 1:].float()
        tokens = self.proj(self.pool_tokens(patches))
        tokens = tokens + self.position_embedding(tokens)
        glb = self.proj_glb(cls)
        return self.dropout(tokens), self.dropout(glb)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.project(self.encode_backbone(mel))


class MelPatchEmbedding(nn.Module):
    """
    Khối (a) THEO KIỂU TimeXer GỐC dùng cho ablation/baseline: cắt log-Mel thành patch thời gian
    không chồng lấn, Linear(patch_len * n_mels -> d_model) + global token khởi tạo ngẫu nhiên.
    """

    def __init__(self, n_mels: int, seq_len: int, d_model: int, patch_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.n_tokens = seq_len // patch_len
        self.value_embedding = nn.Linear(patch_len * n_mels, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = mel.float()
        mn = mel.amin(dim=(1, 2), keepdim=True)
        mx = mel.amax(dim=(1, 2), keepdim=True)
        mel = (mel - mn) / (mx - mn + 1e-6)
        B, M, T = mel.shape
        T_use = self.n_tokens * self.patch_len
        x = mel[:, :, :T_use].reshape(B, M, self.n_tokens, self.patch_len)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_tokens, M * self.patch_len)
        x = self.value_embedding(x)
        x = x + self.position_embedding(x)
        glb = self.glb_token.expand(B, -1, -1)
        return self.dropout(x), self.dropout(glb)
