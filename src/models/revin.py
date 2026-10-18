"""Reversible Instance Normalization — chỉ cho biến nội sinh."""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - self._mean) / self._stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._stdev is None:
            raise RuntimeError("Call norm() before denorm()")
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self._stdev + self._mean
