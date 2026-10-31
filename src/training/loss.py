"""
Hàm mất mát sau khi thay TimeXer (dự báo, MSE) bằng TimeXer-DINOv3 (phân loại 5 nhóm tuổi).

TimeXer gốc tối ưu
    L_MSE = (1/T) * sum_t || yhat_t - y_t ||^2
trên head hồi quy. DINOv3 thay EnEmbedding và head thành ClassificationHead, backbone đóng băng,
nên không còn L_MSE và cũng không tối ưu loss tự giám sát của DINOv3. Hàm học là
cross-entropy có làm trơn nhãn, nhân trọng số lớp:

    p = softmax(z),  z in R^K, K = 5
    q_k = (1 - eps) * 1[k = y] + eps / K
    L_i = - w_{y_i} * sum_k q_k * log p_k
    L   = mean_i L_i

    w_c = (N / n_c)^alpha / mean_j (N / n_j)^alpha,   alpha = 0.5
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int = 5, label_smoothing: float = 0.05, weight: torch.Tensor | None = None):
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing phải nằm trong [0, 1)")
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        nll = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=-1)
        eps = self.label_smoothing
        loss = (1.0 - eps) * nll + eps * smooth
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()
