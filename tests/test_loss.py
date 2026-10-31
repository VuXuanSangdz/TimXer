"""Đối chiếu WeightedLabelSmoothingCE với công thức viết ở README §1."""

import torch

from src.training.loss import WeightedLabelSmoothingCE


def test_matches_closed_form():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    weight = torch.tensor([1.2, 0.8, 1.0, 0.9, 1.6])
    eps = 0.05
    got = WeightedLabelSmoothingCE(5, eps, weight)(logits, target)

    log_p = torch.log_softmax(logits, dim=-1)
    q = torch.full_like(log_p, eps / 5)
    q[torch.arange(8), target] += 1 - eps
    per = -(weight[target] * (q * log_p).sum(-1))
    assert torch.allclose(got, per.mean())


def test_zero_smoothing_is_weighted_nll():
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0, 0.0]])
    target = torch.tensor([0, 2])
    weight = torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0])
    got = WeightedLabelSmoothingCE(5, 0.0, weight)(logits, target)
    log_p = torch.log_softmax(logits, dim=-1)
    ref = (-weight[target] * log_p[torch.arange(2), target]).mean()
    assert torch.allclose(got, ref)
