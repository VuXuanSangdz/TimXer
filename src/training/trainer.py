"""Trainer phân loại: AdamW + cosine LR + early stopping theo macro-F1 trên tập val."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.metrics import classification_metrics


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, model: nn.Module, cfg: dict, device: torch.device, checkpoint_dir: Path, class_weights: torch.Tensor | None = None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        t = cfg["train"]
        params = model.trainable_parameters() if hasattr(model, "trainable_parameters") else model.parameters()
        self.opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.label_smoothing = t.get("label_smoothing", 0.0)
        self.patience = t.get("patience", 10)
        self.min_epochs = t.get("min_epochs", 5)
        self.grad_clip = t.get("grad_clip", 1.0)
        self.best_state = None
        self.best_score = -math.inf
        self.best_epoch = -1
        self.history: list[dict] = []
        self.extra: dict = {}  # metadata lưu kèm checkpoint (vd. mean/std prosody, seq_len, n_exog)

    # ------------------------------------------------------------------
    def _forward(self, batch: dict) -> torch.Tensor:
        x_ex = batch["x_ex"].to(self.device)
        uses_prosody = getattr(self.model, "prosody_embedding", None) is not None
        prosody = batch["prosody"].to(self.device) if (uses_prosody and "prosody" in batch) else None
        if "raw_tokens" in batch:
            return self.model.forward_from_cache(batch["raw_tokens"].to(self.device), x_ex, prosody)
        return self.model(batch["mel"].to(self.device), x_ex, prosody)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y, weight=self.class_weights, label_smoothing=self.label_smoothing)

    def train_epoch(self, loader) -> float:
        self.model.train()
        tot, n = 0.0, 0
        for batch in loader:
            y = batch["label"].to(self.device)
            logits = self._forward(batch)
            loss = self._loss(logits, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        return tot / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader, collect: bool = False) -> dict:
        self.model.eval()
        ys, ps, probs, idxs = [], [], [], []
        tot, n = 0.0, 0
        for batch in loader:
            y = batch["label"].to(self.device)
            logits = self._forward(batch)
            tot += self._loss(logits, y).item() * len(y)
            n += len(y)
            pr = torch.softmax(logits.float(), dim=-1)
            ys.append(y.cpu().numpy())
            ps.append(pr.argmax(-1).cpu().numpy())
            probs.append(pr.cpu().numpy())
            idxs.append(batch["index"].numpy())
        y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
        y_pred = np.concatenate(ps) if ps else np.array([], dtype=int)
        pr = np.concatenate(probs) if probs else np.zeros((0, 5))
        m = classification_metrics(y_true, y_pred, pr)
        m["loss"] = tot / max(n, 1)
        if collect:
            m["_y_true"], m["_y_pred"], m["_probs"], m["_index"] = y_true, y_pred, pr, np.concatenate(idxs)
        return m

    def fit(self, train_loader, val_loader, epochs: int, log=print) -> dict:
        steps = epochs * max(1, len(train_loader))
        warm = int(0.05 * steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))),
        )
        bad = 0
        t0 = time.time()
        for ep in range(1, epochs + 1):
            tl = self.train_epoch(train_loader)
            vm = self.evaluate(val_loader)
            score = vm["macro_f1"] - 0.1 * vm["ordinal_mae"]  # ưu tiên F1, phạt nhầm xa
            rec = {"epoch": ep, "train_loss": tl, "val_loss": vm["loss"], "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "val_ordinal_mae": vm["ordinal_mae"], "lr": self.opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
            self.history.append(rec)
            improved = score > self.best_score
            if improved:
                self.best_score, self.best_epoch, bad = score, ep, 0
                self.best_state = copy.deepcopy({k: v.detach().cpu() for k, v in self.model.state_dict().items() if not k.startswith("en_encoder.backbone.")})
                torch.save({"state_dict": self.best_state, "epoch": ep, "cfg": self.cfg, **self.extra}, self.checkpoint_dir / "best_model.pt")
            else:
                bad += 1
            log(f"ep {ep:3d} | train {tl:.4f} | val loss {vm['loss']:.4f} acc {vm['accuracy']*100:5.1f}% F1 {vm['macro_f1']:.3f} MAE {vm['ordinal_mae']:.3f} {'*' if improved else ''}")
            if ep >= self.min_epochs and bad >= self.patience:
                log(f"Early stop tại epoch {ep} (best {self.best_epoch})")
                break
        (self.checkpoint_dir / "history.json").write_text(json.dumps(self.history, indent=1), encoding="utf-8")
        return {"best_epoch": self.best_epoch, "best_score": self.best_score, "epochs_run": len(self.history)}

    def load_best(self) -> int:
        ck = torch.load(self.checkpoint_dir / "best_model.pt", map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ck["state_dict"], strict=False)
        bad = [k for k in missing if not k.startswith("en_encoder.backbone.")]
        assert not bad and not unexpected, (bad, unexpected)
        return int(ck["epoch"])
