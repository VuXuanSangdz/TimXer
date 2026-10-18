"""Dataset PyTorch đọc data/processed/* (mel, exog, nhãn) + cache token DINOv3 (tuỳ chọn)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.common.constants import AGE_GROUPS, PROSODY_COLS


@dataclass
class VoiceAgeArrays:
    meta: pd.DataFrame
    mel: np.ndarray  # [N, n_mels, T] float16 (memmap)
    exog: np.ndarray  # [N, C, T] float16 (memmap)
    dino_tokens: np.ndarray | None = None  # [N, 1+Ntok, D] float16
    prosody: np.ndarray | None = None  # [N, P] float32 — đã chuẩn hoá z-score theo tập train
    prosody_mean: np.ndarray | None = None
    prosody_std: np.ndarray | None = None

    @classmethod
    def load(cls, processed_dir: str | Path, dino_cache: str | Path | None = None) -> "VoiceAgeArrays":
        d = Path(processed_dir)
        meta = pd.read_csv(d / "metadata.csv", dtype={"clip_id": str, "speaker_id": str})
        mel = np.load(d / "mel_logspec.npy", mmap_mode="r")
        exog = np.load(d / "exog_series.npy", mmap_mode="r")
        tok = np.load(dino_cache, mmap_mode="r") if dino_cache and Path(dino_cache).exists() else None
        obj = cls(meta, mel, exog, tok)
        scal_path = d / "features_scalar.csv"
        if scal_path.exists():
            scal = pd.read_csv(scal_path, dtype={"clip_id": str}).set_index("clip_id")
            raw = scal.reindex(meta["clip_id"])[PROSODY_COLS].to_numpy(dtype=np.float32)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            tr = (meta["split"] == "train").to_numpy()
            mu = raw[tr].mean(0) if tr.any() else raw.mean(0)
            sd = raw[tr].std(0) if tr.any() else raw.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            obj.prosody_mean, obj.prosody_std = mu.astype(np.float32), sd.astype(np.float32)
            obj.prosody = np.clip((raw - mu) / sd, -6, 6).astype(np.float32)
        return obj

    @property
    def n_prosody(self) -> int:
        return 0 if self.prosody is None else int(self.prosody.shape[1])

    def indices(self, split: str) -> np.ndarray:
        return self.meta.index[self.meta["split"] == split].to_numpy()

    @property
    def seq_len(self) -> int:
        return int(self.mel.shape[-1])

    @property
    def n_exog(self) -> int:
        return int(self.exog.shape[1])

    def class_weights(self, split: str = "train", power: float = 0.5) -> torch.Tensor:
        counts = self.meta.loc[self.indices(split), "label"].value_counts().reindex(range(len(AGE_GROUPS))).fillna(0).to_numpy()
        w = (counts.sum() / np.maximum(counts, 1)) ** power
        return torch.tensor(w / w.mean(), dtype=torch.float32)


class SpecAugment:
    """Che ngẫu nhiên dải tần/khung thời gian trên log-Mel (chỉ dùng khi backbone KHÔNG đóng băng)."""

    def __init__(self, freq_mask: int = 16, time_mask: int = 30, n_masks: int = 2):
        self.freq_mask, self.time_mask, self.n_masks = freq_mask, time_mask, n_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        M, T = mel.shape
        fill = mel.min()
        for _ in range(self.n_masks):
            f = int(torch.randint(0, self.freq_mask + 1, (1,)))
            f0 = int(torch.randint(0, max(1, M - f), (1,)))
            mel[f0 : f0 + f, :] = fill
            t = int(torch.randint(0, self.time_mask + 1, (1,)))
            t0 = int(torch.randint(0, max(1, T - t), (1,)))
            mel[:, t0 : t0 + t] = fill
        return mel


class VoiceAgeDataset(Dataset):
    def __init__(self, arrays: VoiceAgeArrays, indices: np.ndarray, use_cache: bool, train: bool = False, token_noise: float = 0.0):
        self.a = arrays
        self.idx = np.asarray(indices)
        self.use_cache = use_cache and arrays.dino_tokens is not None
        self.train = train
        self.aug = SpecAugment() if train and not self.use_cache else None
        self.token_noise = token_noise if train else 0.0

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = int(self.idx[i])
        x_ex = torch.from_numpy(np.asarray(self.a.exog[j], dtype=np.float32)).T  # [T, C]
        label = torch.tensor(int(self.a.meta.at[j, "label"]), dtype=torch.long)
        item = {"x_ex": x_ex, "label": label, "index": torch.tensor(j)}
        if self.a.prosody is not None:
            item["prosody"] = torch.from_numpy(self.a.prosody[j])
        if self.use_cache:
            tok = torch.from_numpy(np.asarray(self.a.dino_tokens[j], dtype=np.float32))
            if self.token_noise > 0:
                tok = tok + self.token_noise * torch.randn_like(tok) * tok.std()
            item["raw_tokens"] = tok
        else:
            mel = torch.from_numpy(np.asarray(self.a.mel[j], dtype=np.float32))
            if self.aug is not None:
                mel = self.aug(mel)
            item["mel"] = mel
        return item


def make_loaders(
    arrays: VoiceAgeArrays,
    batch_size: int,
    use_cache: bool,
    balanced_sampling: bool = True,
    num_workers: int = 0,
    token_noise: float = 0.0,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        idx = arrays.indices(split)
        ds = VoiceAgeDataset(arrays, idx, use_cache, train=(split == "train"), token_noise=token_noise)
        sampler = None
        shuffle = split == "train"
        if split == "train" and balanced_sampling and len(idx):
            labels = arrays.meta.loc[idx, "label"].to_numpy()
            counts = np.bincount(labels, minlength=len(AGE_GROUPS)).astype(np.float64)
            w = 1.0 / np.sqrt(np.maximum(counts[labels], 1))
            sampler = WeightedRandomSampler(torch.tensor(w), num_samples=len(idx), replacement=True)
            shuffle = False
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, drop_last=False)
    return loaders
