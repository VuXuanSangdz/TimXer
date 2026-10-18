#!/usr/bin/env python3
"""Tạo Space Docker trên Hugging Face và upload app + ONNX."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login, whoami

ROOT = Path(__file__).resolve().parents[1]


def assemble(staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "app" / "Dockerfile", staging / "Dockerfile")
    shutil.copy2(ROOT / "deploy" / "hf_space_README.md", staging / "README.md")
    shutil.copy2(ROOT / ".dockerignore", staging / ".dockerignore") if (ROOT / ".dockerignore").exists() else None
    for rel in ("src", "configs", "app"):
        dst = staging / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            ROOT / rel,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data", "certs"),
        )
    # đảm bảo 3 file model có mặt
    model = staging / "app" / "model"
    model.mkdir(parents=True, exist_ok=True)
    for name in ("timexer_dinov3.onnx", "timexer_dinov3_head.pt", "model_card.json"):
        src = ROOT / "app" / "model" / name
        if src.exists():
            shutil.copy2(src, model / name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default=None, help="bỏ trống = dùng token đã login")
    p.add_argument("--repo", default="timexer-ai-voice-age")
    args = p.parse_args()
    if args.token:
        login(token=args.token, add_to_git_credential=False)
    user = whoami()["name"]
    repo_id = f"{user}/{args.repo}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "space"
        assemble(staging)
        print("Staging:", list(staging.rglob("*"))[:20], "...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy TimeXer-DINOv3 voice-age app (Docker + ONNX)",
            ignore_patterns=["**/__pycache__/**"],
        )
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"App:   https://{user}-{args.repo}.hf.space")


if __name__ == "__main__":
    main()
