"""Determinism configuration and full experimental provenance locks."""

from __future__ import annotations

import datetime as dt
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import transformers


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set Python, PyTorch CPU/CUDA RNGs and optionally enforce deterministic kernels."""
    seed_i = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_i)
    random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_i)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """Return a freshly seeded CPU PyTorch generator."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _git_metadata(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None or not (repo_root / ".git").exists():
        return {"git_commit": None, "git_dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = head.stdout.strip() if head.returncode == 0 and head.stdout.strip() else None
        st = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def build_run_lock(
    config: dict[str, Any],
    repo_root: Path | None = None,
    dataset_hashes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture full software, hardware, git, dataset, and config provenance."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "cpu"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda_avail else 0.0
    git_info = _git_metadata(repo_root)
    return {
        "schema": "mnarm-run-lock-v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "allow_test": bool(config.get("allow_test", False)),
        "git_commit": git_info["git_commit"],
        "git_dirty": git_info["git_dirty"],
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "dataset_hashes": dict(dataset_hashes or {}),
        "config": dict(config),
    }
