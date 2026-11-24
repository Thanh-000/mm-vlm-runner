"""Atomic JSON serialization, YAML loading, and SHA-256 verification helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_bytes(data: bytes) -> str:
    """Return lowercase hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Stream-compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically via temp file replacement with allow_nan=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False)
    tmp.write_text(text, encoding="utf8")
    tmp.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping from disk."""
    with path.open("r", encoding="utf8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config {path}, got {type(data).__name__}")
    return data
