"""Safe .tar/.tar.gz/.zip archive extraction and HuggingFace Git-LFS Batch CDN fetcher."""

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


def _safe_target(root: Path, rel_name: str) -> Path:
    root_res = root.resolve()
    target = (root_res / rel_name).resolve()
    if target != root_res and not str(target).startswith(str(root_res) + os.sep):
        raise ValueError(f"Unsafe archive entry escapes destination root: {rel_name!r}")
    return target


def extract_archive_safe(archive_path: Path, dest_dir: Path) -> None:
    """Safely unpack .tar, .tar.gz/.tgz, or .zip into dest_dir blocking path traversal."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                target = _safe_target(dest_dir, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            for member in tf.getmembers():
                if member.islnk() or member.issym():
                    raise ValueError(f"Unsafe symlink/hardlink in archive: {member.name!r}")
                _safe_target(dest_dir, member.name)
            low_name = archive_path.name.lower()
            if os.name == "posix" and shutil.which("tar"):
                if low_name.endswith((".tar.gz", ".tgz")) and shutil.which("pigz"):
                    subprocess.run(["tar", "-I", "pigz", "-xf", str(archive_path), "-C", str(dest_dir)], check=True)
                    return
                if low_name.endswith(".tar"):
                    subprocess.run(["tar", "-xf", str(archive_path), "-C", str(dest_dir)], check=True)
                    return
            kw: dict[str, Any] = {"filter": "data"} if "filter" in inspect.signature(tf.extractall).parameters else {}
            tf.extractall(dest_dir, **kw)
        return
    raise ValueError(f"Unsupported or corrupt archive format: {archive_path}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lfs_batch_resolve(
    repo_id: str, objs: list[tuple[str, int]], hf_token: str = ""
) -> dict[str, tuple[str, dict[str, str]]]:
    actions: dict[str, tuple[str, dict[str, str]]] = {}
    uniq = list(dict.fromkeys(objs))
    hdrs = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json", "User-Agent": "git-lfs/3.4.0"}
    if hf_token:
        hdrs["Authorization"] = f"Bearer {hf_token}"
    url = f"https://huggingface.co/datasets/{repo_id}.git/info/lfs/objects/batch"
    for off in range(0, len(uniq), 100):
        body = json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": o, "size": s} for o, s in uniq[off : off + 100]]}).encode("utf8")
        with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=hdrs, method="POST"), timeout=30) as r:
            resp = json.loads(r.read())
        for obj in resp.get("objects", []):
            dl = obj.get("actions", {}).get("download", {})
            if dl.get("href"):
                actions[str(obj["oid"])] = (str(dl["href"]), {str(k): str(v) for k, v in dl.get("header", {}).items()})
    return actions


def hf_lfs_batch_download(
    repo_id: str, revision: str, rel_paths: list[str], dest_dir: Path, hf_token: str = ""
) -> None:
    """Download HF files at max CDN bandwidth via Git-LFS Batch API + aria2c -j 32."""
    tok = hf_token.strip() or os.environ.get("HF_TOKEN", "").strip()
    need = [rel.replace("\\", "/").lstrip("/") for rel in rel_paths if not (dest_dir / rel).is_file()]
    if not need:
        return

    req_hdrs = {"User-Agent": "Mozilla/5.0"}
    if tok:
        req_hdrs["Authorization"] = f"Bearer {tok}"
    meta_by_rel: dict[str, dict[str, Any]] = {}
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo_id}?blobs=1", headers=req_hdrs), timeout=30) as r:
            info = json.loads(r.read())
        meta_by_rel = {str(s["rfilename"]): s for s in info.get("siblings", [])}
    except Exception:
        meta_by_rel = {}

    lfs_items: list[tuple[str, str, int, Path]] = []
    direct_items: list[tuple[str, dict[str, str], str, Path]] = []
    for rel in need:
        dst = dest_dir / rel
        meta = meta_by_rel.get(rel, {})
        lfs = meta.get("lfs")
        if isinstance(lfs, dict) and lfs.get("sha256") and lfs.get("size"):
            oid, sz = str(lfs["sha256"]), int(lfs["size"])
            if dst.is_file() and dst.stat().st_size == sz:
                continue
            lfs_items.append((rel, oid, sz, dst))
        else:
            res_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{urllib.parse.quote(rel, safe='/')}"
            direct_items.append((res_url, {"Authorization": f"Bearer {tok}"} if tok else {}, "", dst))

    if lfs_items:
        actions = _lfs_batch_resolve(repo_id, [(oid, sz) for _, oid, sz, _ in lfs_items], hf_token=tok)
        for rel, oid, _, dst in lfs_items:
            if oid in actions:
                href, hmap = actions[oid]
                direct_items.append((href, hmap, oid, dst))
            else:
                res_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{urllib.parse.quote(rel, safe='/')}"
                direct_items.append((res_url, {"Authorization": f"Bearer {tok}"} if tok else {}, "", dst))

    if not direct_items:
        return
    if shutil.which("aria2c"):
        lines: list[str] = []
        for url, hmap, oid, dst in direct_items:
            dst.parent.mkdir(parents=True, exist_ok=True)
            lines.extend([url, f"  dir={dst.parent}", f"  out={dst.name}"])
            for hk, hv in hmap.items():
                lines.append(f"  header={hk}: {hv}")
            if oid:
                lines.append(f"  checksum=sha-256={oid}")
        mf = dest_dir / ".aria2_lfs_manifest.txt"
        mf.write_text("\n".join(lines) + "\n", encoding="utf8")
        subprocess.run(["aria2c", f"--input-file={mf}", "-j", "32", "-x", "4", "-s", "4", "-k", "1M", "--check-integrity=true", "--retry-wait=3", "--max-tries=6", "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn"], check=False)
        mf.unlink(missing_ok=True)

    rem = [(u, hmap, oid, dst) for u, hmap, oid, dst in direct_items if not dst.is_file()]
    if rem:
        def _dl_one(item: tuple[str, dict[str, str], str, Path]) -> None:
            url, hmap, oid, dst = item
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **hmap}), timeout=60) as r:
                tmp.write_bytes(r.read())
            if oid and _sha256_file(tmp) != oid:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"SHA-256 mismatch for {dst}")
            tmp.replace(dst)

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(_dl_one, it) for it in rem]):
                fut.result()
