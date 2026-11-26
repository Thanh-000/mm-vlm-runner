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


def _digest_file(path: Path, alg: str) -> str:
    h = hashlib.new(alg)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_binary_archive(path: Path, min_bytes: int = 128) -> bool:
    """Return True if path exists, meets min_bytes, and is not an HTML error/quota page."""
    if not path.is_file() or path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as fh:
        head = fh.read(256).lower()
    return (b"<html" not in head) and (b"<!doctype" not in head)


def _verify_checksum(path: Path, checksum: str) -> bool:
    cs = checksum.strip().lower()
    if not cs:
        return True
    alg, _, expected = cs.partition("=")
    if not expected:
        return True
    norm_alg = "sha256" if alg in ("sha-256", "sha256") else alg
    return _digest_file(path, norm_alg) == expected.strip()


def _aria2_single_pass(
    urls: list[str],
    dest_path: Path,
    *,
    conn: int,
    max_tries: int,
    retry_wait: int,
    headers: dict[str, str] | None = None,
    checksum: str = "",
) -> None:
    cmd = [
        "aria2c", f"-x{conn}", f"-s{conn}", "-k1M", "--min-split-size=1M",
        "--file-allocation=none", "--optimize-concurrent-downloads=true",
        "--connect-timeout=15", "--timeout=45",
        f"--max-tries={max_tries}", f"--retry-wait={retry_wait}",
        "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn",
        "-d", str(dest_path.parent), "-o", dest_path.name,
    ]
    cs = checksum.strip().lower()
    if cs.startswith(("md5=", "sha-256=")):
        cmd.append(f"--checksum={cs}")
    elif cs.startswith("sha256="):
        cmd.append(f"--checksum=sha-256={cs.split('=', 1)[1]}")
    for k, v in (headers or {}).items():
        cmd.append(f"--header={k}: {v}")
    cmd.extend(urls)
    subprocess.run(cmd, check=False)


def aria2_download_file(
    urls: str | list[str],
    dest_path: Path,
    *,
    headers: dict[str, str] | None = None,
    checksum: str = "",
    min_bytes: int = 128,
    max_conn: int = 16,
) -> Path:
    """Download file at max aria2c speed with automatic web-throttle backoff and HTML/checksum guard."""
    url_list = [urls.strip()] if isinstance(urls, str) else [u.strip() for u in urls if u.strip()]
    if not url_list:
        raise ValueError(f"Empty download URL list for {dest_path.name}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_binary_archive(dest_path, min_bytes) and _verify_checksum(dest_path, checksum):
        return dest_path

    aria2_ctrl = dest_path.with_name(dest_path.name + ".aria2")
    hdrs = headers or {}
    if shutil.which("aria2c"):
        _aria2_single_pass(url_list, dest_path, conn=max_conn, max_tries=3, retry_wait=2, headers=hdrs, checksum=checksum)
        if not (is_valid_binary_archive(dest_path, min_bytes) and _verify_checksum(dest_path, checksum)) and max_conn > 4:
            dest_path.unlink(missing_ok=True)
            aria2_ctrl.unlink(missing_ok=True)
            _aria2_single_pass([url_list[0]], dest_path, conn=4, max_tries=4, retry_wait=5, headers=hdrs, checksum=checksum)

    if not (is_valid_binary_archive(dest_path, min_bytes) and _verify_checksum(dest_path, checksum)):
        dest_path.unlink(missing_ok=True)
        aria2_ctrl.unlink(missing_ok=True)
        if shutil.which("curl") and not url_list[0].startswith("file://"):
            cmd_c = ["curl", "-fL", "--retry", "4", url_list[0], "-o", str(dest_path)]
            for k, v in hdrs.items():
                cmd_c.extend(["-H", f"{k}: {v}"])
            subprocess.run(cmd_c, check=False)
        if not (is_valid_binary_archive(dest_path, min_bytes) and _verify_checksum(dest_path, checksum)):
            dest_path.unlink(missing_ok=True)
            with urllib.request.urlopen(urllib.request.Request(url_list[0], headers={"User-Agent": "Mozilla/5.0", **hdrs}), timeout=120) as r:
                dest_path.write_bytes(r.read())

    if not is_valid_binary_archive(dest_path, min_bytes):
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file invalid, HTML error page, or below min_bytes={min_bytes}: {dest_path}")
    if not _verify_checksum(dest_path, checksum):
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch ({checksum}) for downloaded file: {dest_path}")
    return dest_path


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
        def _run_batch(items: list[tuple[str, dict[str, str], str, Path]], j: int, x: int, wait_s: int, tries: int) -> None:
            lines: list[str] = []
            for url, hmap, oid, dst in items:
                dst.parent.mkdir(parents=True, exist_ok=True)
                lines.extend([url, f"  dir={dst.parent}", f"  out={dst.name}"])
                for hk, hv in hmap.items():
                    lines.append(f"  header={hk}: {hv}")
                if oid:
                    lines.append(f"  checksum=sha-256={oid}")
            mf = dest_dir / ".aria2_lfs_manifest.txt"
            mf.write_text("\n".join(lines) + "\n", encoding="utf8")
            subprocess.run(
                ["aria2c", f"--input-file={mf}", f"-j{j}", f"-x{x}", f"-s{x}", "-k1M", "--file-allocation=none",
                 "--check-integrity=true", f"--retry-wait={wait_s}", f"--max-tries={tries}",
                 "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn"],
                check=False,
            )
            mf.unlink(missing_ok=True)

        _run_batch(direct_items, j=32, x=8, wait_s=3, tries=4)
        rem_t1 = [it for it in direct_items if not is_valid_binary_archive(it[3], 1)]
        if rem_t1:
            _run_batch(rem_t1, j=6, x=2, wait_s=5, tries=5)

    rem = [(u, hmap, oid, dst) for u, hmap, oid, dst in direct_items if not dst.is_file()]
    if rem:
        def _dl_one(item: tuple[str, dict[str, str], str, Path]) -> None:
            url, hmap, oid, dst = item
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **hmap}), timeout=60) as r:
                tmp.write_bytes(r.read())
            if oid and _digest_file(tmp, "sha256") != oid:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"SHA-256 mismatch for {dst}")
            tmp.replace(dst)

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(_dl_one, it) for it in rem]):
                fut.result()
