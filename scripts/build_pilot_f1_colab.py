"""Build minimal clean Colab Web notebook for Pilot F1 (dev-only, 0 D_cal labels)."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import textwrap
import warnings
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "colab" / "Pilot_F1_ZeroMNAR_ColabWeb.ipynb"
OUT_ZIP = ROOT / "colab" / "mnarm_pilot_f1_code.zip"

CRISISMMA_HF_REV = "10a5626ba112ab9c50369c473a2062bcf205d708"


def _cell(src: str, kind: str = "code", cid: str = "") -> dict:
    clean = textwrap.dedent(src).lstrip("\n")
    if kind == "markdown":
        return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": clean.splitlines(keepends=True)}
    return {"id": cid, "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": clean.splitlines(keepends=True)}


def _pack_repo_zip(zip_path: Path) -> tuple[str, str]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    include_roots = ("pyproject.toml", "requirements.lock.txt", "LICENSE", "README.md")
    include_dirs = ("configs", "src", "tests")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in include_roots:
            zf.write(ROOT / fn, fn)
        for d in include_dirs:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, p.relative_to(ROOT).as_posix())
    raw = zip_path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii")


def build() -> None:
    code_sha, code_b64 = _pack_repo_zip(OUT_ZIP)
    cells = [
        _cell(
            """
            # Pilot F1 — Falsification Test of H_ZERO-MNAR (Strict dev-Only, 0 D_cal Labels)

            Runs `L_MNAR` + `SelfNormalizedBayesTransportHead` vs `Standard CE` on **CrisisMMD P4 (`dev` only)**:
            1. Installs clean `mnarm` package and runs full `pytest -v` preflight gate FIRST.
            2. Fetches pinned HuggingFace snapshots (`CLIP` + `SigLIP`) and CrisisMMD images (`train` + `dev` only) from official HF commit `10a5626`.
            3. Runs `configs/pilot_f1_clip_p4.yaml` and `configs/pilot_f1_siglip_p4.yaml` with `allow_test=false`.
            4. Displays `PILOT_F1_REPORT.md` and downloads `PILOT_F1_RESULTS.zip`.
            """,
            kind="markdown",
            cid="pf1-cell-0",
        ),
        _cell(
            """
            #@title Configuration { display-mode: "form" }
            RUN_CLIP_P4 = True  # @param {type:"boolean"}
            RUN_SIGLIP_P4 = True  # @param {type:"boolean"}
            PARALLEL_BACKBONES = True  # @param {type:"boolean"}
            MOUNT_DRIVE = False  # @param {type:"boolean"}
            DRIVE_CONTRACTS_PATH = ""  # @param {type:"string"}
            A2Z_INPUTS_URL = ""  # @param {type:"string"}
            CRISISMMA_TAR_URL = ""  # @param {type:"string"}
            AUTO_DOWNLOAD = True  # @param {type:"boolean"}
            """,
            cid="pf1-cell-1",
        ),
        _cell(
            f"""
            import base64, concurrent.futures, hashlib, importlib, io, os, shutil, site, subprocess, sys, zipfile
            from pathlib import Path
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Enable GPU runtime in Colab before executing Pilot F1.")

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            OUT_DIR = WS / "outputs"
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if Path("/dev/shm").exists() else (WS / "crisismmd_v2")
            for d in (WS, CODE_DIR, OUT_DIR, IMG_DIR):
                d.mkdir(parents=True, exist_ok=True)

            CODE_SHA = {code_sha!r}
            CODE_B64 = {code_b64!r}
            raw_zip = base64.b64decode(CODE_B64)
            if hashlib.sha256(raw_zip).hexdigest() != CODE_SHA:
                raise ValueError("Embedded mnarm repo SHA mismatch!")
            with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
                zf.extractall(CODE_DIR)

            def _pip_deps():
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "datasets==3.2.0", "PyYAML", "scipy", "pillow", "huggingface_hub", "pytest", "wheel", "gdown"], check=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(CODE_DIR)], check=True)

            def _apt_deps():
                if shutil.which("aria2c") is None and shutil.which("apt-get"):
                    subprocess.run(["apt-get", "-qq", "update"], check=False)
                    subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz"], check=False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_pip_deps)
                f2 = pool.submit(_apt_deps)
                f1.result()
                f2.result()

            site.addsitedir(str(CODE_DIR / "src"))
            importlib.invalidate_caches()
            import mnarm
            print("[IMPORT OK]", mnarm.__file__, "| aria2c=", shutil.which("aria2c"), "| pigz=", shutil.which("pigz"), flush=True)
            print("[PREFLIGHT] Running pytest gate before any GPU experiment...", flush=True)
            subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(CODE_DIR), check=True)
            """,
            cid="pf1-cell-2",
        ),
        _cell(
            f"""
            import concurrent.futures, importlib, json, shutil, site, subprocess, sys, time, urllib.parse, zipfile
            from pathlib import Path
            import requests
            from google.colab import files as colab_files

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if Path("/dev/shm").exists() else (WS / "crisismmd_v2")
            IMG_DIR.mkdir(parents=True, exist_ok=True)
            site.addsitedir(str(CODE_DIR / "src"))
            importlib.invalidate_caches()
            from mnarm.data import extract_contracts_from_a2z_zip

            LOCAL_CANDIDATES = [
                Path("/content/crisismmd_p1_p4_contracts.zip"),
                Path("/content/JCMIL_A2Z_INPUTS.zip"),
                Path("/content/crisismmd_contracts_dl.zip"),
            ]
            for cand in LOCAL_CANDIDATES:
                if cand.is_file() and not zipfile.is_zipfile(cand):
                    print(f"[WARN] Removing invalid/corrupt cached file {{cand}} (size={{cand.stat().st_size}} B)", flush=True)
                    cand.unlink()

            CONTRACTS_ZIP = next((p for p in LOCAL_CANDIDATES if p.is_file() and zipfile.is_zipfile(p)), None)
            if CONTRACTS_ZIP is None and MOUNT_DRIVE:
                from google.colab import drive as colab_drive
                colab_drive.mount("/content/drive")
                drive_cands = []
                if str(DRIVE_CONTRACTS_PATH).strip():
                    drive_cands.append(Path(str(DRIVE_CONTRACTS_PATH).strip()))
                drive_cands.extend([
                    Path("/content/drive/MyDrive/jcmil_benchmark_v1/JCMIL_A2Z_INPUTS.zip"),
                    Path("/content/drive/MyDrive/crisismmd_p1_p4_contracts.zip"),
                ])
                CONTRACTS_ZIP = next((p for p in drive_cands if p.is_file() and zipfile.is_zipfile(p)), None)

            if CONTRACTS_ZIP is None and str(A2Z_INPUTS_URL).strip():
                url_in = str(A2Z_INPUTS_URL).strip()
                dst_dl = Path("/content/crisismmd_contracts_dl.zip")
                host_in = urllib.parse.urlparse(url_in).netloc.lower()
                is_drive = "drive" in host_in and host_in.endswith("google.com")
                is_direct_drive = is_drive and "confirm=t" in url_in
                if (is_direct_drive or not is_drive) and shutil.which("aria2c"):
                    subprocess.run(["aria2c", "-x16", "-s16", "-k1M", "--allow-overwrite=true", url_in, "-d", "/content", "-o", dst_dl.name], check=False)
                if not dst_dl.is_file() or not zipfile.is_zipfile(dst_dl):
                    dst_dl.unlink(missing_ok=True)
                    if is_drive:
                        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", url_in, "-O", str(dst_dl)], check=True)
                    else:
                        subprocess.run(["curl", "-fL", url_in, "-o", str(dst_dl)], check=True)
                CONTRACTS_ZIP = dst_dl

            if CONTRACTS_ZIP is None:
                print("Upload lightweight crisismmd_p1_p4_contracts.zip (~5.8 MB) or full JCMIL_A2Z_INPUTS.zip:")
                up = colab_files.upload()
                if not up:
                    raise RuntimeError("No contract ZIP uploaded.")
                nm, blob = next(iter(up.items()))
                dst_up = Path("/content") / Path(nm).name
                dst_up.write_bytes(blob)
                CONTRACTS_ZIP = dst_up

            if not CONTRACTS_ZIP.is_file() or not zipfile.is_zipfile(CONTRACTS_ZIP):
                sz = CONTRACTS_ZIP.stat().st_size if CONTRACTS_ZIP.is_file() else 0
                head = b""
                if CONTRACTS_ZIP.is_file():
                    with CONTRACTS_ZIP.open("rb") as fh:
                        head = fh.read(120)
                    if CONTRACTS_ZIP.parent == Path("/content"):
                        CONTRACTS_ZIP.unlink()
                print(f"[ERROR] Invalid ZIP {{CONTRACTS_ZIP}}: size={{sz}} B, head={{head!r}}", flush=True)
                raise RuntimeError("Contracts file is not a valid ZIP; upload crisismmd_p1_p4_contracts.zip (~5.8 MB) or set MOUNT_DRIVE=True.")

            SEQ_DIR = extract_contracts_from_a2z_zip(CONTRACTS_ZIP, WS / "crisismmd_seq_inputs")
            manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
            # Fetch ONLY train and dev images (strict zero test touch in Pilot F1)
            if not (IMG_DIR / "data_image").is_dir() and (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"

            req_set = {{r["image"] for sp in ("train", "dev") for r in manifest["rows"][sp]}}
            need_tar = any(not (IMG_DIR / p).is_file() for p in req_set)
            if need_tar:
                tar_cands = [Path("/content/CrisisMMD_v2.0.tar.gz")]
                for p in tar_cands:
                    if p.is_file() and p.stat().st_size <= 100_000_000:
                        p.unlink(missing_ok=True)
                if MOUNT_DRIVE:
                    from google.colab import drive as colab_drive
                    colab_drive.mount("/content/drive")
                    tar_cands.extend([
                        Path("/content/drive/MyDrive/CrisisMMD_v2.0.tar.gz"),
                        Path("/content/drive/MyDrive/jcmil_benchmark_v1/CrisisMMD_v2.0.tar.gz"),
                    ])
                tar_path = next((p for p in tar_cands if p.is_file() and p.stat().st_size > 100_000_000), None)
                tar_url = str(CRISISMMA_TAR_URL).strip()
                if tar_path is None and tar_url:
                    if shutil.which("aria2c") is None and shutil.which("apt-get"):
                        subprocess.run(["apt-get", "-qq", "update"], check=False)
                        subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz"], check=False)
                    dl_tar = Path("/content/CrisisMMD_v2.0.tar.gz")
                    print("[ARIA2] Downloading CrisisMMD_v2.0.tar.gz (16 streams)...", flush=True)
                    if shutil.which("aria2c"):
                        subprocess.run(["aria2c", "-x16", "-s16", "-k1M", "--allow-overwrite=true", tar_url, "-d", "/content", "-o", dl_tar.name], check=False)
                    if not dl_tar.is_file() or dl_tar.stat().st_size <= 100_000_000:
                        dl_tar.unlink(missing_ok=True)
                        host_t = urllib.parse.urlparse(tar_url).netloc.lower()
                        if "drive" in host_t and host_t.endswith("google.com"):
                            subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", tar_url, "-O", str(dl_tar)], check=True)
                        else:
                            subprocess.run(["curl", "-fL", tar_url, "-o", str(dl_tar)], check=True)
                    if dl_tar.is_file() and dl_tar.stat().st_size > 100_000_000:
                        tar_path = dl_tar

                if tar_path is not None:
                    print(f"[EXTRACT] Extracting {{tar_path}} -> {{IMG_DIR}}...", flush=True)
                    if shutil.which("pigz"):
                        subprocess.run(["tar", "-I", "pigz", "-xf", str(tar_path), "-C", str(IMG_DIR)], check=True)
                    else:
                        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(IMG_DIR)], check=True)
                    if tar_path.parent == Path("/content"):
                        tar_path.unlink(missing_ok=True)
                    if not (IMG_DIR / "data_image").is_dir() and (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                        IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"

            test_only = {{r["image"] for r in manifest["rows"]["test"]}} - req_set
            removed_test = 0
            for rel_t in test_only:
                tp = IMG_DIR / rel_t
                if tp.is_file():
                    tp.unlink()
                    removed_test += 1
            if removed_test:
                print(f"[ISOLATION] Removed {{removed_test}} test-only image files from disk for strict dev-only Pilot F1.", flush=True)

            req_imgs = sorted(p for p in req_set if not (IMG_DIR / p).is_file())
            if req_imgs:
                print(f"[HF DOWNLOAD] Fetching {{len(req_imgs)}} train/dev CrisisMMD images from pinned QCRI/CrisisMMD commit...", flush=True)
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
                session.mount("https://", adapter)
                session.mount("http://", adapter)

                def _fetch(rel_p):
                    dst = IMG_DIR / rel_p
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + ".partial")
                    url = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{CRISISMMA_HF_REV}/" + urllib.parse.quote(rel_p, safe="/")
                    for attempt in range(5):
                        try:
                            r = session.get(url, timeout=(20, 60))
                            if r.status_code == 429 or r.status_code >= 500:
                                if attempt == 4:
                                    r.raise_for_status()
                                delay = r.headers.get("Retry-After", "")
                                time.sleep(min(60, max(2, int(delay) if delay.isdigit() else 2 * (2 ** attempt))))
                                continue
                            r.raise_for_status()
                            tmp.write_bytes(r.content)
                            tmp.replace(dst)
                            return
                        except (requests.Timeout, requests.ConnectionError):
                            if attempt == 4:
                                raise
                            time.sleep(min(60, 2 * (2 ** attempt)))
                    raise RuntimeError(f"Exhausted retries downloading {{rel_p}}")

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(_fetch, p) for p in req_imgs]
                    for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                        fut.result()
                        if idx % 500 == 0 or idx == len(futures):
                            print(f"[HF DOWNLOAD] {{idx}}/{{len(futures)}} train/dev images ready", flush=True)
            print("[DATA READY] train/dev images verified:", len(manifest["rows"]["train"]) + len(manifest["rows"]["dev"]))
            """,
            cid="pf1-cell-3",
        ),
        _cell(
            """
            import concurrent.futures, os, threading
            from collections import deque
            import torch
            from IPython.display import Markdown, display

            run_env = {**os.environ, "PYTHONPATH": str(CODE_DIR / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
            print_lock = threading.Lock()

            def _run_job(tag, cfg_rel, out_sub):
                out_path = OUT_DIR / out_sub
                out_path.mkdir(parents=True, exist_ok=True)
                log_path = OUT_DIR / f"{out_sub}.log"
                cmd = [
                    sys.executable, "-u", "-m", "mnarm.runners.pilot_f1",
                    "--config", str(CODE_DIR / cfg_rel),
                    "--seq-inputs", str(SEQ_DIR), "--images", str(IMG_DIR),
                    "--out", str(out_path),
                ]
                tail = deque(maxlen=40)
                with log_path.open("w", encoding="utf8") as lf:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=run_env)
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        lf.write(line)
                        tail.append(line.rstrip("\\n"))
                        with print_lock:
                            print(f"[{tag}] {line}", end="", flush=True)
                    rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"{tag} failed (exit={rc}). Last 40 log lines:\\n" + "\\n".join(tail))
                return out_sub

            jobs = []
            if RUN_CLIP_P4:
                jobs.append(("CLIP_P4", "configs/pilot_f1_clip_p4.yaml", "clip_p4"))
            if RUN_SIGLIP_P4:
                jobs.append(("SIGLIP_P4", "configs/pilot_f1_siglip_p4.yaml", "siglip_p4"))

            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if PARALLEL_BACKBONES and len(jobs) > 1 and vram_gb >= 23.0:
                print(f"[PARALLEL] Running {', '.join(j[0] for j in jobs)} concurrently on {vram_gb:.1f} GB GPU", flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                    for fut in [pool.submit(_run_job, *j) for j in jobs]:
                        fut.result()
            else:
                for j in jobs:
                    _run_job(*j)

            for _, _, out_sub in jobs:
                display(Markdown((OUT_DIR / out_sub / "PILOT_F1_REPORT.md").read_text(encoding="utf8")))

            RES_ZIP = Path("/content/PILOT_F1_RESULTS.zip")
            with zipfile.ZipFile(RES_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for fp in sorted(OUT_DIR.rglob("*")):
                    if fp.is_file():
                        zout.write(fp, fp.relative_to(OUT_DIR).as_posix())
            print("[PACKAGED]", RES_ZIP, RES_ZIP.stat().st_size, "bytes")
            if AUTO_DOWNLOAD:
                colab_files.download(str(RES_ZIP))
            """,
            cid="pf1-cell-4",
        ),
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    nb_json = json.dumps(nb, indent=2, ensure_ascii=False)
    if "$" in nb_json:
        raise AssertionError("Forbidden $ in Pilot F1 Colab notebook")
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", SyntaxWarning)
                ast.parse("".join(c["source"]))
                if w:
                    raise SyntaxError(f"SyntaxWarning in cell {idx}: {w[0].message}")
    OUT_NB.write_text(nb_json, encoding="utf8")
    print(f"Built {OUT_ZIP} ({len(base64.b64decode(code_b64))} B, sha={code_sha[:12]})")
    print(f"Built {OUT_NB} ({len(nb_json)} B)")


if __name__ == "__main__":
    build()
