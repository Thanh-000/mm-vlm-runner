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
            A2Z_INPUTS_URL = ""  # @param {type:"string"}
            CRISISMMA_TAR_URL = ""  # @param {type:"string"}
            AUTO_DOWNLOAD = True  # @param {type:"boolean"}
            """,
            cid="pf1-cell-1",
        ),
        _cell(
            f"""
            import base64, hashlib, importlib, io, os, site, subprocess, sys, zipfile
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

            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "datasets==3.2.0", "PyYAML", "scipy", "pillow", "huggingface_hub", "pytest", "wheel"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(CODE_DIR)], check=True)
            site.addsitedir(str(CODE_DIR / "src"))
            importlib.invalidate_caches()
            import mnarm
            print("[IMPORT OK]", mnarm.__file__, flush=True)
            print("[PREFLIGHT] Running pytest gate before any GPU experiment...", flush=True)
            subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(CODE_DIR), check=True)
            """,
            cid="pf1-cell-2",
        ),
        _cell(
            f"""
            import concurrent.futures, importlib, json, site, subprocess, tarfile, time, urllib.parse
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

            A2Z_ZIP = Path("/content/JCMIL_A2Z_INPUTS.zip")
            if not A2Z_ZIP.is_file():
                if str(A2Z_INPUTS_URL).strip():
                    subprocess.run(["curl", "-fL", str(A2Z_INPUTS_URL).strip(), "-o", str(A2Z_ZIP)], check=True)
                else:
                    print("Upload JCMIL_A2Z_INPUTS.zip (contract manifest + splits):")
                    up = colab_files.upload()
                    nm, blob = next(iter(up.items()))
                    A2Z_ZIP.write_bytes(blob)

            SEQ_DIR = extract_contracts_from_a2z_zip(A2Z_ZIP, WS / "crisismmd_seq_inputs")
            manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
            # Fetch ONLY train and dev images (strict zero test touch in Pilot F1)
            req_set = {{r["image"] for sp in ("train", "dev") for r in manifest["rows"][sp]}}
            tar_path = Path("/content/CrisisMMD_v2.0.tar.gz")
            if not tar_path.is_file() and str(CRISISMMA_TAR_URL).strip():
                subprocess.run(["curl", "-fL", str(CRISISMMA_TAR_URL).strip(), "-o", str(tar_path)], check=True)
            if tar_path.is_file():
                missing_tar = {{p for p in req_set if not (IMG_DIR / p).is_file()}}
                if missing_tar:
                    print(f"[TAR EXTRACT] Extracting up to {{len(missing_tar)}} train/dev images from {{tar_path}}...", flush=True)
                    extracted = 0
                    with tarfile.open(tar_path, "r:gz") as tf:
                        for member in tf:
                            if not member.isfile():
                                continue
                            norm = member.name[2:] if member.name.startswith("./") else member.name.lstrip("/")
                            if norm.startswith("CrisisMMD_v2.0/"):
                                norm = norm[len("CrisisMMD_v2.0/"):]
                            if norm in missing_tar:
                                dst = IMG_DIR / norm
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                src_f = tf.extractfile(member)
                                if src_f is not None:
                                    tmp = dst.with_name(dst.name + ".partial")
                                    tmp.write_bytes(src_f.read())
                                    tmp.replace(dst)
                                    extracted += 1
                    print(f"[TAR EXTRACT] Extracted {{extracted}} train/dev images.", flush=True)

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
            import os
            from IPython.display import Markdown, display

            run_env = {**os.environ, "PYTHONPATH": str(CODE_DIR / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}

            if RUN_CLIP_P4:
                subprocess.run([
                    sys.executable, "-u", "-m", "mnarm.runners.pilot_f1",
                    "--config", str(CODE_DIR / "configs/pilot_f1_clip_p4.yaml"),
                    "--seq-inputs", str(SEQ_DIR), "--images", str(IMG_DIR),
                    "--out", str(OUT_DIR / "clip_p4"),
                ], env=run_env, check=True)
                display(Markdown((OUT_DIR / "clip_p4/PILOT_F1_REPORT.md").read_text(encoding="utf8")))

            if RUN_SIGLIP_P4:
                subprocess.run([
                    sys.executable, "-u", "-m", "mnarm.runners.pilot_f1",
                    "--config", str(CODE_DIR / "configs/pilot_f1_siglip_p4.yaml"),
                    "--seq-inputs", str(SEQ_DIR), "--images", str(IMG_DIR),
                    "--out", str(OUT_DIR / "siglip_p4"),
                ], env=run_env, check=True)
                display(Markdown((OUT_DIR / "siglip_p4/PILOT_F1_REPORT.md").read_text(encoding="utf8")))

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
