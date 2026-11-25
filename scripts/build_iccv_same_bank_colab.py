"""Build clean Colab Web notebook for unified ICCV Same-Bank evaluation (dev-only)."""

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
OUT_NB = ROOT / "colab" / "ICCV_SameBank_ZeroMNAR_ColabWeb.ipynb"
OUT_ZIP = ROOT / "colab" / "mnarm_iccv_same_bank_code.zip"

CRISISMMA_HF_REV = "10a5626ba112ab9c50369c473a2062bcf205d708"


def _cell(src: str, kind: str = "code", cid: str = "") -> dict:
    clean = textwrap.dedent(src).lstrip("\n")
    if kind == "markdown":
        return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": clean.splitlines(keepends=True)}
    return {"id": cid, "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": clean.splitlines(keepends=True)}


def _pack_repo_zip(zip_path: Path) -> tuple[str, str]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    include_roots = ("pyproject.toml", "requirements.lock.txt", "LICENSE", "README.md")
    include_dirs = ("configs", "docs", "src", "tests")
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
            # ICCV Same-Bank Benchmark — `CSS-CT` + 14 Verbatim Author Mergers + VLM-CL/Analytic SOTA (dev-Only)

            Runs all **14 verbatim author merging operators** (`mean`, `exposure`, `ta_sum`, `ties20`, `magmax`, `magmax_s1`, `knots` [ICLR'25], `tsvm` [CVPR'25], `iso_c` [ICML'25], `iso_cts` [ICML'25], `wudi` [ICML'25], `opcm` [ICLR'25], `dop` [NeurIPS'25], `dc_lora` [CVPR'26]) alongside **5 continual/VLM-CL/oracle tracks** (`seq_ft`, `seq_replay256`, `mg_clip` [ICCV'25 Highlight], `pal_rls` [Analytic MMCIL'25], `joint_oracle`) on the **EXACT SAME source expert bank** across **5 information regimes** (`raw`, `gr_clip` [NeurIPS'25, 0-label], `unlab_center` [0-label], `zero_mnar` [`CSS-CT` Ours, 0 D_cal labels, O(|Y|d + d^2) moments], `cal_bias` [100% D_cal]) + encoder-preservation checks (`Unseen-Prompt OOD`, `I->T R@1` Raw vs Re-Anchored) + inline ICCV diagnostic figures (`fig1..fig4`):
            1. Runs full `pytest -v` preflight gate FIRST (all parity + hygiene + double-blind checks).
            2. Reuses/downloads CrisisMMD `train` + `dev` images (`allow_test=false`) with RAM-aware `/dev/shm` caching (auto-enabled on A100 / G4 / L4).
            3. Runs `mnarm.runners.iccv_same_bank` concurrently for CLIP and SigLIP across `P4 -> P3 -> P2 -> P1`.
            4. Displays `ICCV_SAME_BANK_*_SUMMARY.md`, renders inline diagnostic PNGs (`fig1..fig4`), exports `.tex` tables + `.pdf` vector figures, and downloads `ICCV_SAME_BANK_DEV_RESULTS.zip`.
            """,
            kind="markdown",
            cid="iccv-cell-0",
        ),
        _cell(
            """
            #@title Configuration { display-mode: "form" }
            RUN_CLIP = True  # @param {type:"boolean"}
            RUN_SIGLIP = True  # @param {type:"boolean"}
            PARALLEL_BACKBONES = True  # @param {type:"boolean"}
            RUN_P4 = True  # @param {type:"boolean"}
            RUN_P3 = True  # @param {type:"boolean"}
            RUN_P2 = True  # @param {type:"boolean"}
            RUN_P1 = True  # @param {type:"boolean"}
            RESUME_COMPLETED = True  # @param {type:"boolean"}
            MOUNT_DRIVE = False  # @param {type:"boolean"}
            DRIVE_CONTRACTS_PATH = ""  # @param {type:"string"}
            A2Z_INPUTS_URL = ""  # @param {type:"string"}
            CRISISMMA_TAR_URL = ""  # @param {type:"string"}
            AUTO_DOWNLOAD = True  # @param {type:"boolean"}
            """,
            cid="iccv-cell-1",
        ),
        _cell(
            f"""
            import base64, concurrent.futures, hashlib, importlib, io, os, shutil, site, subprocess, sys, zipfile
            from pathlib import Path
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Enable GPU runtime in Colab before executing ICCV Same-Bank benchmark.")

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            sys_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9) if hasattr(os, "sysconf") else 32.0
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if (Path("/dev/shm").exists() and sys_ram_gb >= 24.0) else (WS / "crisismmd_v2")
            print(f"[ENV] GPU={{torch.cuda.get_device_name(0)}} VRAM={{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}}GB RAM={{sys_ram_gb:.1f}}GB IMG_DIR={{IMG_DIR}}", flush=True)
            for d in (WS, CODE_DIR, IMG_DIR):
                d.mkdir(parents=True, exist_ok=True)

            CODE_SHA = {code_sha!r}
            CODE_B64 = {code_b64!r}
            raw_zip = base64.b64decode(CODE_B64)
            if hashlib.sha256(raw_zip).hexdigest() != CODE_SHA:
                raise ValueError("Embedded mnarm repo SHA mismatch!")
            if CODE_DIR.exists():
                shutil.rmtree(CODE_DIR)
            CODE_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
                zf.extractall(CODE_DIR)

            def _pip_deps():
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "datasets==3.2.0", "PyYAML", "scipy", "pillow", "matplotlib>=3.8.0", "huggingface_hub", "pytest", "wheel", "gdown"], check=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(CODE_DIR)], check=True)

            def _apt_deps():
                if shutil.which("aria2c") is None and shutil.which("apt-get"):
                    subprocess.run(["apt-get", "-qq", "update"], check=False)
                    subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz"], check=False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f1, f2 = pool.submit(_pip_deps), pool.submit(_apt_deps)
                f1.result(); f2.result()

            site.addsitedir(str(CODE_DIR / "src"))
            importlib.invalidate_caches()
            import mnarm
            print("[IMPORT OK]", mnarm.__file__, flush=True)
            subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(CODE_DIR), check=True)
            """,
            cid="iccv-cell-2",
        ),
        _cell(
            f"""
            import concurrent.futures, importlib, json, os, shutil, site, subprocess, sys, time, urllib.parse, zipfile
            from pathlib import Path
            import requests
            try:
                from google.colab import drive as colab_drive, files as colab_files
            except ImportError:
                colab_drive = colab_files = None

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            sys_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9) if hasattr(os, "sysconf") else 32.0
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if (Path("/dev/shm").exists() and sys_ram_gb >= 24.0) else (WS / "crisismmd_v2")
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
                    cand.unlink()

            CONTRACTS_ZIP = next((p for p in LOCAL_CANDIDATES if p.is_file() and zipfile.is_zipfile(p)), None)
            if CONTRACTS_ZIP is None and MOUNT_DRIVE and colab_drive is not None:
                colab_drive.mount("/content/drive")
                drive_cands = [Path(str(DRIVE_CONTRACTS_PATH).strip())] if str(DRIVE_CONTRACTS_PATH).strip() else []
                drive_cands.extend([Path("/content/drive/MyDrive/jcmil_benchmark_v1/JCMIL_A2Z_INPUTS.zip"), Path("/content/drive/MyDrive/crisismmd_p1_p4_contracts.zip")])
                CONTRACTS_ZIP = next((p for p in drive_cands if p.is_file() and zipfile.is_zipfile(p)), None)

            if CONTRACTS_ZIP is None and str(A2Z_INPUTS_URL).strip():
                url_in = str(A2Z_INPUTS_URL).strip()
                dst_dl = Path("/content/crisismmd_contracts_dl.zip")
                host_in = urllib.parse.urlparse(url_in).netloc.lower()
                is_drive = "drive" in host_in and host_in.endswith("google.com")
                if not is_drive and shutil.which("aria2c"):
                    subprocess.run(["aria2c", "-x16", "-s16", "-k1M", "--allow-overwrite=true", url_in, "-d", "/content", "-o", dst_dl.name], check=False)
                if not dst_dl.is_file() or not zipfile.is_zipfile(dst_dl):
                    dst_dl.unlink(missing_ok=True)
                    subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", url_in, "-O", str(dst_dl)] if is_drive else ["curl", "-fL", url_in, "-o", str(dst_dl)], check=True)
                CONTRACTS_ZIP = dst_dl

            if CONTRACTS_ZIP is None:
                if colab_files is None:
                    raise RuntimeError("No contract ZIP found and google.colab.files unavailable; provide A2Z_INPUTS_URL or local /content/crisismmd_p1_p4_contracts.zip.")
                print("Upload crisismmd_p1_p4_contracts.zip (~5.8 MB):")
                up = colab_files.upload()
                if not up:
                    raise RuntimeError("No contract ZIP uploaded.")
                nm, blob = next(iter(up.items()))
                CONTRACTS_ZIP = Path("/content") / Path(nm).name
                CONTRACTS_ZIP.write_bytes(blob)

            SEQ_DIR = extract_contracts_from_a2z_zip(CONTRACTS_ZIP, WS / "crisismmd_seq_inputs")
            manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
            if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"

            req_set = {{r["image"] for sp in ("train", "dev") for r in manifest["rows"][sp]}}
            req_imgs = sorted(p for p in req_set if not (IMG_DIR / p).is_file())
            if req_imgs:
                print(f"[HF DOWNLOAD] Fetching {{len(req_imgs)}} train/dev CrisisMMD images...", flush=True)
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
                session.mount("https://", adapter); session.mount("http://", adapter)

                def _fetch(rel_p):
                    dst = IMG_DIR / rel_p; dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + ".partial")
                    url = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{CRISISMMA_HF_REV}/" + urllib.parse.quote(rel_p, safe="/")
                    for attempt in range(5):
                        r = session.get(url, timeout=(20, 60))
                        if r.status_code == 429 or r.status_code >= 500:
                            time.sleep(min(60, 2 * (2 ** attempt))); continue
                        r.raise_for_status(); tmp.write_bytes(r.content); tmp.replace(dst); return
                    raise RuntimeError(f"Exhausted retries downloading {{rel_p}}")

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for idx, fut in enumerate(concurrent.futures.as_completed([pool.submit(_fetch, p) for p in req_imgs]), 1):
                        fut.result()
                        if idx % 500 == 0 or idx == len(req_imgs):
                            print(f"[HF DOWNLOAD] {{idx}}/{{len(req_imgs)}} images ready", flush=True)
            print("[DATA READY] train/dev images verified:", len(manifest["rows"]["train"]) + len(manifest["rows"]["dev"]))
            """,
            cid="iccv-cell-3",
        ),
        _cell(
            """
            import concurrent.futures, os, threading
            from collections import deque
            import torch
            from IPython.display import Image as IPyImage, Markdown, display
            try:
                from google.colab import files as colab_files
            except ImportError:
                colab_files = None

            OUT_DIR = WS / "outputs_iccv_same_bank"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            run_env = {**os.environ, "PYTHONPATH": str(CODE_DIR / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
            print_lock = threading.Lock()
            sel_protos = [p for flag, p in ((RUN_P4, "P4"), (RUN_P3, "P3"), (RUN_P2, "P2"), (RUN_P1, "P1")) if flag]
            if not sel_protos:
                raise RuntimeError("Enable at least one protocol.")

            def _run_job(tag, cfg_rel, out_sub, bb_name):
                out_path = OUT_DIR / out_sub
                out_path.mkdir(parents=True, exist_ok=True)
                log_path = OUT_DIR / f"{out_sub}.log"
                cmd = [
                    sys.executable, "-u", "-m", "mnarm.runners.iccv_same_bank",
                    "--config", str(CODE_DIR / cfg_rel),
                    "--seq-inputs", str(SEQ_DIR), "--images", str(IMG_DIR),
                    "--out", str(out_path), "--protocols", *sel_protos,
                ]
                if RESUME_COMPLETED:
                    cmd.append("--resume")
                tail = deque(maxlen=40)
                with log_path.open("w", encoding="utf8") as lf:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=run_env)
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        lf.write(line); tail.append(line.rstrip("\\n"))
                        with print_lock:
                            print(f"[{tag}] {line}", end="", flush=True)
                    rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"{tag} failed (exit={rc}). Last 40 log lines:\\n" + "\\n".join(tail))
                return out_sub, bb_name

            jobs = []
            if RUN_CLIP:
                jobs.append(("CLIP", "configs/iccv_crisismmd_clip.yaml", "clip_same_bank", "clip_b16"))
            if RUN_SIGLIP:
                jobs.append(("SIGLIP", "configs/iccv_crisismmd_siglip.yaml", "siglip_same_bank", "siglip_b16"))

            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            sys_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9) if hasattr(os, "sysconf") else 32.0
            gpu_name = torch.cuda.get_device_name(0)
            if PARALLEL_BACKBONES and len(jobs) > 1 and vram_gb >= 22.0 and sys_ram_gb >= 28.0:
                print(f"[PARALLEL] Running {', '.join(j[0] for j in jobs)} ({','.join(sel_protos)}) on {gpu_name} ({vram_gb:.1f} GB VRAM / {sys_ram_gb:.1f} GB RAM)", flush=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    for fut in [pool.submit(_run_job, *j) for j in jobs]:
                        fut.result()
            else:
                print(f"[SEQUENTIAL] Running on {gpu_name} ({vram_gb:.1f} GB VRAM / {sys_ram_gb:.1f} GB RAM)", flush=True)
                for j in jobs:
                    _run_job(*j)

            for _, _, out_sub, bb_name in jobs:
                sum_md = OUT_DIR / out_sub / f"ICCV_SAME_BANK_{bb_name}_SUMMARY.md"
                if sum_md.is_file():
                    display(Markdown(sum_md.read_text(encoding="utf8")))
                fig_dir = OUT_DIR / out_sub / "figures"
                for fig_png in sorted(fig_dir.glob("*.png")):
                    display(IPyImage(filename=str(fig_png)))

            RES_ZIP = Path("/content/ICCV_SAME_BANK_DEV_RESULTS.zip")
            with zipfile.ZipFile(RES_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for fp in sorted(OUT_DIR.rglob("*")):
                    if fp.is_file():
                        zout.write(fp, fp.relative_to(OUT_DIR).as_posix())
            print("[PACKAGED]", RES_ZIP, RES_ZIP.stat().st_size, "bytes")
            if AUTO_DOWNLOAD and colab_files is not None:
                colab_files.download(str(RES_ZIP))
            """,
            cid="iccv-cell-4",
        ),
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU"},
        "cells": cells,
    }
    nb_json = json.dumps(nb, indent=2, ensure_ascii=False)
    if "$" in nb_json:
        raise AssertionError("Forbidden $ in ICCV Same-Bank notebook")
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
