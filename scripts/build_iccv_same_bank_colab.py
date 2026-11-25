"""Build clean Colab Web notebook for unified ICCV Same-Bank evaluation across disaster benchmarks."""

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
            # ICCV Same-Bank Benchmark — `Canonical CSS-CT` v2 across Multi-Altitude Disaster Benchmarks (dev-Only)

            Runs all **14 verbatim author merging operators** alongside **6 continual/VLM-CL/oracle tracks** on the **EXACT SAME source expert bank** across **5 information regimes** (`raw`, `gr_clip`, `unlab_center`, `zero_mnar` [`CSS-CT` Ours: 0 D_cal labels, 0 exemplars], `cal_bias` [100% D_cal]) with **Blackwell / A100 High-VRAM Turbo Mode** (`VRAM >= 40 GB` GPU tensor pre-staging + batched GPU DOP + `use_fast=True`) across **4 zero-overlap disaster benchmarks**:
            1. `crisismmd` (CrisisMMD v2.0 — Ground social media)
            2. `dmd` (Damage Multimodal Dataset, UCI 456 — Ground damage inspection)
            3. `incidents_vqa` (DisasterVQA `incidents1m` subset — 0% CrisisMMD/MEDIC overlap)
            4. `disaster_m3` (DisasterM3 NeurIPS'25 — Orbital Optical + SAR satellite imagery)
            """,
            kind="markdown",
            cid="iccv-cell-0",
        ),
        _cell(
            """
            #@title Configuration { display-mode: "form" }
            RUN_CRISISMMD = True  # @param {type:"boolean"}
            RUN_DMD = False  # @param {type:"boolean"}
            RUN_INCIDENTS_VQA = False  # @param {type:"boolean"}
            RUN_DISASTER_M3 = False  # @param {type:"boolean"}
            RUN_CLIP = True  # @param {type:"boolean"}
            RUN_SIGLIP = True  # @param {type:"boolean"}
            PARALLEL_BACKBONES = True  # @param {type:"boolean"}
            RUN_P4 = True  # @param {type:"boolean"}
            RUN_P3 = True  # @param {type:"boolean"}
            RUN_P2 = True  # @param {type:"boolean"}
            RUN_P1 = True  # @param {type:"boolean"}
            RESUME_COMPLETED = True  # @param {type:"boolean"}
            MOUNT_DRIVE = False  # @param {type:"boolean"}
            CACHE_ARCHIVES_TO_DRIVE = False  # @param {type:"boolean"}
            DRIVE_CONTRACTS_PATH = ""  # @param {type:"string"}
            DRIVE_IMAGES_PATH = ""  # @param {type:"string"}
            DRIVE_SAVE_DIR = "/content/drive/MyDrive/mnarm_iccv_outputs"  # @param {type:"string"}
            A2Z_INPUTS_URL = ""  # @param {type:"string"}
            CRISISMMA_TAR_URL = ""  # @param {type:"string"}
            DMD_ZIP_URL = ""  # @param {type:"string"}
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
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            DATA_ROOT = Path("/dev/shm/mnarm_data") if (Path("/dev/shm").exists() and sys_ram_gb >= 24.0) else (WS / "data")
            IMG_DIR = DATA_ROOT / "crisismmd_v2"
            print(f"[ENV] GPU={{torch.cuda.get_device_name(0)}} VRAM={{vram_gb:.1f}}GB RAM={{sys_ram_gb:.1f}}GB DATA_ROOT={{DATA_ROOT}} TURBO_GPU_CACHE={{vram_gb >= 36.0}}", flush=True)
            for d in (WS, CODE_DIR, DATA_ROOT, IMG_DIR):
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
                need_bins = [b for b in ("aria2c", "pigz", "unzip") if shutil.which(b) is None]
                if need_bins and shutil.which("apt-get"):
                    subprocess.run(["apt-get", "-qq", "update"], check=False)
                    subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz", "unzip"], check=False)

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
            DATA_ROOT = Path("/dev/shm/mnarm_data") if (Path("/dev/shm").exists() and sys_ram_gb >= 24.0) else (WS / "data")
            IMG_DIR = DATA_ROOT / "crisismmd_v2"
            IMG_DIR.mkdir(parents=True, exist_ok=True)
            site.addsitedir(str(CODE_DIR / "src"))
            importlib.invalidate_caches()
            from mnarm.data import extract_contracts_from_a2z_zip, prepare_disaster_m3, prepare_dmd, prepare_incidents_vqa

            if MOUNT_DRIVE and colab_drive is not None and not Path("/content/drive/MyDrive").exists():
                colab_drive.mount("/content/drive")
            drv_cache = (Path(str(DRIVE_SAVE_DIR).strip()) / "datasets_cache") if (MOUNT_DRIVE and Path("/content/drive/MyDrive").exists() and str(DRIVE_SAVE_DIR).strip()) else None
            if drv_cache is not None and CACHE_ARCHIVES_TO_DRIVE:
                drv_cache.mkdir(parents=True, exist_ok=True)

            hf_tok = os.environ.get("HF_TOKEN", "").strip()
            if not hf_tok:
                try:
                    from google.colab import userdata as colab_userdata
                    hf_tok = (colab_userdata.get("HF_TOKEN") or "").strip()
                    if hf_tok:
                        os.environ["HF_TOKEN"] = hf_tok
                except Exception:
                    hf_tok = ""

            DATASET_PATHS = {{}}

            def _fetch_url_or_drive(url_in: str, dst: Path, min_bytes: int = 1000) -> Path | None:
                if not url_in.strip():
                    return None
                dst.unlink(missing_ok=True)
                host_in = urllib.parse.urlparse(url_in).netloc.lower()
                is_drive = "drive" in host_in and host_in.endswith("google.com")
                if not is_drive and shutil.which("aria2c"):
                    subprocess.run(["aria2c", "-x16", "-s16", "-k1M", "--allow-overwrite=true", url_in, "-d", str(dst.parent), "-o", dst.name], check=False)
                if not dst.is_file() or dst.stat().st_size <= min_bytes:
                    dst.unlink(missing_ok=True)
                    subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", url_in, "-O", str(dst)] if is_drive else ["curl", "-fL", url_in, "-o", str(dst)], check=True)
                return dst if (dst.is_file() and dst.stat().st_size > min_bytes) else None

            if RUN_CRISISMMD:
                LOCAL_CANDS = [Path("/content/crisismmd_p1_p4_contracts.zip"), Path("/content/JCMIL_A2Z_INPUTS.zip"), Path("/content/crisismmd_contracts_dl.zip")]
                for cand in LOCAL_CANDS:
                    if cand.is_file() and not zipfile.is_zipfile(cand):
                        cand.unlink()
                CONTRACTS_ZIP = next((p for p in LOCAL_CANDS if p.is_file() and zipfile.is_zipfile(p)), None)
                if CONTRACTS_ZIP is None and MOUNT_DRIVE and Path("/content/drive/MyDrive").exists():
                    drive_cands = [Path(str(DRIVE_CONTRACTS_PATH).strip())] if str(DRIVE_CONTRACTS_PATH).strip() else []
                    drive_cands.extend([Path("/content/drive/MyDrive/jcmil_benchmark_v1/JCMIL_A2Z_INPUTS.zip"), Path("/content/drive/MyDrive/crisismmd_p1_p4_contracts.zip")])
                    CONTRACTS_ZIP = next((p for p in drive_cands if p.is_file() and zipfile.is_zipfile(p)), None)
                if CONTRACTS_ZIP is None and str(A2Z_INPUTS_URL).strip():
                    CONTRACTS_ZIP = _fetch_url_or_drive(str(A2Z_INPUTS_URL), Path("/content/crisismmd_contracts_dl.zip"), min_bytes=1024)
                if CONTRACTS_ZIP is None:
                    if colab_files is None:
                        raise RuntimeError("No contract ZIP found; provide A2Z_INPUTS_URL or upload crisismmd_p1_p4_contracts.zip.")
                    print("Upload crisismmd_p1_p4_contracts.zip (~5.8 MB):")
                    up = colab_files.upload()
                    nm, blob = next(iter(up.items()))
                    CONTRACTS_ZIP = Path("/content") / Path(nm).name
                    CONTRACTS_ZIP.write_bytes(blob)

                SEQ_DIR = extract_contracts_from_a2z_zip(CONTRACTS_ZIP, WS / "crisismmd_seq_inputs")
                manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
                if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                    IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"
                req_set = {{r["image"] for sp in ("train", "dev") for r in manifest["rows"][sp]}}
                if any(not (IMG_DIR / p).is_file() for p in req_set):
                    arc_cands = [Path(str(DRIVE_IMAGES_PATH).strip())] if str(DRIVE_IMAGES_PATH).strip() else []
                    arc_cands.extend([Path("/content/CrisisMMD_v2.0.tar.gz"), Path("/content/CrisisMMD_v2.0.zip")])
                    if drv_cache is not None:
                        arc_cands.extend([drv_cache / "CrisisMMD_v2.0.tar.gz", drv_cache / "CrisisMMD_v2.0.zip"])
                    if Path("/content/drive/MyDrive").exists():
                        arc_cands.extend([Path("/content/drive/MyDrive/CrisisMMD_v2.0.tar.gz"), Path("/content/drive/MyDrive/jcmil_benchmark_v1/CrisisMMD_v2.0.tar.gz")])
                    archive_path = next((p for p in arc_cands if p.is_file() and p.stat().st_size > 100_000_000), None)
                    if archive_path is None and str(CRISISMMA_TAR_URL).strip():
                        archive_path = _fetch_url_or_drive(str(CRISISMMA_TAR_URL), Path("/content/crisismmd_archive.dl"), min_bytes=100_000_000)
                    if archive_path is None and shutil.which("aria2c"):
                        dl_qcri = Path("/content/crisismmd_qcri.tar.gz")
                        subprocess.run(["aria2c", "-x16", "-s16", "-k1M", "--connect-timeout=15", "--timeout=30", "--max-tries=2", "--allow-overwrite=true", "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz", "-d", "/content", "-o", dl_qcri.name], check=False)
                        if dl_qcri.is_file() and dl_qcri.stat().st_size > 100_000_000:
                            archive_path = dl_qcri
                    if archive_path is not None:
                        if zipfile.is_zipfile(archive_path):
                            subprocess.run(["unzip", "-q", "-o", str(archive_path), "-d", str(IMG_DIR)], check=True)
                        elif shutil.which("pigz"):
                            subprocess.run(["tar", "-I", "pigz", "-xf", str(archive_path), "-C", str(IMG_DIR)], check=True)
                        else:
                            subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(IMG_DIR)], check=True)
                        if drv_cache is not None and CACHE_ARCHIVES_TO_DRIVE and archive_path.parent == Path("/content") and not (drv_cache / "CrisisMMD_v2.0.tar.gz").is_file():
                            shutil.copy2(archive_path, drv_cache / "CrisisMMD_v2.0.tar.gz")
                        if archive_path.parent == Path("/content"):
                            archive_path.unlink(missing_ok=True)
                        if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                            IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"

                test_only = {{r["image"] for r in manifest["rows"]["test"]}} - req_set
                sum(1 for rel_t in test_only if (IMG_DIR / rel_t).is_file() and ((IMG_DIR / rel_t).unlink() or True))
                req_imgs = sorted(p for p in req_set if not (IMG_DIR / p).is_file())
                if req_imgs and shutil.which("aria2c"):
                    m_lines = []
                    for rel_p in req_imgs:
                        dst = IMG_DIR / rel_p; dst.parent.mkdir(parents=True, exist_ok=True)
                        m_lines.extend(["https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{CRISISMMA_HF_REV}/" + urllib.parse.quote(rel_p, safe="/"), f"  dir={{dst.parent}}", f"  out={{dst.name}}"])
                    mf = Path("/content/hf_aria2_manifest.txt"); mf.write_text("\\n".join(m_lines) + "\\n", encoding="utf8")
                    cmd = ["aria2c", f"--input-file={{mf}}", "-j", "16", "-x", "4", "-s", "4", "--retry-wait=5", "--max-tries=8", "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn"]
                    if hf_tok:
                        cmd.append(f"--header=Authorization: Bearer {{hf_tok}}")
                    subprocess.run(cmd, check=False); mf.unlink(missing_ok=True)
                req_imgs = sorted(p for p in req_set if not (IMG_DIR / p).is_file())
                if req_imgs:
                    sess = requests.Session()
                    if hf_tok:
                        sess.headers["Authorization"] = f"Bearer {{hf_tok}}"
                    def _fetch(rel_p):
                        dst = IMG_DIR / rel_p; dst.parent.mkdir(parents=True, exist_ok=True)
                        url = "https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/{CRISISMMA_HF_REV}/" + urllib.parse.quote(rel_p, safe="/")
                        for att in range(5):
                            r = sess.get(url, timeout=(20, 60))
                            if r.status_code == 429 or r.status_code >= 500:
                                time.sleep(min(60, 2 * (2 ** att))); continue
                            r.raise_for_status(); dst.write_bytes(r.content); return
                        raise RuntimeError(f"Failed {{rel_p}}")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        for fut in concurrent.futures.as_completed([pool.submit(_fetch, p) for p in req_imgs]):
                            fut.result()
                DATASET_PATHS["crisismmd"] = (SEQ_DIR, IMG_DIR)
                print("[READY] crisismmd:", IMG_DIR, flush=True)

            if RUN_DMD:
                dmd_arc = None
                if str(DMD_ZIP_URL).strip():
                    dmd_arc = _fetch_url_or_drive(str(DMD_ZIP_URL), Path("/content/dmd_archive.zip"), min_bytes=1_000_000)
                elif drv_cache is not None and (drv_cache / "dmd_uci_456.zip").is_file():
                    dmd_arc = drv_cache / "dmd_uci_456.zip"
                dmd_dir = prepare_dmd(DATA_ROOT / "dmd", archive_path=dmd_arc)
                DATASET_PATHS["dmd"] = (dmd_dir, dmd_dir)
                print("[READY] dmd:", dmd_dir, flush=True)

            if RUN_INCIDENTS_VQA:
                inc_dir = prepare_incidents_vqa(DATA_ROOT / "incidents_vqa")
                DATASET_PATHS["incidents_vqa"] = (inc_dir, inc_dir)
                print("[READY] incidents_vqa:", inc_dir, flush=True)

            if RUN_DISASTER_M3:
                m3_dir = prepare_disaster_m3(DATA_ROOT / "disaster_m3")
                DATASET_PATHS["disaster_m3"] = (m3_dir, m3_dir)
                print("[READY] disaster_m3:", m3_dir, flush=True)
            """,
            cid="iccv-cell-3",
        ),
        _cell(
            """
            import concurrent.futures, os, shutil, threading
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
            if not sel_protos or not DATASET_PATHS:
                raise RuntimeError("Enable at least one dataset and one protocol.")

            def _run_job(tag, cfg_rel, seq_p, img_p, out_sub, bb_name):
                out_path = OUT_DIR / out_sub
                out_path.mkdir(parents=True, exist_ok=True)
                log_path = OUT_DIR / (out_sub.replace("/", "_") + ".log")
                cmd = [
                    sys.executable, "-u", "-m", "mnarm.runners.iccv_same_bank",
                    "--config", str(CODE_DIR / cfg_rel),
                    "--seq-inputs", str(seq_p), "--images", str(img_p),
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
                    raise RuntimeError(f"{tag} failed (exit={rc}). Last 40 lines:\\n" + "\\n".join(tail))
                return out_sub, bb_name

            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            sys_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9) if hasattr(os, "sysconf") else 32.0
            gpu_name = torch.cuda.get_device_name(0)
            ran_jobs = []
            for ds_name, (seq_p, img_p) in DATASET_PATHS.items():
                ds_jobs = []
                if RUN_CLIP:
                    sub = "clip_same_bank" if ds_name == "crisismmd" else f"{ds_name}/clip_same_bank"
                    ds_jobs.append((f"{ds_name.upper()}:CLIP", f"configs/iccv_{ds_name}_clip.yaml", seq_p, img_p, sub, "clip_b16"))
                if RUN_SIGLIP:
                    sub = "siglip_same_bank" if ds_name == "crisismmd" else f"{ds_name}/siglip_same_bank"
                    ds_jobs.append((f"{ds_name.upper()}:SIGLIP", f"configs/iccv_{ds_name}_siglip.yaml", seq_p, img_p, sub, "siglip_b16"))
                if PARALLEL_BACKBONES and len(ds_jobs) > 1 and vram_gb >= 22.0 and sys_ram_gb >= 28.0:
                    print(f"[PARALLEL:{ds_name}] Running CLIP + SigLIP ({','.join(sel_protos)}) on {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        for fut in [pool.submit(_run_job, *j) for j in ds_jobs]:
                            fut.result()
                else:
                    for j in ds_jobs:
                        _run_job(*j)
                ran_jobs.extend(ds_jobs)

            for _, _, _, _, out_sub, bb_name in ran_jobs:
                sum_md = OUT_DIR / out_sub / f"ICCV_SAME_BANK_{bb_name}_SUMMARY.md"
                if sum_md.is_file():
                    display(Markdown(sum_md.read_text(encoding="utf8")))
                for fig_png in sorted((OUT_DIR / out_sub / "figures").glob("*.png")):
                    display(IPyImage(filename=str(fig_png)))

            RES_ZIP = Path("/content/ICCV_SAME_BANK_DEV_RESULTS.zip")
            with zipfile.ZipFile(RES_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for fp in sorted(OUT_DIR.rglob("*")):
                    if fp.is_file():
                        zout.write(fp, fp.relative_to(OUT_DIR).as_posix())
            print("[PACKAGED]", RES_ZIP, RES_ZIP.stat().st_size, "bytes")
            if MOUNT_DRIVE and Path("/content/drive/MyDrive").exists() and str(DRIVE_SAVE_DIR).strip():
                drv_out = Path(str(DRIVE_SAVE_DIR).strip())
                drv_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(RES_ZIP, drv_out / RES_ZIP.name)
                print("[DRIVE BACKUP]", drv_out / RES_ZIP.name, flush=True)
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
        raise AssertionError("Forbidden dollar character in ICCV Same-Bank notebook")
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
