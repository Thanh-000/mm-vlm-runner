"""Build standalone Colab Web notebook for Level-3 Full-Matrix ICCV evaluation (5 Datasets x 4 Backbones x 4 Protocols x 5 Seeds + MNAR Sweep)."""

from __future__ import annotations

import ast
import json
import textwrap
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB_ALL = ROOT / "colab" / "ICCV_Supplementary_3Piece_ColabWeb.ipynb"
OUT_NB_PART_A = ROOT / "colab" / "ICCV_FullMatrix_PartA_ColabWeb.ipynb"
OUT_NB_PART_B = ROOT / "colab" / "ICCV_FullMatrix_PartB_ColabWeb.ipynb"
DEFAULT_GIT_URL = "https://github.com/Thanh-000/mm-vlm-runner.git"


def _cell(src: str, kind: str = "code", cid: str = "") -> dict:
    clean = textwrap.dedent(src).lstrip("\n")
    if kind == "markdown":
        return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": clean.splitlines(keepends=True)}
    return {"id": cid, "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": clean.splitlines(keepends=True)}


def _build_notebook(out_nb: Path, *, part_tag: str, default_seeds: str, default_sweep: bool, subtitle: str) -> None:
    cells = [
        _cell(
            f"""
            # ICCV Level-3 Full-Matrix Suite — `{part_tag}` ({subtitle})

            Runs a clean, collision-free shard of the complete **`5 Datasets × 4 Backbones × 4 Protocols × 5 Seeds` + `MNAR Sweep`** matrix inside `[VRAM-QUEUE]`:
            - **5 Datasets:** `CrisisMMD` (18k), `DMD` (5.8k), `DisasterM3` (2.3k), `IncidentsVQA` (368), and out-of-domain **`ScienceQA-MM8`** (~10.3k, auto-fetched via `aria2c`).
            - **4 Backbones in Parallel:** `CLIP ViT-B/32`, `CLIP ViT-B/16`, `CLIP ViT-L/14` (428M), `SigLIP ViT-B/16`.
            - **Shard Configuration:** `PART_TAG = {part_tag!r}` | `SEEDS = {default_seeds!r}` | `RUN_MNAR_SWEEP = {default_sweep!r}` (isolated Drive & local checkpoint names so Part A and Part B run simultaneously on 2 Colab tabs with zero conflicts).
            """,
            kind="markdown",
            cid="supp-cell-0",
        ),
        _cell(
            f"""
            #@title Level-3 Full-Matrix Configuration ({part_tag}) {{ display-mode: "form" }}
            REPO_GIT_URL = {DEFAULT_GIT_URL!r}  # @param {{type:"string"}}
            REPO_BRANCH = "main"  # @param {{type:"string"}}
            PART_TAG = {part_tag!r}  # @param {{type:"string"}}
            CONFIRM_FREEZE_TEST = True  # @param {{type:"boolean"}}
            SEEDS = {default_seeds!r}  # @param {{type:"string"}}
            RUN_CRISISMMD = True  # @param {{type:"boolean"}}
            RUN_DMD = True  # @param {{type:"boolean"}}
            RUN_DISASTER_M3 = True  # @param {{type:"boolean"}}
            RUN_INCIDENTS_VQA = True  # @param {{type:"boolean"}}
            RUN_SCIENCEQA_OOD = True  # @param {{type:"boolean"}}
            RUN_CLIP_B32 = True  # @param {{type:"boolean"}}
            RUN_CLIP_B16 = True  # @param {{type:"boolean"}}
            RUN_CLIP_L14 = True  # @param {{type:"boolean"}}
            RUN_SIGLIP_B16 = True  # @param {{type:"boolean"}}
            RUN_MNAR_SWEEP = {default_sweep!r}  # @param {{type:"boolean"}}
            SWEEP_RATES = "0.0,0.25,0.50,0.75,0.90,1.0"  # @param {{type:"string"}}
            PARALLEL_WORKERS = True  # @param {{type:"boolean"}}
            RESUME_COMPLETED = True  # @param {{type:"boolean"}}
            SAVE_CHECKPOINT_TO_GDRIVE = False  # @param {{type:"boolean"}}
            CRISISMMA_TAR_URL = "https://drive.google.com/uc?id=1wOJzIezlrH-o2LmWHtcxv8KdAKL6a7K6&export=download"  # @param {{type:"string"}}
            DMD_ARCHIVE_URL = "https://drive.google.com/uc?id=1_1Bn0dHKqZ7sk4r2mVcShoQlvgZXcPL0&export=download"  # @param {{type:"string"}}
            INCIDENTS_VQA_ARCHIVE_URL = "https://drive.google.com/uc?id=1WUMyBQ68cMkhW3ONCzvPl2W1KpV1QPO5&export=download"  # @param {{type:"string"}}
            DISASTER_M3_ARCHIVE_URL = "https://drive.google.com/uc?id=1oknnyMNKCXXK3tTrVVD4M7XAx3VzLmiO&export=download"  # @param {{type:"string"}}
            AUTO_DOWNLOAD = True  # @param {{type:"boolean"}}
            """,
            cid="supp-cell-1",
        ),
        _cell(
            """
            import concurrent.futures, importlib, os, shutil, site, subprocess, sys
            from pathlib import Path
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Enable GPU runtime in Colab before executing ICCV Level-3 Full-Matrix Suite.")

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            shm_free_gb = (shutil.disk_usage("/dev/shm").free / 1e9) if Path("/dev/shm").exists() else 0.0
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            DATA_ROOT = Path("/dev/shm/mnarm_data") if shm_free_gb >= 12.0 else (WS / "data")
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if (Path("/dev/shm/crisismmd_v2").is_dir() and any(Path("/dev/shm/crisismmd_v2").rglob("*.jpg"))) else (DATA_ROOT / "crisismmd_v2")
            for d in (WS, DATA_ROOT, IMG_DIR):
                d.mkdir(parents=True, exist_ok=True)
            print(f"[ENV] GPU={torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB SHM_FREE={shm_free_gb:.1f}GB DATA_ROOT={DATA_ROOT}", flush=True)

            git_url = str(REPO_GIT_URL).strip()
            git_branch = str(REPO_BRANCH).strip() or "main"
            gh_tok = os.environ.get("GH_TOKEN", "").strip()
            if not gh_tok:
                try:
                    from google.colab import userdata as colab_userdata
                    gh_tok = (colab_userdata.get("GH_TOKEN") or "").strip()
                except Exception:
                    gh_tok = ""
            clone_url = git_url.replace("https://", f"https://{gh_tok}@", 1) if (gh_tok and git_url.startswith("https://")) else git_url

            if (CODE_DIR / ".git").is_dir():
                subprocess.run(["git", "-C", str(CODE_DIR), "fetch", "--depth", "1", "origin", git_branch], check=True)
                subprocess.run(["git", "-C", str(CODE_DIR), "reset", "--hard", f"origin/{git_branch}"], check=True)
            else:
                if CODE_DIR.exists():
                    shutil.rmtree(CODE_DIR)
                subprocess.run(["git", "clone", "--depth", "1", "--branch", git_branch, clone_url, str(CODE_DIR)], check=True)

            def _pip_deps():
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "datasets==3.2.0", "pyarrow", "PyYAML", "scipy", "pillow", "matplotlib>=3.8.0", "huggingface_hub", "hf_transfer", "pytest", "wheel", "gdown"], check=True)
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
            pt_res = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(CODE_DIR), text=True, capture_output=True)
            print(pt_res.stdout, end="", flush=True)
            if pt_res.returncode != 0:
                if pt_res.stderr:
                    print(pt_res.stderr, flush=True)
                raise RuntimeError(f"pytest failed (exit={pt_res.returncode})")
            """,
            cid="supp-cell-2",
        ),
        _cell(
            """
            import concurrent.futures, html, json, os, re, shutil, site, subprocess, sys, tarfile, urllib.parse
            from pathlib import Path
            import requests

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            shm_free_gb = (shutil.disk_usage("/dev/shm").free / 1e9) if Path("/dev/shm").exists() else 0.0
            DATA_ROOT = Path("/dev/shm/mnarm_data") if shm_free_gb >= 12.0 else (WS / "data")
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if (Path("/dev/shm/crisismmd_v2").is_dir() and any(Path("/dev/shm/crisismmd_v2").rglob("*.jpg"))) else (DATA_ROOT / "crisismmd_v2")
            site.addsitedir(str(CODE_DIR / "src"))
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            from mnarm.data import (
                aria2_download_file,
                hf_lfs_batch_download,
                is_valid_binary_archive,
                load_benchmark_contracts,
                prepare_disaster_m3,
                prepare_dmd,
                prepare_incidents_vqa,
                prepare_scienceqa_mm8,
            )

            hf_tok = os.environ.get("HF_TOKEN", "").strip()
            if not hf_tok:
                try:
                    from google.colab import userdata as colab_userdata
                    hf_tok = (colab_userdata.get("HF_TOKEN") or "").strip()
                    if hf_tok:
                        os.environ["HF_TOKEN"] = hf_tok
                except Exception:
                    hf_tok = ""
            if not hf_tok:
                os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

            def _prewarm_hf() -> None:
                try:
                    from huggingface_hub import snapshot_download
                    sk = ["*.msgpack", "*.h5", "*.ot", "tf_model*"]
                    if RUN_CLIP_B32:
                        snapshot_download("openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", ignore_patterns=sk)
                    if RUN_CLIP_B16:
                        snapshot_download("openai/clip-vit-base-patch16", revision="57c216476eefef5ab752ec549e440a49ae4ae5f3", ignore_patterns=sk)
                    if RUN_CLIP_L14:
                        snapshot_download("openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41", ignore_patterns=sk)
                    if RUN_SIGLIP_B16:
                        snapshot_download("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", ignore_patterns=sk)
                except Exception as exc_hf:
                    print(f"[WARN] HF snapshot pre-warm skipped: {exc_hf}", flush=True)

            hf_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            hf_fut = hf_pool.submit(_prewarm_hf)

            def _resolve_drive(u: str) -> tuple[str, str]:
                u = u.strip()
                host = urllib.parse.urlparse(u).netloc.lower()
                if "drive" not in host or not host.endswith("google.com"):
                    return u, ""
                uc_host = "drive." + "usercontent.google.com"
                m = re.search(r"/file/d/([A-Za-z0-9_-]+)", u)
                fid = m.group(1) if m else urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("id", [""])[0]
                if not fid:
                    return u, ""
                sess = requests.Session(); sess.headers["User-Agent"] = "Mozilla/5.0"
                probe = f"https://{uc_host}/download?id={fid}&export=download&confirm=t"
                with sess.get(probe, stream=True, timeout=30, allow_redirects=True) as r:
                    ck = "; ".join(f"{k}={v}" for k, v in sess.cookies.get_dict().items())
                    if "text/html" not in (r.headers.get("Content-Type") or "").lower():
                        return r.url, ck
                    body = r.text
                ins = {}
                for tag in re.findall(r"<input[^>]+>", body, flags=re.I):
                    mn = re.search(r'name=["\\\']([^"\\\']+)["\\\']', tag, flags=re.I)
                    mv = re.search(r'value=["\\\']([^"\\\']*)["\\\']', tag, flags=re.I)
                    if mn and mv:
                        ins[mn.group(1)] = html.unescape(mv.group(1))
                return (f"https://{uc_host}/download?" + urllib.parse.urlencode(ins), ck) if ins else (probe, ck)

            def _dl_arc(u: str, dst: Path, min_b: int = 1_000_000) -> Path:
                dst.unlink(missing_ok=True)
                du, ck = _resolve_drive(u)
                try:
                    return aria2_download_file(du, dst, headers=({"Cookie": ck} if ck else None), min_bytes=min_b, max_conn=16)
                except Exception:
                    dst.unlink(missing_ok=True)
                    subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", u, "-O", str(dst)], check=True)
                    return dst

            def _prep_ds(stem: str, url_v: str, prep_fn, chk_fn) -> Path:
                d = DATA_ROOT / stem
                if chk_fn(d):
                    return d
                arc = next((p for p in (Path(f"/content/{stem}.tar"), Path(f"/content/{stem}.tar.gz"), Path(f"/content/{stem}.zip")) if is_valid_binary_archive(p, 1_000_000)), None)
                if arc is None and url_v.strip():
                    try:
                        arc = _dl_arc(url_v.strip(), Path(f"/content/{stem}_dl.tar"))
                    except Exception as exc_d:
                        print(f"[WARN] {stem} Drive download failed ({exc_d}); falling back to direct mirror.", flush=True)
                        arc = None
                out_d = prep_fn(d, archive_path=arc)
                if arc is not None and arc.parent == Path("/content"):
                    arc.unlink(missing_ok=True)
                return out_d

            DATASET_PATHS = {}
            if RUN_CRISISMMD:
                SEQ_DIR = WS / "crisismmd_seq_inputs"; SEQ_DIR.mkdir(parents=True, exist_ok=True)
                with tarfile.open(CODE_DIR / "assets" / "crisismmd_contracts_json.tar.xz", mode="r:xz") as tf:
                    tf.extractall(SEQ_DIR)
                if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                    IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"
                manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
                active_sp = ("train", "dev", "test") if CONFIRM_FREEZE_TEST else ("train", "dev")
                req_set = {r["image"] for sp in active_sp for r in manifest["rows"][sp]}
                has_sp = (SEQ_DIR / "splits.zip").is_file() or (IMG_DIR / "crisismmd_datasplit_all.zip").is_file() or (IMG_DIR.parent / "CrisisMMD_v2.0" / "crisismmd_datasplit_all.zip").is_file()
                if (not has_sp) or any(not (IMG_DIR / p).is_file() for p in req_set):
                    arc = next((p for p in (Path("/content/crisismmd_v2.tar"), Path("/content/CrisisMMD_v2.0.tar.gz"), Path("/content/CrisisMMD_v2.0.zip")) if p.is_file() and p.stat().st_size > 100_000_000), None)
                    if arc is None and str(CRISISMMA_TAR_URL).strip():
                        try:
                            arc = _dl_arc(str(CRISISMMA_TAR_URL), Path("/content/crisismmd_archive.dl"), min_b=100_000_000)
                        except Exception as exc_cm:
                            print(f"[WARN] CrisisMMD Drive link failed ({exc_cm}); trying QCRI/HF mirror.", flush=True)
                    if arc is None and shutil.which("aria2c"):
                        dl_q = Path("/content/crisismmd_qcri.tar.gz")
                        subprocess.run(["aria2c", "-x16", "-s16", "-k4M", "--connect-timeout=15", "--timeout=30", "--max-tries=2", "--allow-overwrite=true", "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz", "-d", "/content", "-o", dl_q.name], check=False)
                        if is_valid_binary_archive(dl_q, 100_000_000):
                            arc = dl_q
                    if arc is not None:
                        subprocess.run(["tar", "-xf", str(arc), "-C", str(IMG_DIR)], check=True)
                        arc.unlink(missing_ok=True)
                        if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                            IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"
                if not (SEQ_DIR / "splits.zip").is_file():
                    sp_src = next((p for p in (IMG_DIR / "crisismmd_datasplit_all.zip", IMG_DIR.parent / "crisismmd_datasplit_all.zip", IMG_DIR.parent / "CrisisMMD_v2.0" / "crisismmd_datasplit_all.zip") if p.is_file()), None)
                    if sp_src is not None:
                        shutil.copy2(sp_src, SEQ_DIR / "splits.zip")
                req_imgs = sorted(p for p in req_set if not (IMG_DIR / p).is_file())
                if req_imgs:
                    hf_lfs_batch_download("QCRI/CrisisMMD", "10a5626ba112ab9c50369c473a2062bcf205d708", req_imgs, IMG_DIR, hf_token=hf_tok)
                DATASET_PATHS["crisismmd"] = (SEQ_DIR, IMG_DIR)
                print("[READY] crisismmd:", IMG_DIR, flush=True)
            if RUN_DMD or RUN_MNAR_SWEEP:
                dmd_d = _prep_ds("dmd", str(DMD_ARCHIVE_URL), prepare_dmd, lambda d: any(d.rglob("*.txt")) and any(d.rglob("*.jpg")))
                DATASET_PATHS["dmd"] = (dmd_d, dmd_d)
                print("[READY] dmd:", dmd_d, flush=True)
            if RUN_INCIDENTS_VQA:
                inc_d = _prep_ds("incidents_vqa", str(INCIDENTS_VQA_ARCHIVE_URL), prepare_incidents_vqa, lambda d: any(d.rglob("disastervqa_annotations.jsonl")) and any(d.rglob("*.jpg")))
                DATASET_PATHS["incidents_vqa"] = (inc_d, inc_d)
                print("[READY] incidents_vqa:", inc_d, flush=True)
            if RUN_DISASTER_M3:
                m3_d = _prep_ds("disaster_m3", str(DISASTER_M3_ARCHIVE_URL), prepare_disaster_m3, lambda d: any(d.rglob("benchmark_release.json")) and any(d.rglob("*.png")))
                DATASET_PATHS["disaster_m3"] = (m3_d, m3_d)
                print("[READY] disaster_m3:", m3_d, flush=True)
            if RUN_SCIENCEQA_OOD:
                sqa_d = prepare_scienceqa_mm8(DATA_ROOT / "scienceqa_mm8")
                DATASET_PATHS["scienceqa_mm8"] = (sqa_d, sqa_d)
                print("[READY] scienceqa_mm8:", sqa_d, flush=True)

            hf_fut.result(); hf_pool.shutdown(wait=True)
            for ds_name, (sp, ip) in DATASET_PATHS.items():
                chk = load_benchmark_contracts(ds_name, sp, ip, "P4", allow_test=bool(CONFIRM_FREEZE_TEST), seed=11)
                n_tr = sum(len(s["fit"]) for s in chk["stages"]) + len(chk["stages"][-1]["calibration_latest_views"])
                print(f"[VERIFIED] {ds_name}: train+cal={n_tr} dev={len(chk['dev'])} test={len(chk['test']) if chk.get('test') else 0}", flush=True)
            """,
            cid="supp-cell-3",
        ),
        _cell(
            """
            import concurrent.futures, os, shutil, subprocess, sys, threading, zipfile
            from collections import deque
            from pathlib import Path
            import torch
            from IPython.display import Image as IPyImage, Markdown, display
            try:
                from google.colab import drive as colab_drive, files as colab_files
            except ImportError:
                colab_drive, colab_files = None, None

            part_tag_norm = (str(PART_TAG).strip().upper() or "ALL").replace(" ", "_")
            SUPP_OUT = WS / f"outputs_iccv_{part_tag_norm.lower()}"
            SUPP_OUT.mkdir(parents=True, exist_ok=True)
            run_env = {
                **os.environ, "USE_TF": "0", "USE_JAX": "0", "TF_CPP_MIN_LOG_LEVEL": "3",
                "TOKENIZERS_PARALLELISM": "false", "MNARM_TURBO_KERNELS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1" if not os.environ.get("HF_TOKEN") else "0"),
                "PYTHONPATH": str(CODE_DIR / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            }
            print_lock = threading.Lock()
            from mnarm.runners.iccv_same_bank import _is_v21_valid
            from mnarm.runners.iccv_supplementary import write_supplementary_suite

            seeds = [int(x.strip()) for x in str(SEEDS).split(",") if x.strip()]
            CKPT_ZIP = Path(f"/content/ICCV_SUPP_{part_tag_norm}_CHECKPOINT.zip")
            DRIVE_CKPT = None
            if SAVE_CHECKPOINT_TO_GDRIVE and colab_drive is not None:
                colab_drive.mount("/content/drive", force_remount=False)
                DRIVE_CKPT = Path(f"/content/drive/MyDrive/ICCV_FULL_MATRIX_{part_tag_norm}_CHECKPOINT.zip")

            def _restore_zip_into(zp: Path, dst_root: Path, tag_label: str) -> None:
                if zp.is_file() and zipfile.is_zipfile(zp):
                    with zipfile.ZipFile(zp, "r") as zin:
                        zin.extractall(dst_root)
                    print(f"[RESTORED CHECKPOINT] {tag_label}: {zp}", flush=True)

            if RESUME_COMPLETED:
                for z_cand in (DRIVE_CKPT, CKPT_ZIP):
                    if z_cand is not None:
                        _restore_zip_into(z_cand, SUPP_OUT, part_tag_norm)
                if 11 in seeds:
                    for main_z in (Path("/content/ICCV_FINAL_DEV_AND_TEST_RESULTS.zip"), Path("/content/ICCV_FINAL_DEV_AND_TEST_RESULTS (1).zip")):
                        if main_z.is_file() and zipfile.is_zipfile(main_z):
                            imported = 0
                            with zipfile.ZipFile(main_z, "r") as zin:
                                for nm in zin.namelist():
                                    if nm.endswith(".json") and ("_same_bank/" in nm):
                                        parts = nm.split("/")
                                        if parts[0] in ("clip_same_bank", "siglip_same_bank"):
                                            ds_m, bb_f, rest_p = "crisismmd", ("clip_b16_same_bank" if parts[0] == "clip_same_bank" else "siglip_b16_same_bank"), "/".join(parts[1:])
                                        elif len(parts) >= 3 and parts[1] in ("clip_same_bank", "siglip_same_bank"):
                                            ds_m, bb_f, rest_p = parts[0], ("clip_b16_same_bank" if parts[1] == "clip_same_bank" else "siglip_b16_same_bank"), "/".join(parts[2:])
                                        else:
                                            continue
                                        out_f = SUPP_OUT / "multiseed" / ds_m / "seed_11" / bb_f / rest_p
                                        if not out_f.is_file():
                                            out_f.parent.mkdir(parents=True, exist_ok=True)
                                            out_f.write_bytes(zin.read(nm))
                                            imported += 1
                            if imported:
                                print(f"[IMPORTED SEED-11] Reused {imported} JSONs from {main_z.name}", flush=True)

            def _pack_zip(target: Path) -> None:
                tmp = target.with_suffix(".tmp.zip")
                with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                    for fp in sorted(SUPP_OUT.rglob("*")):
                        if fp.is_file():
                            zout.write(fp, fp.relative_to(SUPP_OUT).as_posix())
                tmp.replace(target)
                if DRIVE_CKPT is not None and target == CKPT_ZIP:
                    dtmp = DRIVE_CKPT.with_suffix(".tmp.zip")
                    shutil.copy2(target, dtmp)
                    dtmp.replace(DRIVE_CKPT)

            def _run_job(tag, cfg_rel, seq_p, img_p, out_dir_sub, protos, extra_flags, skip_suite=False):
                out_p = SUPP_OUT / out_dir_sub
                out_p.mkdir(parents=True, exist_ok=True)
                log_p = out_p / ("job_" + "_".join(protos).lower() + ".log")
                cmd = [
                    sys.executable, "-u", "-m", "mnarm.runners.iccv_same_bank",
                    "--config", str(CODE_DIR / cfg_rel), "--seq-inputs", str(seq_p), "--images", str(img_p),
                    "--out", str(out_p), "--protocols", *protos, *extra_flags,
                ]
                if CONFIRM_FREEZE_TEST:
                    cmd.append("--confirm-freeze")
                if RESUME_COMPLETED:
                    cmd.append("--resume")
                if skip_suite:
                    cmd.append("--skip-suite-summary")
                tail = deque(maxlen=35)
                with log_p.open("w", encoding="utf8") as lf:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=run_env)
                    assert proc.stdout is not None
                    for ln in proc.stdout:
                        lf.write(ln); tail.append(ln.rstrip("\\n"))
                        with print_lock:
                            print(f"[{tag}] {ln}", end="", flush=True)
                    rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"{tag} failed (exit={rc}):\\n" + "\\n".join(tail))

            vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            sys_ram_gib = ((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)) if hasattr(os, "sysconf") else 32.0
            max_slots = 1 if not PARALLEL_WORKERS else (8 if (vram_gib >= 88 and sys_ram_gib >= 60) else (6 if (vram_gib >= 68 and sys_ram_gib >= 44) else (4 if (vram_gib >= 36 and sys_ram_gib >= 28) else (2 if vram_gib >= 20 else 1))))
            cpu_th = str(max(2, min(6, (os.cpu_count() or 8) // max(max_slots, 1))))
            run_env["OMP_NUM_THREADS"] = cpu_th; run_env["MKL_NUM_THREADS"] = cpu_th
            budget_gib = max(12.5, vram_gib * 0.88)

            vram_tbl = {
                "clip_b32": {"crisismmd": 6.5, "scienceqa_mm8": 5.5, "dmd": 4.5, "disaster_m3": 3.5, "incidents_vqa": 2.2},
                "clip_b16": {"crisismmd": 10.2, "scienceqa_mm8": 8.2, "dmd": 6.5, "disaster_m3": 4.5, "incidents_vqa": 2.5},
                "siglip_b16": {"crisismmd": 10.2, "scienceqa_mm8": 8.2, "dmd": 6.5, "disaster_m3": 4.5, "incidents_vqa": 2.5},
                "clip_l14": {"crisismmd": 12.8, "scienceqa_mm8": 11.2, "dmd": 9.2, "disaster_m3": 7.5, "incidents_vqa": 5.5},
            }
            bb_specs = [
                (bb_id, cfg_suf)
                for flag, bb_id, cfg_suf in (
                    (RUN_CLIP_B32, "clip_b32", "clip_b32"),
                    (RUN_CLIP_B16, "clip_b16", "clip"),
                    (RUN_CLIP_L14, "clip_l14", "clip_l14"),
                    (RUN_SIGLIP_B16, "siglip_b16", "siglip"),
                )
                if flag
            ]
            ds_order = [d for d in ("crisismmd", "dmd", "scienceqa_mm8", "disaster_m3", "incidents_vqa") if d in DATASET_PATHS]
            rates = [float(x.strip()) for x in str(SWEEP_RATES).split(",") if x.strip()]
            tasks, grp_rem, grp_info, skipped_cnt = [], {}, {}, [0]

            def _proto_done(out_sub: str, bb_id: str, pr: str) -> bool:
                if not RESUME_COMPLETED:
                    return False
                p_dir = SUPP_OUT / out_sub / f"{bb_id}_{pr.lower()}"
                ok_d = _is_v21_valid(p_dir / "ICCV_PROTOCOL_RESULTS.json") is not None
                ok_t = (not CONFIRM_FREEZE_TEST) or (_is_v21_valid(p_dir / "ICCV_PROTOCOL_TEST_RESULTS.json") is not None)
                return ok_d and ok_t

            def _add_group(grp_key, cfg_rel, sp, ip, out_sub, bb_id, protos, flags, vram_cost):
                rem = [pr for pr in protos if not _proto_done(out_sub, bb_id, pr)]
                skipped_cnt[0] += len(protos) - len(rem)
                if not rem:
                    return
                grp_info[grp_key] = (cfg_rel, sp, ip, out_sub, protos, flags)
                grp_rem[grp_key] = len(rem)
                for pr in rem:
                    tasks.append({"group": grp_key, "vram": vram_cost, "args": (f"{grp_key}:{pr}", cfg_rel, sp, ip, out_sub, [pr], flags, len(protos) > 1)})

            for idx_s, s in enumerate(seeds):
                for ds in ds_order:
                    sp, ip = DATASET_PATHS[ds]
                    for bb_id, cfg_suf in bb_specs:
                        fl = ["--seed", str(s), "--split-seed", "11", "--class-order-seed", str(s), "--mnar-missing-rate", "1.0"]
                        _add_group(f"s{s}_{ds}_{bb_id}", f"configs/iccv_{ds}_{cfg_suf}.yaml", sp, ip, f"multiseed/{ds}/seed_{s}/{bb_id}_same_bank", bb_id, ["P4", "P3", "P2", "P1"], fl, vram_tbl[bb_id][ds])
                if idx_s == 0 and RUN_MNAR_SWEEP and "dmd" in DATASET_PATHS:
                    sp_d, ip_d = DATASET_PATHS["dmd"]
                    for r in rates:
                        if abs(r - 1.0) < 1e-6 and (11 in seeds) and RUN_DMD:
                            continue
                        r_tag = f"rho_{int(round(r * 100)):03d}"
                        for sw_bb, sw_suf in (("clip_b16", "clip"), ("siglip_b16", "siglip")):
                            fl = ["--seed", "11", "--split-seed", "11", "--mnar-missing-rate", f"{r:.4f}"]
                            _add_group(f"sweep_dmd_{r_tag}_{sw_bb}", f"configs/iccv_dmd_{sw_suf}.yaml", sp_d, ip_d, f"mnar_sweep/dmd/{r_tag}/{sw_bb}_same_bank", sw_bb, ["P4", "P3"], fl, 6.5)

            if skipped_cnt[0]:
                print(f"[RESUME] Skipped {skipped_cnt[0]} already-verified protocol jobs.", flush=True)
            if max_slots >= 2 and len(tasks) > 1:
                print(f"[VRAM-QUEUE] {len(tasks)} pending tasks | slots={max_slots} budget={budget_gib:.1f}/{vram_gib:.1f} GiB CPU_threads={cpu_th}", flush=True)
                pending, in_flight, used = deque(tasks), {}, 0.0
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_slots) as pool:
                    while pending or in_flight:
                        launched = False
                        for i in range(len(pending)):
                            cand = pending[i]
                            if len(in_flight) < max_slots and (len(in_flight) == 0 or used + cand["vram"] <= budget_gib):
                                del pending[i]; used += cand["vram"]
                                in_flight[pool.submit(_run_job, *cand["args"])] = cand
                                launched = True; break
                        if launched:
                            continue
                        done, _ = concurrent.futures.wait(in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                        for df in done:
                            dt = in_flight.pop(df); used = max(0.0, used - dt["vram"]); df.result()
                            gk = dt["group"]; grp_rem[gk] -= 1
                            if grp_rem[gk] == 0:
                                cfg_rel, sp, ip, out_sub, protos, flags = grp_info[gk]
                                if len(protos) > 1:
                                    _run_job(f"{gk}:SUMMARY", cfg_rel, sp, ip, out_sub, protos, flags, False)
                                write_supplementary_suite(SUPP_OUT)
                                _pack_zip(CKPT_ZIP)
                                print(f"[CHECKPOINT SAVED] {gk} -> {CKPT_ZIP} ({CKPT_ZIP.stat().st_size} B)", flush=True)
            else:
                for gk, (cfg_rel, sp, ip, out_sub, protos, flags) in grp_info.items():
                    _run_job(gk, cfg_rel, sp, ip, out_sub, protos, flags, False)
                    write_supplementary_suite(SUPP_OUT)
                    _pack_zip(CKPT_ZIP)

            if RUN_MNAR_SWEEP and any(abs(r - 1.0) < 1e-6 for r in rates) and (11 in seeds):
                for bs in ("clip_b16", "siglip_b16"):
                    for pr in ("p3", "p4"):
                        src_d = SUPP_OUT / "multiseed" / "dmd" / "seed_11" / f"{bs}_same_bank" / f"{bs}_{pr}"
                        dst_d = SUPP_OUT / "mnar_sweep" / "dmd" / "rho_100" / f"{bs}_same_bank" / f"{bs}_{pr}"
                        if src_d.is_dir():
                            shutil.copytree(src_d, dst_d, dirs_exist_ok=True)
                print("[DEDUP] Linked multiseed/dmd/seed_11 (P3, P4) -> mnar_sweep/dmd/rho_100", flush=True)

            master_md = write_supplementary_suite(SUPP_OUT)
            display(Markdown(master_md.read_text(encoding="utf8")))
            fig_png = SUPP_OUT / "figures" / "fig_mnar_severity_sweep.png"
            if fig_png.is_file():
                display(IPyImage(filename=str(fig_png)))

            RES_ZIP = Path(f"/content/ICCV_FULL_MATRIX_{part_tag_norm}_RESULTS.zip")
            _pack_zip(RES_ZIP)
            print("[PACKAGED]", RES_ZIP, RES_ZIP.stat().st_size, "bytes", flush=True)
            if AUTO_DOWNLOAD and colab_files is not None:
                colab_files.download(str(RES_ZIP))
            """,
            cid="supp-cell-4",
        ),
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU"},
        "cells": cells,
    }
    nb_json = json.dumps(nb, indent=2, ensure_ascii=False)
    if "$" in nb_json:
        raise AssertionError(f"Forbidden dollar character in {out_nb.name}")
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            src_str = "".join(c["source"])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", SyntaxWarning)
                ast.parse(src_str)
                compile(src_str, f"{out_nb.name}:cell_{idx}", "exec")
                if w:
                    raise SyntaxError(f"SyntaxWarning in {out_nb.name} cell {idx}: {w[0].message}")
    out_nb.write_text(nb_json, encoding="utf8")
    print(f"Built {out_nb} ({len(nb_json)} B)")


def build() -> None:
    _build_notebook(OUT_NB_PART_A, part_tag="PART_A", default_seeds="11,22", default_sweep=True, subtitle="Seeds 11, 22 + MNAR Sweep")
    _build_notebook(OUT_NB_PART_B, part_tag="PART_B", default_seeds="33,44,55", default_sweep=False, subtitle="Seeds 33, 44, 55")
    _build_notebook(OUT_NB_ALL, part_tag="ALL", default_seeds="11,22,33,44,55", default_sweep=True, subtitle="Full 5-Seed Single Session")


if __name__ == "__main__":
    build()
