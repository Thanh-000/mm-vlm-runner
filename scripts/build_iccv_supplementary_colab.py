"""Build standalone Colab Web notebook for the 3-piece ICCV supplementary suite."""

from __future__ import annotations

import ast
import json
import textwrap
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "colab" / "ICCV_Supplementary_3Piece_ColabWeb.ipynb"
DEFAULT_GIT_URL = "https://github.com/Thanh-000/mm-vlm-runner.git"


def _cell(src: str, kind: str = "code", cid: str = "") -> dict:
    clean = textwrap.dedent(src).lstrip("\n")
    if kind == "markdown":
        return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": clean.splitlines(keepends=True)}
    return {"id": cid, "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": clean.splitlines(keepends=True)}


def build() -> None:
    cells = [
        _cell(
            """
            # ICCV Supplementary 3-Piece Verification Suite (`CSS-CT v2.2-fair`)

            Runs **3 targeted, reviewer-proofing experiments** (~55–65 min total on A100-80GB) cleanly separated from the frozen main-benchmark results:
            1. **Piece 1 — Multi-Seed & Class-Order Permutation (`seeds = 11, 22, 33`):** Fixed `dev`/`test` splits with permuted continual class entry order -> `Mean ± Std` across 3 independent continual trajectories.
            2. **Piece 2 — Backbone Scaling (`CLIP ViT-L/14`, 428M) & Out-of-Domain Generalization (`ScienceQA-MM8`, 8 academic topics):** Auto-fetched via `aria2c` from official HuggingFace Parquets.
            3. **Piece 3 — Continuous MNAR Missingness Severity Sweep (`0%, 25%, 50%, 75%, 90%, 100%` on `DMD` `P3/P4`):** Quantifies phase-transition degradation and plots `fig_mnar_severity_sweep.pdf/.png`.
            """,
            kind="markdown",
            cid="supp-cell-0",
        ),
        _cell(
            f"""
            #@title Supplementary Suite Configuration {{ display-mode: "form" }}
            REPO_GIT_URL = {DEFAULT_GIT_URL!r}  # @param {{type:"string"}}
            REPO_BRANCH = "main"  # @param {{type:"string"}}
            CONFIRM_FREEZE_TEST = True  # @param {{type:"boolean"}}
            RUN_PIECE1_MULTISEED = True  # @param {{type:"boolean"}}
            SEEDS = "11,22,33"  # @param {{type:"string"}}
            INCLUDE_CRISISMMD_MULTISEED = False  # @param {{type:"boolean"}}
            RUN_PIECE2A_CLIP_L14 = True  # @param {{type:"boolean"}}
            RUN_PIECE2B_SCIENCEQA_OOD = True  # @param {{type:"boolean"}}
            RUN_PIECE3_MNAR_SWEEP = True  # @param {{type:"boolean"}}
            SWEEP_RATES = "0.0,0.25,0.50,0.75,0.90,1.0"  # @param {{type:"string"}}
            PARALLEL_WORKERS = True  # @param {{type:"boolean"}}
            RESUME_COMPLETED = True  # @param {{type:"boolean"}}
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
                raise RuntimeError("Enable GPU runtime in Colab before executing ICCV Supplementary Suite.")

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
            import concurrent.futures, html, json, os, re, shutil, site, subprocess, sys, tarfile, urllib.parse, zipfile
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
                    snapshot_download("openai/clip-vit-base-patch16", revision="57c216476eefef5ab752ec549e440a49ae4ae5f3", ignore_patterns=sk)
                    snapshot_download("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", ignore_patterns=sk)
                    if RUN_PIECE2A_CLIP_L14:
                        snapshot_download("openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41", ignore_patterns=sk)
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
            need_dis = bool(RUN_PIECE1_MULTISEED or RUN_PIECE2A_CLIP_L14 or RUN_PIECE3_MNAR_SWEEP)
            if need_dis:
                dmd_d = _prep_ds("dmd", str(DMD_ARCHIVE_URL), prepare_dmd, lambda d: any(d.rglob("*.txt")) and any(d.rglob("*.jpg")))
                DATASET_PATHS["dmd"] = (dmd_d, dmd_d)
                print("[READY] dmd:", dmd_d, flush=True)
            if RUN_PIECE1_MULTISEED or RUN_PIECE2A_CLIP_L14:
                inc_d = _prep_ds("incidents_vqa", str(INCIDENTS_VQA_ARCHIVE_URL), prepare_incidents_vqa, lambda d: any(d.rglob("disastervqa_annotations.jsonl")) and any(d.rglob("*.jpg")))
                m3_d = _prep_ds("disaster_m3", str(DISASTER_M3_ARCHIVE_URL), prepare_disaster_m3, lambda d: any(d.rglob("benchmark_release.json")) and any(d.rglob("*.png")))
                DATASET_PATHS["incidents_vqa"] = (inc_d, inc_d)
                DATASET_PATHS["disaster_m3"] = (m3_d, m3_d)
                print("[READY] incidents_vqa & disaster_m3", flush=True)
            if RUN_PIECE1_MULTISEED and INCLUDE_CRISISMMD_MULTISEED:
                SEQ_DIR = WS / "crisismmd_seq_inputs"; SEQ_DIR.mkdir(parents=True, exist_ok=True)
                with tarfile.open(CODE_DIR / "assets" / "crisismmd_contracts_json.tar.xz", mode="r:xz") as tf:
                    tf.extractall(SEQ_DIR)
                if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                    IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"
                manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
                active_sp = ("train", "dev", "test") if CONFIRM_FREEZE_TEST else ("train", "dev")
                req_set = {r["image"] for sp in active_sp for r in manifest["rows"][sp]}
                if any(not (IMG_DIR / p).is_file() for p in req_set):
                    arc = _dl_arc(str(CRISISMMA_TAR_URL), Path("/content/crisismmd_archive.dl"), min_b=100_000_000)
                    subprocess.run(["tar", "-xf", str(arc), "-C", str(IMG_DIR)], check=True)
                    arc.unlink(missing_ok=True)
                    if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                        IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"
                DATASET_PATHS["crisismmd"] = (SEQ_DIR, IMG_DIR)
                print("[READY] crisismmd:", IMG_DIR, flush=True)
            if RUN_PIECE2B_SCIENCEQA_OOD:
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
                from google.colab import files as colab_files
            except ImportError:
                colab_files = None

            SUPP_OUT = WS / "outputs_iccv_supplementary"
            SUPP_OUT.mkdir(parents=True, exist_ok=True)
            run_env = {
                **os.environ, "USE_TF": "0", "USE_JAX": "0", "TF_CPP_MIN_LOG_LEVEL": "3",
                "TOKENIZERS_PARALLELISM": "false", "MNARM_TURBO_KERNELS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1" if not os.environ.get("HF_TOKEN") else "0"),
                "PYTHONPATH": str(CODE_DIR / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            }
            print_lock = threading.Lock()
            from mnarm.runners.iccv_supplementary import write_supplementary_suite

            def _pack_zip(target: Path) -> None:
                tmp = target.with_suffix(".tmp.zip")
                with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                    for fp in sorted(SUPP_OUT.rglob("*")):
                        if fp.is_file():
                            zout.write(fp, fp.relative_to(SUPP_OUT).as_posix())
                tmp.replace(target)

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
            budget_gib = max(10.5, vram_gib * 0.88)

            seeds = [int(x.strip()) for x in str(SEEDS).split(",") if x.strip()]
            rates = [float(x.strip()) for x in str(SWEEP_RATES).split(",") if x.strip()]
            tasks, grp_rem, grp_info = [], {}, {}

            def _add_group(grp_key, cfg_rel, sp, ip, out_sub, protos, flags, vram_cost):
                grp_info[grp_key] = (cfg_rel, sp, ip, out_sub, protos, flags)
                grp_rem[grp_key] = len(protos)
                for pr in protos:
                    tasks.append({"group": grp_key, "vram": vram_cost, "args": (f"{grp_key}:{pr}", cfg_rel, sp, ip, out_sub, [pr], flags, len(protos) > 1)})

            if RUN_PIECE1_MULTISEED:
                ms_ds = ["dmd", "disaster_m3", "incidents_vqa"] + (["crisismmd"] if INCLUDE_CRISISMMD_MULTISEED else [])
                for s in seeds:
                    for ds in ms_ds:
                        if ds not in DATASET_PATHS:
                            continue
                        sp, ip = DATASET_PATHS[ds]
                        vc = 10.2 if ds == "crisismmd" else (6.5 if ds == "dmd" else (4.5 if ds == "disaster_m3" else 2.5))
                        for bb, bs in (("clip", "clip_b16"), ("siglip", "siglip_b16")):
                            fl = ["--seed", str(s), "--split-seed", "11", "--class-order-seed", str(s), "--mnar-missing-rate", "1.0"]
                            _add_group(f"p1_{ds}_s{s}_{bs}", f"configs/iccv_{ds}_{bb}.yaml", sp, ip, f"multiseed/{ds}/seed_{s}/{bs}_same_bank", ["P4", "P3", "P2", "P1"], fl, vc)

            if RUN_PIECE2A_CLIP_L14:
                for ds in ("dmd", "disaster_m3", "incidents_vqa"):
                    if ds not in DATASET_PATHS:
                        continue
                    sp, ip = DATASET_PATHS[ds]
                    vc = 9.5 if ds == "dmd" else (7.5 if ds == "disaster_m3" else 5.5)
                    _add_group(f"p2a_{ds}_l14", f"configs/iccv_{ds}_clip_l14.yaml", sp, ip, f"scaling_l14/{ds}/clip_l14_same_bank", ["P4", "P3", "P2", "P1"], ["--seed", "11", "--split-seed", "11"], vc)

            if RUN_PIECE2B_SCIENCEQA_OOD and "scienceqa_mm8" in DATASET_PATHS:
                sp, ip = DATASET_PATHS["scienceqa_mm8"]
                for bb, bs in (("clip", "clip_b16"), ("siglip", "siglip_b16")):
                    _add_group(f"p2b_sqa_{bs}", f"configs/iccv_scienceqa_mm8_{bb}.yaml", sp, ip, f"ood_general/scienceqa_mm8/{bs}_same_bank", ["P4", "P3", "P2", "P1"], ["--seed", "11", "--split-seed", "11"], 6.8)

            if RUN_PIECE3_MNAR_SWEEP and "dmd" in DATASET_PATHS:
                sp, ip = DATASET_PATHS["dmd"]
                for r in rates:
                    if abs(r - 1.0) < 1e-6 and RUN_PIECE1_MULTISEED and (11 in seeds):
                        continue
                    r_tag = f"rho_{int(round(r * 100)):03d}"
                    for bb, bs in (("clip", "clip_b16"), ("siglip", "siglip_b16")):
                        fl = ["--seed", "11", "--split-seed", "11", "--mnar-missing-rate", f"{r:.4f}"]
                        _add_group(f"p3_dmd_{r_tag}_{bs}", f"configs/iccv_dmd_{bb}.yaml", sp, ip, f"mnar_sweep/dmd/{r_tag}/{bs}_same_bank", ["P4", "P3"], fl, 6.5)

            CKPT_ZIP = Path("/content/ICCV_SUPP_CHECKPOINT_LATEST.zip")
            if max_slots >= 2 and len(tasks) > 1:
                print(f"[VRAM-QUEUE] {len(tasks)} supp tasks | slots={max_slots} budget={budget_gib:.1f}/{vram_gib:.1f} GiB CPU_threads={cpu_th}", flush=True)
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
                                _pack_zip(CKPT_ZIP)
                                print(f"[SUPP CHECKPOINT] {gk} -> {CKPT_ZIP} ({CKPT_ZIP.stat().st_size} B)", flush=True)
            else:
                for gk, (cfg_rel, sp, ip, out_sub, protos, flags) in grp_info.items():
                    _run_job(gk, cfg_rel, sp, ip, out_sub, protos, flags, False)
                    _pack_zip(CKPT_ZIP)

            if RUN_PIECE3_MNAR_SWEEP and any(abs(r - 1.0) < 1e-6 for r in rates) and RUN_PIECE1_MULTISEED and (11 in seeds):
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

            RES_ZIP = Path("/content/ICCV_SUPPLEMENTARY_3PIECE_RESULTS.zip")
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
        raise AssertionError("Forbidden dollar character in ICCV Supplementary notebook")
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", SyntaxWarning)
                ast.parse("".join(c["source"]))
                if w:
                    raise SyntaxError(f"SyntaxWarning in cell {idx}: {w[0].message}")
    OUT_NB.write_text(nb_json, encoding="utf8")
    print(f"Built {OUT_NB} ({len(nb_json)} B)")


if __name__ == "__main__":
    build()
