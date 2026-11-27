"""Build clean Colab Web notebook for unified ICCV Same-Bank evaluation (dev-Only, link/upload input)."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import tarfile
import textwrap
import warnings
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "colab" / "ICCV_SameBank_ZeroMNAR_ColabWeb.ipynb"
OUT_ZIP = ROOT / "colab" / "mnarm_iccv_same_bank_code.zip"
ASSETS_XZ = ROOT / "assets" / "crisismmd_contracts_json.tar.xz"
DEFAULT_GIT_URL = "https://github.com/Thanh-000/mm-vlm-runner.git"

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


def _ensure_contracts_xz() -> Path:
    zp = ROOT / "colab" / "crisismmd_p1_p4_contracts.zip"
    if zp.is_file() and not ASSETS_XZ.is_file():
        ASSETS_XZ.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as zf, tarfile.open(ASSETS_XZ, mode="w:xz", preset=9) as tf:
            for nm in ("CONTRACT_SHA256.json", "manifest.json", "PLANS.json", "partition_contract.json"):
                raw = zf.read(nm)
                ti = tarfile.TarInfo(nm)
                ti.size = len(raw)
                tf.addfile(ti, io.BytesIO(raw))
    return ASSETS_XZ


def build() -> None:
    code_sha, code_b64 = _pack_repo_zip(OUT_ZIP)
    _ensure_contracts_xz()
    cells = [
        _cell(
            """
            # ICCV Same-Bank Full-Suite Benchmark — `CSS-CT v2.2-fair` across 4 Multi-Altitude Benchmarks (`dev` + Confirmatory `test`)

            Runs **all 4 full datasets** (`CrisisMMD` 18,082 rows, `DMD` UCI #456, `IncidentsVQA` incidents1m, `DisasterM3` NeurIPS'25) x **2 backbones** (`CLIP-B/16`, `SigLIP-B/16`) x **4 protocols** (`P1–P4`) on the **EXACT SAME source expert bank**: **14 verbatim author merging operators** alongside **7 continual/VLM-CL/oracle tracks** (`seq_magmax`, `mg_clip`, `pal_rls`, `joint_oracle`, etc.) across **7 regimes** (`raw`, `gr_clip`, `unlab_center`, `ncm`, `oas_lda`, `zero_mnar` [`CSS-CT` Ours: 0 D_cal labels, 0 exemplars], `cal_bias` [100% D_cal]).

            - **Full-Dataset & Zero-Subsampling Execution:** Pre-warms HuggingFace models in parallel with **16-connection `aria2c -x16 -s16`** Drive/HF downloads into `/dev/shm`, verifies full split sizes (`0%` overlap) upfront, runs VRAM-safe parallel scheduling across Blackwell / A100 / L4, and saves an incremental checkpoint archive after every dataset.
            """,
            kind="markdown",
            cid="iccv-cell-0",
        ),
        _cell(
            f"""
            #@title Configuration {{ display-mode: "form" }}
            REPO_GIT_URL = {DEFAULT_GIT_URL!r}  # @param {{type:"string"}}
            REPO_BRANCH = "main"  # @param {{type:"string"}}
            CONFIRM_FREEZE_TEST = True  # @param {{type:"boolean"}}
            RUN_CRISISMMD = True  # @param {{type:"boolean"}}
            RUN_DMD = True  # @param {{type:"boolean"}}
            RUN_INCIDENTS_VQA = True  # @param {{type:"boolean"}}
            RUN_DISASTER_M3 = True  # @param {{type:"boolean"}}
            RUN_CLIP = True  # @param {{type:"boolean"}}
            RUN_SIGLIP = True  # @param {{type:"boolean"}}
            PARALLEL_BACKBONES = True  # @param {{type:"boolean"}}
            RUN_P4 = True  # @param {{type:"boolean"}}
            RUN_P3 = True  # @param {{type:"boolean"}}
            RUN_P2 = True  # @param {{type:"boolean"}}
            RUN_P1 = True  # @param {{type:"boolean"}}
            RESUME_COMPLETED = True  # @param {{type:"boolean"}}
            A2Z_INPUTS_URL = ""  # @param {{type:"string"}}
            CRISISMMA_TAR_URL = "https://drive.google.com/uc?id=1wOJzIezlrH-o2LmWHtcxv8KdAKL6a7K6&export=download"  # @param {{type:"string"}}
            DMD_ARCHIVE_URL = "https://drive.google.com/uc?id=1_1Bn0dHKqZ7sk4r2mVcShoQlvgZXcPL0&export=download"  # @param {{type:"string"}}
            INCIDENTS_VQA_ARCHIVE_URL = "https://drive.google.com/uc?id=1WUMyBQ68cMkhW3ONCzvPl2W1KpV1QPO5&export=download"  # @param {{type:"string"}}
            DISASTER_M3_ARCHIVE_URL = "https://drive.google.com/uc?id=1oknnyMNKCXXK3tTrVVD4M7XAx3VzLmiO&export=download"  # @param {{type:"string"}}
            AUTO_DOWNLOAD = True  # @param {{type:"boolean"}}
            """,
            cid="iccv-cell-1",
        ),
        _cell(
            """
            import concurrent.futures, importlib, os, shutil, site, subprocess, sys
            from pathlib import Path
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Enable GPU runtime in Colab before executing ICCV Same-Bank benchmark.")

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            sys_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9) if hasattr(os, "sysconf") else 32.0
            shm_free_gb = (shutil.disk_usage("/dev/shm").free / 1e9) if Path("/dev/shm").exists() else 0.0
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            DATA_ROOT = Path("/dev/shm/mnarm_data") if shm_free_gb >= 12.0 else (WS / "data")
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if (Path("/dev/shm/crisismmd_v2").is_dir() and any(Path("/dev/shm/crisismmd_v2").rglob("*.jpg"))) else (DATA_ROOT / "crisismmd_v2")
            print(f"[ENV] GPU={torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB RAM={sys_ram_gb:.1f}GB SHM_FREE={shm_free_gb:.1f}GB DATA_ROOT={DATA_ROOT}", flush=True)
            for d in (WS, DATA_ROOT, IMG_DIR):
                d.mkdir(parents=True, exist_ok=True)

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
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "datasets==3.2.0", "PyYAML", "scipy", "pillow", "matplotlib>=3.8.0", "huggingface_hub", "hf_transfer", "pytest", "wheel", "gdown"], check=True)
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
            pt_res = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(CODE_DIR), text=True, capture_output=True)
            print(pt_res.stdout, end="", flush=True)
            if pt_res.returncode != 0:
                if pt_res.stderr:
                    print(pt_res.stderr, flush=True)
                raise RuntimeError(f"pytest failed (exit={pt_res.returncode})")
            """,
            cid="iccv-cell-2",
        ),
        _cell(
            f"""
            import base64, concurrent.futures, hashlib, html, importlib, io, json, os, re, shutil, site, subprocess, sys, tarfile, time, urllib.parse, zipfile
            from pathlib import Path
            import requests
            try:
                from google.colab import files as colab_files
            except ImportError:
                colab_files = None

            WS = Path("/content/mnarm_ws")
            CODE_DIR = WS / "repo"
            shm_free_gb = (shutil.disk_usage("/dev/shm").free / 1e9) if Path("/dev/shm").exists() else 0.0
            DATA_ROOT = Path("/dev/shm/mnarm_data") if shm_free_gb >= 12.0 else (WS / "data")
            IMG_DIR = Path("/dev/shm/crisismmd_v2") if (Path("/dev/shm/crisismmd_v2").is_dir() and any(Path("/dev/shm/crisismmd_v2").rglob("*.jpg"))) else (DATA_ROOT / "crisismmd_v2")
            IMG_DIR.mkdir(parents=True, exist_ok=True)
            site.addsitedir(str(CODE_DIR / "src"))
            importlib.invalidate_caches()
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            from mnarm.data import (
                aria2_download_file,
                extract_contracts_from_a2z_zip,
                hf_lfs_batch_download,
                is_valid_binary_archive,
                load_benchmark_contracts,
                prepare_disaster_m3,
                prepare_dmd,
                prepare_incidents_vqa,
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

            def _prewarm_hf_models() -> None:
                try:
                    from huggingface_hub import snapshot_download
                    hf_skip = ["*.msgpack", "*.h5", "*.ot", "tf_model*"]
                    if RUN_CLIP:
                        snapshot_download("openai/clip-vit-base-patch16", revision="57c216476eefef5ab752ec549e440a49ae4ae5f3", ignore_patterns=hf_skip)
                    if RUN_SIGLIP:
                        snapshot_download("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", ignore_patterns=hf_skip)
                except Exception as exc_hf:
                    print(f"[WARN] HF snapshot pre-warm skipped: {{exc_hf}}", flush=True)

            hf_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            hf_fut = hf_pool.submit(_prewarm_hf_models)

            def _extract_drive_id(url_in: str) -> str:
                m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url_in)
                if m:
                    return m.group(1)
                q = urllib.parse.parse_qs(urllib.parse.urlparse(url_in).query)
                return q.get("id", [""])[0]

            def _resolve_drive_direct_url(url_in: str) -> tuple[str, str]:
                u = url_in.strip()
                host = urllib.parse.urlparse(u).netloc.lower()
                if "drive" not in host or not host.endswith("google.com"):
                    return u, ""
                uc_host = "drive." + "usercontent.google.com"
                if uc_host in host and "confirm=" in u and ("uuid=" in u or "at=" in u):
                    return u, ""
                fid = _extract_drive_id(u)
                if not fid:
                    return u, ""
                sess = requests.Session()
                sess.headers["User-Agent"] = "Mozilla/5.0"
                probe = f"https://{{uc_host}}/download?id={{fid}}&export=download&confirm=t"
                with sess.get(probe, stream=True, timeout=30, allow_redirects=True) as r:
                    ck = "; ".join(f"{{k}}={{v}}" for k, v in sess.cookies.get_dict().items())
                    ct = (r.headers.get("Content-Type") or "").lower()
                    if "text/html" not in ct:
                        return r.url, ck
                    body = r.text
                inputs = {{}}
                for tag in re.findall(r"<input[^>]+>", body, flags=re.I):
                    mn = re.search(r'name=["\\\']([^"\\\']+)["\\\']', tag, flags=re.I)
                    mv = re.search(r'value=["\\\']([^"\\\']*)["\\\']', tag, flags=re.I)
                    if mn and mv:
                        inputs[mn.group(1)] = html.unescape(mv.group(1))
                if inputs:
                    return f"https://{{uc_host}}/download?" + urllib.parse.urlencode(inputs), ck
                return probe, ck

            def _download_archive_aria2(url_in: str, dst: Path, min_bytes: int = 1000) -> Path:
                u = url_in.strip()
                if not u:
                    raise ValueError(f"Empty download URL for {{dst.name}}")
                dst.unlink(missing_ok=True)
                host_in = urllib.parse.urlparse(u).netloc.lower()
                is_drive = "drive" in host_in and host_in.endswith("google.com")
                direct_u, cookie_hdr = _resolve_drive_direct_url(u)
                hdrs = {{"Cookie": cookie_hdr}} if cookie_hdr else None
                try:
                    return aria2_download_file(direct_u, dst, headers=hdrs, min_bytes=min_bytes, max_conn=16)
                except Exception:
                    dst.unlink(missing_ok=True)
                    if is_drive:
                        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", u, "-O", str(dst)], check=True)
                    if not is_valid_binary_archive(dst, min_bytes):
                        raise
                    return dst

            def _unpack_to_dir(arc: Path, dest_dir: Path) -> None:
                dest_dir.mkdir(parents=True, exist_ok=True)
                with arc.open("rb") as fh:
                    magic = fh.read(4)
                if zipfile.is_zipfile(arc):
                    subprocess.run(["unzip", "-q", "-o", str(arc), "-d", str(dest_dir)], check=True)
                elif magic[:2] == b"\\x1f\\x8b":
                    if shutil.which("pigz"):
                        subprocess.run(["tar", "-I", "pigz", "-xf", str(arc), "-C", str(dest_dir)], check=True)
                    else:
                        subprocess.run(["tar", "-xzf", str(arc), "-C", str(dest_dir)], check=True)
                else:
                    subprocess.run(["tar", "-xf", str(arc), "-C", str(dest_dir)], check=True)

            DATASET_PATHS = {{}}

            if RUN_CRISISMMD:
                SEQ_DIR = WS / "crisismmd_seq_inputs"
                SEQ_DIR.mkdir(parents=True, exist_ok=True)
                req_json = ("CONTRACT_SHA256.json", "manifest.json", "PLANS.json", "partition_contract.json")
                xz_asset = CODE_DIR / "assets" / "crisismmd_contracts_json.tar.xz"
                if xz_asset.is_file() and any(not (SEQ_DIR / fn).is_file() for fn in req_json):
                    with tarfile.open(xz_asset, mode="r:xz") as tf:
                        tf.extractall(SEQ_DIR)

                if not (SEQ_DIR / "manifest.json").is_file():
                    LOCAL_CANDS = [Path("/content/crisismmd_p1_p4_contracts.zip"), Path("/content/JCMIL_A2Z_INPUTS.zip"), Path("/content/crisismmd_contracts_dl.zip")]
                    for cand in LOCAL_CANDS:
                        if cand.is_file() and not zipfile.is_zipfile(cand):
                            cand.unlink()
                    CONTRACTS_ZIP = next((p for p in LOCAL_CANDS if p.is_file() and zipfile.is_zipfile(p)), None)
                    if CONTRACTS_ZIP is None and str(A2Z_INPUTS_URL).strip():
                        CONTRACTS_ZIP = _download_archive_aria2(str(A2Z_INPUTS_URL), Path("/content/crisismmd_contracts_dl.zip"), min_bytes=1024)
                    if CONTRACTS_ZIP is None:
                        if colab_files is None:
                            raise RuntimeError("Provide A2Z_INPUTS_URL or local /content/crisismmd_p1_p4_contracts.zip.")
                        print("Upload crisismmd_p1_p4_contracts.zip (~5.8 MB):")
                        up = colab_files.upload()
                        nm, blob = next(iter(up.items()))
                        CONTRACTS_ZIP = Path("/content") / Path(nm).name
                        CONTRACTS_ZIP.write_bytes(blob)
                    SEQ_DIR = extract_contracts_from_a2z_zip(CONTRACTS_ZIP, SEQ_DIR)

                manifest = json.loads((SEQ_DIR / "manifest.json").read_bytes())
                if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                    IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"
                active_sp = ("train", "dev", "test") if CONFIRM_FREEZE_TEST else ("train", "dev")
                req_set = {{r["image"] for sp in active_sp for r in manifest["rows"][sp]}}
                has_splits = (SEQ_DIR / "splits.zip").is_file() or (IMG_DIR / "crisismmd_datasplit_all.zip").is_file() or (IMG_DIR.parent / "CrisisMMD_v2.0" / "crisismmd_datasplit_all.zip").is_file()
                if (not has_splits) or any(not (IMG_DIR / p).is_file() for p in req_set):
                    arc_cands = [Path("/content/crisismmd_v2.tar"), Path("/content/CrisisMMD_v2.0.tar.gz"), Path("/content/CrisisMMD_v2.0.zip")]
                    archive_path = next((p for p in arc_cands if p.is_file() and p.stat().st_size > 100_000_000), None)
                    tar_u = str(CRISISMMA_TAR_URL).strip()
                    if archive_path is None and tar_u:
                        try:
                            archive_path = _download_archive_aria2(tar_u, Path("/content/crisismmd_archive.dl"), min_bytes=100_000_000)
                        except Exception as exc_cm_dl:
                            print(f"[WARN] CrisisMMD Drive link failed ({{exc_cm_dl}}); trying QCRI/HF aria2c fallback.", flush=True)
                    if archive_path is None and shutil.which("aria2c"):
                        dl_qcri = Path("/content/crisismmd_qcri.tar.gz")
                        subprocess.run(["aria2c", "-x16", "-s16", "-k4M", "--connect-timeout=15", "--timeout=30", "--max-tries=2", "--allow-overwrite=true", "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz", "-d", "/content", "-o", dl_qcri.name], check=False)
                        if is_valid_binary_archive(dl_qcri, 100_000_000):
                            archive_path = dl_qcri
                    if archive_path is not None:
                        _unpack_to_dir(archive_path, IMG_DIR)
                        if archive_path.parent == Path("/content"):
                            archive_path.unlink(missing_ok=True)
                        if (IMG_DIR / "CrisisMMD_v2.0" / "data_image").is_dir():
                            IMG_DIR = IMG_DIR / "CrisisMMD_v2.0"

                if not (SEQ_DIR / "splits.zip").is_file():
                    split_src = next((p for p in (IMG_DIR / "crisismmd_datasplit_all.zip", IMG_DIR.parent / "crisismmd_datasplit_all.zip", IMG_DIR.parent / "CrisisMMD_v2.0" / "crisismmd_datasplit_all.zip") if p.is_file() and zipfile.is_zipfile(p)), None)
                    if split_src is not None:
                        shutil.copy2(split_src, SEQ_DIR / "splits.zip")
                if (SEQ_DIR / "CONTRACT_SHA256.json").is_file():
                    csha = json.loads((SEQ_DIR / "CONTRACT_SHA256.json").read_bytes())
                    for fn, exp in csha.items():
                        fp = SEQ_DIR / fn
                        if not fp.is_file() or hashlib.sha256(fp.read_bytes()).hexdigest() != exp:
                            raise RuntimeError(f"CrisisMMD contract verification failed for {{fn}}")

                if not CONFIRM_FREEZE_TEST:
                    test_only = {{r["image"] for r in manifest["rows"]["test"]}} - req_set
                    sum(1 for rel_t in test_only if (IMG_DIR / rel_t).is_file() and ((IMG_DIR / rel_t).unlink() or True))
                req_imgs = sorted(p for p in req_set if not (IMG_DIR / p).is_file())
                if req_imgs:
                    hf_lfs_batch_download("QCRI/CrisisMMD", "{CRISISMMA_HF_REV}", req_imgs, IMG_DIR, hf_token=hf_tok)
                DATASET_PATHS["crisismmd"] = (SEQ_DIR, IMG_DIR)
                print("[READY] crisismmd:", IMG_DIR, flush=True)

            def _prepare_from_link(stem: str, url_val: str, prep_fn, marker_check) -> Path:
                ds_dir = DATA_ROOT / stem
                if marker_check(ds_dir):
                    print(f"[REUSE] {{stem}} already unpacked at {{ds_dir}}", flush=True)
                    return ds_dir
                arc = next((p for p in (Path(f"/content/{{stem}}.tar"), Path(f"/content/{{stem}}.tar.gz"), Path(f"/content/{{stem}}.zip")) if is_valid_binary_archive(p, 1_000_000)), None)
                u_str = url_val.strip()
                if arc is None and u_str:
                    try:
                        arc = _download_archive_aria2(u_str, Path(f"/content/{{stem}}_dl.tar"), min_bytes=1_000_000)
                    except Exception as exc_dl:
                        print(f"[WARN] Archive link failed for {{stem}} ({{exc_dl}}); falling back to direct Web/HF aria2c CDN.", flush=True)
                        arc = None
                out_dir = prep_fn(ds_dir, archive_path=arc)
                if arc is not None and arc.parent == Path("/content"):
                    arc.unlink(missing_ok=True)
                return out_dir

            if RUN_DMD:
                dmd_dir = _prepare_from_link("dmd", str(DMD_ARCHIVE_URL), prepare_dmd, lambda d: any(d.rglob("*.txt")) and any(d.rglob("*.jpg")))
                DATASET_PATHS["dmd"] = (dmd_dir, dmd_dir)
                print("[READY] dmd:", dmd_dir, flush=True)

            if RUN_INCIDENTS_VQA:
                inc_dir = _prepare_from_link("incidents_vqa", str(INCIDENTS_VQA_ARCHIVE_URL), prepare_incidents_vqa, lambda d: any(d.rglob("disastervqa_annotations.jsonl")) and any(d.rglob("*.jpg")))
                DATASET_PATHS["incidents_vqa"] = (inc_dir, inc_dir)
                print("[READY] incidents_vqa:", inc_dir, flush=True)

            if RUN_DISASTER_M3:
                m3_dir = _prepare_from_link("disaster_m3", str(DISASTER_M3_ARCHIVE_URL), prepare_disaster_m3, lambda d: any(d.rglob("benchmark_release.json")) and any(d.rglob("*.png")))
                DATASET_PATHS["disaster_m3"] = (m3_dir, m3_dir)
                print("[READY] disaster_m3:", m3_dir, flush=True)

            hf_fut.result()
            hf_pool.shutdown(wait=True)
            for ds_name, (seq_p, img_p) in DATASET_PATHS.items():
                chk = load_benchmark_contracts(ds_name, seq_p, img_p, "P4", allow_test=bool(CONFIRM_FREEZE_TEST), seed=11)
                n_tr = sum(len(s["fit"]) for s in chk["stages"]) + len(chk["stages"][-1]["calibration_latest_views"])
                n_dv, n_ts = len(chk["dev"]), (len(chk["test"]) if chk.get("test") else 0)
                print(f"[FULL-DATASET VERIFIED] {{ds_name}}: train+cal={{n_tr}} dev={{n_dv}} test={{n_ts}} lookup_total={{len(chk['lookup'])}}", flush=True)
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
            run_env = {
                **os.environ,
                "USE_TF": "0",
                "USE_JAX": "0",
                "TF_CPP_MIN_LOG_LEVEL": "3",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "MNARM_TURBO_KERNELS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1" if not os.environ.get("HF_TOKEN") else "0"),
                "PYTHONPATH": str(CODE_DIR / "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            }

            print_lock = threading.Lock()
            sel_protos = [p for flag, p in ((RUN_P4, "P4"), (RUN_P3, "P3"), (RUN_P2, "P2"), (RUN_P1, "P1")) if flag]
            if not sel_protos or not DATASET_PATHS:
                raise RuntimeError("Enable at least one dataset and one protocol.")

            from mnarm.runners.iccv_report import write_ecosystem_master_summary

            class _OOMRetryError(RuntimeError):
                pass

            def _run_job(tag, cfg_rel, seq_p, img_p, out_sub, bb_name, protos_list=None, skip_suite=False, force_resume=False):
                out_path = OUT_DIR / out_sub
                out_path.mkdir(parents=True, exist_ok=True)
                p_args = list(protos_list or sel_protos)
                log_suffix = ("_" + "_".join(p_args).lower()) if skip_suite else ""
                log_path = OUT_DIR / (out_sub.replace("/", "_") + log_suffix + ".log")
                cmd = [
                    sys.executable, "-u", "-m", "mnarm.runners.iccv_same_bank",
                    "--config", str(CODE_DIR / cfg_rel),
                    "--seq-inputs", str(seq_p), "--images", str(img_p),
                    "--out", str(out_path), "--protocols", *p_args,
                ]
                if CONFIRM_FREEZE_TEST:
                    cmd.append("--confirm-freeze")
                if RESUME_COMPLETED or force_resume:
                    cmd.append("--resume")
                if skip_suite:
                    cmd.append("--skip-suite-summary")
                tail = deque(maxlen=80)
                with log_path.open("w", encoding="utf8") as lf:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=run_env)
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        lf.write(line); tail.append(line.rstrip("\\n"))
                        with print_lock:
                            print(f"[{tag}] {line}", end="", flush=True)
                    rc = proc.wait()
                if rc != 0:
                    tail_msg = "\\n".join(tail)
                    is_oom = any(k in tail_msg for k in ("OutOfMemoryError", "CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "memory mapping failed with OOM"))
                    if is_oom:
                        raise _OOMRetryError(f"{tag} OOM (exit={rc}):\\n{tail_msg}")
                    raise RuntimeError(f"{tag} failed (exit={rc}). Last lines:\\n{tail_msg}")
                return out_sub, bb_name

            def _pack_results_zip(target_zip: Path) -> None:
                tmp_z = target_zip.with_suffix(".tmp.zip")
                with zipfile.ZipFile(tmp_z, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                    for fp in sorted(OUT_DIR.rglob("*")):
                        if fp.is_file():
                            zout.write(fp, fp.relative_to(OUT_DIR).as_posix())
                tmp_z.replace(target_zip)

            vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            sys_ram_gib = ((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)) if hasattr(os, "sysconf") else 32.0
            gpu_name = torch.cuda.get_device_name(0)
            if not PARALLEL_BACKBONES:
                max_slots = 1
            elif vram_gib >= 88.0 and sys_ram_gib >= 60.0:
                max_slots = 8
            elif vram_gib >= 68.0 and sys_ram_gib >= 44.0:
                max_slots = 6
            elif vram_gib >= 36.0 and sys_ram_gib >= 28.0:
                max_slots = 4
            elif vram_gib >= 20.0 and sys_ram_gib >= 18.0:
                max_slots = 2
            else:
                max_slots = 1
            cpu_threads = str(max(2, min(6, (os.cpu_count() or 8) // max(max_slots, 1))))
            run_env["OMP_NUM_THREADS"] = cpu_threads
            run_env["MKL_NUM_THREADS"] = cpu_threads
            cost_map = {"crisismmd": 15.8, "dmd": 12.2, "disaster_m3": 10.0, "incidents_vqa": 7.0} if vram_gib >= 36.0 else {"crisismmd": 12.0, "dmd": 9.5, "disaster_m3": 7.5, "incidents_vqa": 5.5}
            budget_gib = max(16.0, vram_gib * 0.82)

            CKPT_ZIP = Path("/content/ICCV_CHECKPOINT_LATEST.zip")
            ds_order = [d for d in ("crisismmd", "dmd", "disaster_m3", "incidents_vqa") if d in DATASET_PATHS] + [d for d in DATASET_PATHS if d not in ("crisismmd", "dmd", "disaster_m3", "incidents_vqa")]
            ran_jobs, group_meta, rem_cnt, queue_tasks = [], {}, {}, []
            for ds_name in ds_order:
                seq_p, img_p = DATASET_PATHS[ds_name]
                ds_jobs = []
                if RUN_CLIP:
                    sub = "clip_same_bank" if ds_name == "crisismmd" else f"{ds_name}/clip_same_bank"
                    ds_jobs.append((f"{ds_name.upper()}:CLIP", f"configs/iccv_{ds_name}_clip.yaml", seq_p, img_p, sub, "clip_b16"))
                if RUN_SIGLIP:
                    sub = "siglip_same_bank" if ds_name == "crisismmd" else f"{ds_name}/siglip_same_bank"
                    ds_jobs.append((f"{ds_name.upper()}:SIGLIP", f"configs/iccv_{ds_name}_siglip.yaml", seq_p, img_p, sub, "siglip_b16"))
                ran_jobs.extend(ds_jobs)
                for tag, cfg_rel, sp2, ip2, sub, bb in ds_jobs:
                    k_grp = (ds_name, sub)
                    group_meta[k_grp] = {"tag": tag, "cfg": cfg_rel, "seq": sp2, "img": ip2, "sub": sub, "bb": bb}
                    rem_cnt[k_grp] = len(sel_protos)
                    for pr in sel_protos:
                        queue_tasks.append({"group": k_grp, "vram": cost_map.get(ds_name, 10.0), "retries": 0, "args": (f"{tag}:{pr}", cfg_rel, sp2, ip2, sub, bb, [pr], len(sel_protos) > 1, False)})

            try:
                if max_slots >= 2 and len(queue_tasks) > 1:
                    print(f"[VRAM-QUEUE] {len(queue_tasks)} tasks across {len(ds_order)} datasets | slots={max_slots} budget={budget_gib:.1f}/{vram_gib:.1f} GiB CPU_threads={cpu_threads}/worker ({gpu_name})", flush=True)
                    pending, in_flight, used_vram = deque(queue_tasks), {}, 0.0
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_slots) as pool:
                        while pending or in_flight:
                            launched = False
                            for idx in range(len(pending)):
                                t_cand = pending[idx]
                                c_gib = t_cand["vram"]
                                if len(in_flight) < max_slots and (len(in_flight) == 0 or used_vram + c_gib <= budget_gib):
                                    del pending[idx]
                                    used_vram += c_gib
                                    fut = pool.submit(_run_job, *t_cand["args"])
                                    in_flight[fut] = t_cand
                                    launched = True
                                    break
                            if launched:
                                continue
                            done_futs, _ = concurrent.futures.wait(in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                            for df in done_futs:
                                done_t = in_flight.pop(df)
                                used_vram = max(0.0, used_vram - done_t["vram"])
                                try:
                                    df.result()
                                except _OOMRetryError as exc_oom:
                                    if done_t.get("retries", 0) < 2:
                                        done_t["retries"] = done_t.get("retries", 0) + 1
                                        done_t["vram"] = budget_gib if done_t["retries"] >= 2 else min(budget_gib, round(done_t["vram"] * 1.4, 2))
                                        print(f"[OOM-RETRY] Re-queueing {done_t['args'][0]} with VRAM={done_t['vram']:.1f} GiB (attempt {done_t['retries']}/2)", flush=True)
                                        pending.appendleft(done_t)
                                        continue
                                    raise exc_oom
                                k_grp = done_t["group"]
                                rem_cnt[k_grp] -= 1
                                if rem_cnt[k_grp] == 0:
                                    g_meta = group_meta[k_grp]
                                    if len(sel_protos) > 1:
                                        _run_job(f"{g_meta['tag']}:SUMMARY", g_meta["cfg"], g_meta["seq"], g_meta["img"], g_meta["sub"], g_meta["bb"], sel_protos, False, True)
                                    write_ecosystem_master_summary(OUT_DIR)
                                    _pack_results_zip(CKPT_ZIP)
                                    print(f"[CHECKPOINT SAVED] {k_grp[0]}:{g_meta['bb']} -> {CKPT_ZIP} ({CKPT_ZIP.stat().st_size} bytes)", flush=True)
                else:
                    for ds_name in ds_order:
                        for tag, cfg_rel, sp2, ip2, sub, bb in ran_jobs:
                            if (sub == "clip_same_bank" or sub == "siglip_same_bank" if ds_name == "crisismmd" else sub.startswith(f"{ds_name}/")):
                                _run_job(tag, cfg_rel, sp2, ip2, sub, bb, sel_protos, False, False)
                        write_ecosystem_master_summary(OUT_DIR)
                        _pack_results_zip(CKPT_ZIP)
                        print(f"[CHECKPOINT SAVED] {ds_name} -> {CKPT_ZIP} ({CKPT_ZIP.stat().st_size} bytes)", flush=True)
            finally:
                if any(OUT_DIR.rglob("ICCV_PROTOCOL*RESULTS.json")):
                    try:
                        write_ecosystem_master_summary(OUT_DIR)
                        _pack_results_zip(CKPT_ZIP)
                        print(f"[FINALLY-CHECKPOINT] Preserved completed protocols -> {CKPT_ZIP} ({CKPT_ZIP.stat().st_size} bytes)", flush=True)
                    except Exception:
                        pass

            master_md = write_ecosystem_master_summary(OUT_DIR)
            display(Markdown(master_md.read_text(encoding="utf8")))
            for _, _, _, _, out_sub, bb_name in ran_jobs:
                suf = "_TEST" if CONFIRM_FREEZE_TEST else ""
                fig_sub = "figures_test" if CONFIRM_FREEZE_TEST else "figures"
                sum_md = OUT_DIR / out_sub / f"ICCV_SAME_BANK_{bb_name}{suf}_SUMMARY.md"
                if sum_md.is_file():
                    display(Markdown(sum_md.read_text(encoding="utf8")))
                for fig_png in sorted((OUT_DIR / out_sub / fig_sub).glob("*.png")):
                    display(IPyImage(filename=str(fig_png)))

            RES_ZIP = Path("/content/ICCV_FINAL_DEV_AND_TEST_RESULTS.zip" if CONFIRM_FREEZE_TEST else "/content/ICCV_SAME_BANK_DEV_RESULTS.zip")
            _pack_results_zip(RES_ZIP)
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
        raise AssertionError("Forbidden dollar character in ICCV Same-Bank notebook")
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            src_str = "".join(c["source"])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", SyntaxWarning)
                ast.parse(src_str)
                compile(src_str, f"{OUT_NB.name}:cell_{idx}", "exec")
                if w:
                    raise SyntaxError(f"SyntaxWarning in cell {idx}: {w[0].message}")
    OUT_NB.write_text(nb_json, encoding="utf8")
    print(f"Built {OUT_ZIP} ({len(base64.b64decode(code_b64))} B, sha={code_sha[:12]})")
    print(f"Built {OUT_NB} ({len(nb_json)} B)")


if __name__ == "__main__":
    build()
