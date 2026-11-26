"""Build standalone high-bandwidth Colab-to-Drive .tar stager notebook using Google Drive API v3."""

from __future__ import annotations

import ast
import json
import textwrap
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "colab" / "Stage_Disaster_Datasets_To_Drive.ipynb"


def _cell(src: str, kind: str = "code", cid: str = "") -> dict:
    clean = textwrap.dedent(src).lstrip("\n")
    if kind == "markdown":
        return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": clean.splitlines(keepends=True)}
    return {"id": cid, "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": clean.splitlines(keepends=True)}


def build() -> None:
    cells = [
        _cell(
            """
            # High-Bandwidth Disaster Dataset-to-Drive `.tar` Stager (Colab -> Drive API v3 Direct Push)

            Tối đa hoá băng thông Google Cloud của Colab (**chạy tốt trên `CPU + High-RAM` để tiết kiệm Compute Units, hoặc `G4` nếu muốn cổng mạng `50 Gbps` tối đa**) để kéo các bộ dữ liệu thảm hoạ, làm sạch `0%` trùng lặp, đóng `.tar` không nén, đẩy thẳng lên **Google Drive API v3 (`64 MB` resumable chunks)** và **tự động xuất link tải dùng ngay**:

            - **Hugging Face (`DisasterM3`, `DisasterVQA-Incidents`, `CrisisMMD`, `hf://...`):** Dùng **Git-LFS Batch API (`us.aws.cdn.hf.co`) + `aria2c -j 32 -x 8` + tự hạ luồng khi quá tải (`-j 6 -x 2`) + `SHA-256`**, vượt hoàn toàn giới hạn `/resolve/` và tránh DNS chết.
            - **Archive HTTP (`DMD UCI #456`, `CIFAR-100`, `EXTRA_DATASET_SPECS_JSON`):** Dùng **`aria2c -x 16 -s 16` + tự động backoff `-x 4 -s 4`**, sau đó đóng `.tar` không nén, đẩy thẳng lên **Google Drive API v3 (`64 MB` chunks)**, kiểm tra **HTTP `206 Partial Content` (`Accept-Ranges: bytes`)** và in sẵn các biến `*_ARCHIVE_URL` dùng cho `aria2c`.
            """,
            kind="markdown",
            cid="stage-cell-0",
        ),
        _cell(
            """
            #@title Dataset Staging Configuration { display-mode: "form" }
            DRIVE_FOLDER_NAME = "mnarm_iccv_datasets"  # @param {type:"string"}
            MAKE_SHAREABLE_LINKS = True  # @param {type:"boolean"}
            STAGE_DMD = True  # @param {type:"boolean"}
            STAGE_INCIDENTS_VQA = True  # @param {type:"boolean"}
            STAGE_DISASTER_M3 = True  # @param {type:"boolean"}
            STAGE_CRISISMMD = False  # @param {type:"boolean"}
            STAGE_CIFAR100 = True  # @param {type:"boolean"}
            OVERWRITE_EXISTING = False  # @param {type:"boolean"}
            CUSTOM_DMD_URL = ""  # @param {type:"string"}
            CUSTOM_INCIDENTS_ZIP_URL = ""  # @param {type:"string"}
            CUSTOM_CRISISMMD_TAR_URL = ""  # @param {type:"string"}
            CUSTOM_CIFAR100_URL = ""  # @param {type:"string"}
            EXTRA_DATASET_SPECS_JSON = ""  # @param {type:"string"}
            """,
            cid="stage-cell-1",
        ),
        _cell(
            """
            import os, shutil, subprocess
            from pathlib import Path
            from google.colab import auth
            from googleapiclient.discovery import build

            if shutil.which("aria2c") is None and shutil.which("apt-get"):
                subprocess.run(["apt-get", "-qq", "update"], check=False)
                subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz", "unzip"], check=False)

            auth.authenticate_user()
            DRIVE_SVC = build("drive", "v3", cache_discovery=False)

            folder_name = str(DRIVE_FOLDER_NAME).strip() or "mnarm_iccv_datasets"
            q_folder = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and 'root' in parents and trashed = false"
            found = DRIVE_SVC.files().list(q=q_folder, fields="files(id, name)").execute().get("files", [])
            if found:
                DRIVE_FOLDER_ID = found[0]["id"]
            else:
                meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": ["root"]}
                DRIVE_FOLDER_ID = DRIVE_SVC.files().create(body=meta, fields="id").execute()["id"]

            shm_free_gb = shutil.disk_usage("/dev/shm").free / 1e9 if Path("/dev/shm").exists() else 0.0
            root_free_gb = shutil.disk_usage("/content").free / 1e9
            STAGE_ROOT = Path("/dev/shm/mnarm_stage_tmp") if shm_free_gb >= 10.0 else Path("/content/mnarm_stage_tmp")
            STAGE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[READY] Drive folder='{folder_name}' (id={DRIVE_FOLDER_ID}) | STAGE_ROOT={STAGE_ROOT} | /dev/shm={shm_free_gb:.1f}GB | /content={root_free_gb:.1f}GB", flush=True)
            """,
            cid="stage-cell-2",
        ),
        _cell(
            """
            import concurrent.futures, hashlib, html, json, os, re, shutil, subprocess, time, urllib.parse, urllib.request
            from pathlib import Path
            import requests
            from googleapiclient.http import MediaFileUpload

            try:
                from google.colab import userdata as colab_userdata
                if not os.environ.get("HF_TOKEN"):
                    tok_u = (colab_userdata.get("HF_TOKEN") or "").strip()
                    if tok_u:
                        os.environ["HF_TOKEN"] = tok_u
            except Exception:
                pass

            HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
            INCIDENTS_REV = "f341ce34814a93bad8abfe73da898a309116a59e"
            M3_REV = "1da6da73fad6bd100454c099edd0cee972823810"
            CRISIS_REV = "10a5626ba112ab9c50369c473a2062bcf205d708"
            DMD_UCI_URL = "https://archive.ics.uci.edu/static/public/456/multimodal+damage+identification+for+humanitarian+computing.zip"
            CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            CIFAR100_MD5 = "md5=eb9058c3a382ffc7106e4002c42a8d85"

            def _digest_file(path: Path, alg: str = "sha256") -> str:
                h = hashlib.new(alg)
                with path.open("rb") as fh:
                    while True:
                        buf = fh.read(16 * 1024 * 1024)
                        if not buf:
                            break
                        h.update(buf)
                return h.hexdigest()

            def _is_valid_binary(p: Path, min_bytes: int = 128) -> bool:
                if not p.is_file() or p.stat().st_size < min_bytes:
                    return False
                with p.open("rb") as fh:
                    head = fh.read(256).lower()
                return (b"<html" not in head) and (b"<!doctype" not in head)

            def _aria2_fetch_with_backoff(url: str, dst: Path, *, checksum: str = "", min_bytes: int = 1024, max_conn: int = 16) -> Path:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.unlink(missing_ok=True)
                ctrl = dst.with_name(dst.name + ".aria2")
                ctrl.unlink(missing_ok=True)
                cs = checksum.strip().lower()
                cs_ok = (lambda p: True) if not cs else (lambda p: _digest_file(p, "sha256" if cs.startswith(("sha-256=", "sha256=")) else cs.split("=", 1)[0]) == cs.split("=", 1)[1])

                def _try_aria2(conn: int, wait_s: int, tries: int) -> None:
                    cmd = [
                        "aria2c", f"-x{conn}", f"-s{conn}", "-k1M", "--min-split-size=1M",
                        "--file-allocation=none", "--optimize-concurrent-downloads=true",
                        "--connect-timeout=15", "--timeout=45", f"--retry-wait={wait_s}", f"--max-tries={tries}",
                        "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn",
                        "-d", str(dst.parent), "-o", dst.name, url,
                    ]
                    if cs.startswith(("md5=", "sha-256=")):
                        cmd.append(f"--checksum={cs}")
                    subprocess.run(cmd, check=False)

                if shutil.which("aria2c"):
                    _try_aria2(max_conn, 2, 3)
                    if (not _is_valid_binary(dst, min_bytes) or not cs_ok(dst)) and max_conn > 4:
                        dst.unlink(missing_ok=True); ctrl.unlink(missing_ok=True)
                        print(f"  [BACKOFF] Retrying {dst.name} with aria2c -x4 -s4", flush=True)
                        _try_aria2(4, 5, 4)

                if not _is_valid_binary(dst, min_bytes) or not cs_ok(dst):
                    dst.unlink(missing_ok=True); ctrl.unlink(missing_ok=True)
                    subprocess.run(["curl", "-fL", "--retry", "4", url, "-o", str(dst)], check=False)
                if not _is_valid_binary(dst, min_bytes) or not cs_ok(dst):
                    dst.unlink(missing_ok=True)
                    raise RuntimeError(f"Download failed or failed binary/checksum check for {url}")
                return dst

            def _unpack_archive(arc: Path, out_dir: Path) -> None:
                out_dir.mkdir(parents=True, exist_ok=True)
                with arc.open("rb") as fh:
                    magic = fh.read(4)
                if magic[:2] == b"PK":
                    subprocess.run(["unzip", "-q", "-o", str(arc), "-d", str(out_dir)], check=True)
                elif magic[:2] == b"\\x1f\\x8b":
                    if shutil.which("pigz"):
                        subprocess.run(["tar", "-I", "pigz", "-xf", str(arc), "-C", str(out_dir)], check=True)
                    else:
                        subprocess.run(["tar", "-xzf", str(arc), "-C", str(out_dir)], check=True)
                else:
                    subprocess.run(["tar", "-xf", str(arc), "-C", str(out_dir)], check=True)

            def _verify_drive_aria2_range(file_id: str) -> bool:
                uc_host = "drive." + "usercontent.google.com"
                sess = requests.Session()
                sess.headers["User-Agent"] = "Mozilla/5.0"
                probe = f"https://{uc_host}/download?id={file_id}&export=download&confirm=t"
                try:
                    with sess.get(probe, stream=True, timeout=20, allow_redirects=True) as r:
                        ct = (r.headers.get("Content-Type") or "").lower()
                        target_u = r.url
                        if "text/html" in ct:
                            inputs = {}
                            for tag in re.findall(r"<input[^>]+>", r.text, flags=re.I):
                                mn = re.search(r'name=["\\\']([^"\\\']+)["\\\']', tag, flags=re.I)
                                mv = re.search(r'value=["\\\']([^"\\\']*)["\\\']', tag, flags=re.I)
                                if mn and mv:
                                    inputs[mn.group(1)] = html.unescape(mv.group(1))
                            if inputs:
                                target_u = f"https://{uc_host}/download?" + urllib.parse.urlencode(inputs)
                    with sess.get(target_u, headers={"Range": "bytes=0-1023"}, stream=True, timeout=20, allow_redirects=True) as rr:
                        return rr.status_code == 206
                except Exception:
                    return False

            def _find_drive_file(name: str) -> dict | None:
                q = f"name = '{name}' and '{DRIVE_FOLDER_ID}' in parents and trashed = false"
                res = DRIVE_SVC.files().list(q=q, fields="files(id, name, size)").execute().get("files", [])
                return res[0] if res else None

            def _share_if_requested(file_id: str) -> None:
                if MAKE_SHAREABLE_LINKS:
                    DRIVE_SVC.permissions().create(fileId=file_id, body={"type": "anyone", "role": "reader"}, fields="id").execute()

            def _lfs_batch_aria2(repo_id: str, rev: str, rel_paths: list[str] | None, dest_dir: Path) -> None:
                dest_dir.mkdir(parents=True, exist_ok=True)
                req_hdrs = {"User-Agent": "Mozilla/5.0"}
                if HF_TOKEN:
                    req_hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
                t0 = time.perf_counter()
                with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo_id}?blobs=1", headers=req_hdrs), timeout=30) as r:
                    siblings = json.loads(r.read()).get("siblings", [])
                meta_map = {str(s["rfilename"]): s for s in siblings}
                want_paths = sorted(set(r.replace("\\\\", "/").lstrip("/") for r in (rel_paths if rel_paths is not None else [s["rfilename"] for s in siblings])))

                lfs_items, direct_items = [], []
                for rel in want_paths:
                    dst = dest_dir / rel
                    meta = meta_map.get(rel, {})
                    lfs = meta.get("lfs")
                    if isinstance(lfs, dict) and lfs.get("sha256") and lfs.get("size"):
                        oid, sz = str(lfs["sha256"]), int(lfs["size"])
                        if dst.is_file() and dst.stat().st_size == sz:
                            continue
                        lfs_items.append((rel, oid, sz, dst))
                    else:
                        if dst.is_file() and meta.get("size") and dst.stat().st_size == int(meta["size"]):
                            continue
                        direct_items.append((f"https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{urllib.parse.quote(rel, safe='/')}", {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}, "", dst))

                if lfs_items:
                    uniq_objs = list(dict.fromkeys((o, s) for _, o, s, _ in lfs_items))
                    actions = {}
                    lfs_hdrs = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json", "User-Agent": "git-lfs/3.4.0"}
                    if HF_TOKEN:
                        lfs_hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
                    for off in range(0, len(uniq_objs), 100):
                        body = json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": o, "size": s} for o, s in uniq_objs[off:off+100]]}).encode("utf8")
                        with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/datasets/{repo_id}.git/info/lfs/objects/batch", data=body, headers=lfs_hdrs, method="POST"), timeout=30) as r:
                            for obj in json.loads(r.read()).get("objects", []):
                                dl = obj.get("actions", {}).get("download", {})
                                if dl.get("href"):
                                    actions[str(obj["oid"])] = (str(dl["href"]), {str(k): str(v) for k, v in dl.get("header", {}).items()})
                    for rel, oid, _, dst in lfs_items:
                        if oid in actions:
                            direct_items.append((actions[oid][0], actions[oid][1], oid, dst))
                        else:
                            direct_items.append((f"https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{urllib.parse.quote(rel, safe='/')}", {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}, "", dst))

                def _run_aria2_batch(items, j: int, x: int, wait_s: int, tries: int) -> None:
                    mf_lines = []
                    for url, hmap, oid, dst in items:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        mf_lines.extend([url, f"  dir={dst.parent}", f"  out={dst.name}"])
                        for hk, hv in hmap.items():
                            mf_lines.append(f"  header={hk}: {hv}")
                        if oid:
                            mf_lines.append(f"  checksum=sha-256={oid}")
                    mf = STAGE_ROOT / f"aria2_{repo_id.replace('/', '_')}.txt"
                    mf.write_text("\\n".join(mf_lines) + "\\n", encoding="utf8")
                    subprocess.run(["aria2c", f"--input-file={mf}", f"-j{j}", f"-x{x}", f"-s{x}", "-k1M", "--file-allocation=none", "--check-integrity=true", f"--retry-wait={wait_s}", f"--max-tries={tries}", "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn", "--summary-interval=10"], check=False)
                    mf.unlink(missing_ok=True)

                if direct_items:
                    _run_aria2_batch(direct_items, j=32, x=8, wait_s=3, tries=4)
                    rem_t1 = [it for it in direct_items if not it[3].is_file()]
                    if rem_t1:
                        print(f"  [BACKOFF] Retrying {len(rem_t1)} throttled HF files with -j6 -x2", flush=True)
                        _run_aria2_batch(rem_t1, j=6, x=2, wait_s=5, tries=5)

                missing = [it for it in direct_items if not it[3].is_file()]
                if missing:
                    def _dl_one(it):
                        url, hmap, _, dst = it
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **hmap}), timeout=60) as r:
                            dst.write_bytes(r.read())
                    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                        for f in concurrent.futures.as_completed([pool.submit(_dl_one, m) for m in missing]):
                            f.result()
                tot_mb = sum(dst.stat().st_size for _, _, _, dst in direct_items if dst.is_file()) / 1e6
                dt = max(time.perf_counter() - t0, 0.01)
                print(f"  [CDN] {repo_id}: {len(direct_items)} files ({tot_mb:.1f} MB) in {dt:.1f}s ({tot_mb/dt:.1f} MB/s)", flush=True)

            def _pack_and_upload_drive_api(src_dir: Path, tar_name: str) -> dict:
                local_tar = STAGE_ROOT / tar_name
                local_tar.unlink(missing_ok=True)
                t0 = time.perf_counter()
                subprocess.run(["tar", "-cf", str(local_tar), "-C", str(src_dir), "."], check=True)
                sz = local_tar.stat().st_size
                dt_pack = max(time.perf_counter() - t0, 0.01)
                digest = _digest_file(local_tar, "sha256")
                shutil.rmtree(src_dir, ignore_errors=True)

                existing = _find_drive_file(tar_name)
                if existing:
                    DRIVE_SVC.files().delete(fileId=existing["id"]).execute()

                t1 = time.perf_counter()
                media = MediaFileUpload(str(local_tar), mimetype="application/x-tar", chunksize=64 * 1024 * 1024, resumable=True)
                req = DRIVE_SVC.files().create(body={"name": tar_name, "parents": [DRIVE_FOLDER_ID]}, media_body=media, fields="id, size, webViewLink")
                resp = None
                while resp is None:
                    status, resp = req.next_chunk()
                    if status:
                        print(f"  [DRIVE API] {tar_name}: {status.progress()*100:.0f}%", flush=True)
                dt_up = max(time.perf_counter() - t1, 0.01)
                fid = resp["id"]
                _share_if_requested(fid)
                range_ok = _verify_drive_aria2_range(fid) if MAKE_SHAREABLE_LINKS else False
                local_tar.unlink(missing_ok=True)
                uc_url = f"https://drive.google.com/uc?id={fid}&export=download"
                view_url = resp.get("webViewLink", f"https://drive.google.com/file/d/{fid}/view?usp=sharing")
                print(f"[DONE] {tar_name}: {sz/1e6:.1f} MB | tar={(sz/1e6)/dt_pack:.0f} MB/s | api_up={(sz/1e6)/dt_up:.1f} MB/s | aria2_206={range_ok} | id={fid}", flush=True)
                return {"tar": tar_name, "file_id": fid, "bytes": sz, "sha256": digest, "aria2_range_206": range_ok, "direct_url": uc_url, "view_url": view_url}

            RESULTS = {}

            def _maybe_skip(ds_key: str, tar_name: str) -> bool:
                ex = _find_drive_file(tar_name)
                if ex and not OVERWRITE_EXISTING and int(ex.get("size", 0)) > 1_000_000:
                    fid = ex["id"]
                    _share_if_requested(fid)
                    range_ok = _verify_drive_aria2_range(fid) if MAKE_SHAREABLE_LINKS else False
                    RESULTS[ds_key] = {
                        "tar": tar_name, "file_id": fid, "bytes": int(ex["size"]), "sha256": "existing",
                        "aria2_range_206": range_ok,
                        "direct_url": f"https://drive.google.com/uc?id={fid}&export=download",
                        "view_url": f"https://drive.google.com/file/d/{fid}/view?usp=sharing",
                    }
                    print(f"[SKIP EXISTING] {tar_name} ({int(ex['size'])/1e6:.1f} MB) aria2_206={range_ok} id={fid}", flush=True)
                    return True
                return False

            if STAGE_DMD and not _maybe_skip("dmd", "dmd.tar"):
                print("\\n=== [1] STAGING DMD (UCI #456 via aria2c) ===", flush=True)
                dmd_work = STAGE_ROOT / "dmd_work"
                shutil.rmtree(dmd_work, ignore_errors=True); dmd_work.mkdir(parents=True, exist_ok=True)
                dmd_zip = _aria2_fetch_with_backoff(str(CUSTOM_DMD_URL).strip() or DMD_UCI_URL, STAGE_ROOT / "dmd_raw.zip", min_bytes=500_000)
                _unpack_archive(dmd_zip, dmd_work); dmd_zip.unlink(missing_ok=True)
                for sub_zip in list(dmd_work.rglob("*.zip")):
                    _unpack_archive(sub_zip, dmd_work); sub_zip.unlink(missing_ok=True)
                RESULTS["dmd"] = _pack_and_upload_drive_api(dmd_work, "dmd.tar")

            if STAGE_INCIDENTS_VQA and not _maybe_skip("incidents_vqa", "incidents_vqa.tar"):
                print("\\n=== [2] STAGING DisasterVQA-Incidents (Incidents1M-only, 0% CrisisMMD) ===", flush=True)
                vq_work = STAGE_ROOT / "incidents_vqa_work"
                shutil.rmtree(vq_work, ignore_errors=True); vq_work.mkdir(parents=True, exist_ok=True)
                if str(CUSTOM_INCIDENTS_ZIP_URL).strip():
                    vzip = _aria2_fetch_with_backoff(str(CUSTOM_INCIDENTS_ZIP_URL).strip(), STAGE_ROOT / "vqa_custom.zip", min_bytes=1_000_000)
                    _unpack_archive(vzip, vq_work); vzip.unlink(missing_ok=True)
                    for drop_n in ("CrisisMMD", "MEDIC"):
                        for d in vq_work.rglob(drop_n):
                            if d.is_dir():
                                shutil.rmtree(d, ignore_errors=True)
                else:
                    _lfs_batch_aria2("QCRI/DisasterVQA", INCIDENTS_REV, ["disastervqa_annotations.jsonl", "taxonomy.json"], vq_work)
                    inc_rels = sorted({
                        str(json.loads(ln)["image_path"]).replace("\\\\", "/").lstrip("/")
                        for ln in (vq_work / "disastervqa_annotations.jsonl").read_text(encoding="utf8").splitlines()
                        if ln.strip() and json.loads(ln).get("dataset_source") == "incidents1m"
                    })
                    _lfs_batch_aria2("QCRI/DisasterVQA", INCIDENTS_REV, inc_rels, vq_work)
                RESULTS["incidents_vqa"] = _pack_and_upload_drive_api(vq_work, "incidents_vqa.tar")

            if STAGE_DISASTER_M3 and not _maybe_skip("disaster_m3", "disaster_m3.tar"):
                print("\\n=== [3] STAGING DisasterM3 (NeurIPS'25 Optical + SAR, 2363 tiles via LFS Batch CDN) ===", flush=True)
                m3_work = STAGE_ROOT / "disaster_m3_work"
                shutil.rmtree(m3_work, ignore_errors=True); m3_work.mkdir(parents=True, exist_ok=True)
                _lfs_batch_aria2("Kingdrone-Junjue/DisasterM3", M3_REV, ["DisasterM3_Bench/benchmark_release.json"], m3_work)
                rows_m3 = json.loads((m3_work / "DisasterM3_Bench/benchmark_release.json").read_text(encoding="utf8"))
                m3_rels = sorted({
                    "DisasterM3_Bench/" + str(r["post_image_path"]).replace("\\\\", "/").lstrip("/")
                    for r in rows_m3 if r.get("task") == "Disaster Report" and r.get("post_image_path")
                })
                _lfs_batch_aria2("Kingdrone-Junjue/DisasterM3", M3_REV, m3_rels, m3_work)
                RESULTS["disaster_m3"] = _pack_and_upload_drive_api(m3_work, "disaster_m3.tar")

            if STAGE_CRISISMMD and not _maybe_skip("crisismmd", "crisismmd_v2.tar"):
                print("\\n=== [4] STAGING CrisisMMD_v2 (aria2c -> HF LFS fallback if QCRI CDN unreachable) ===", flush=True)
                cm_work = STAGE_ROOT / "crisismmd_work"
                shutil.rmtree(cm_work, ignore_errors=True); cm_work.mkdir(parents=True, exist_ok=True)
                cm_u = str(CUSTOM_CRISISMMD_TAR_URL).strip()
                pulled_ok = False
                for try_u in ([cm_u] if cm_u else ["https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"]):
                    try:
                        cm_arc = _aria2_fetch_with_backoff(try_u, STAGE_ROOT / "CrisisMMD_v2.0.dl", min_bytes=100_000_000)
                        _unpack_archive(cm_arc, cm_work); cm_arc.unlink(missing_ok=True)
                        pulled_ok = True
                        break
                    except Exception as exc_cm:
                        print(f"  [WARN] QCRI URL unavailable ({exc_cm}); falling back to HuggingFace QCRI/CrisisMMD LFS CDN.", flush=True)
                if not pulled_ok:
                    _lfs_batch_aria2("QCRI/CrisisMMD", CRISIS_REV, None, cm_work / "CrisisMMD_v2.0")
                RESULTS["crisismmd"] = _pack_and_upload_drive_api(cm_work, "crisismmd_v2.tar")

            if STAGE_CIFAR100 and not _maybe_skip("cifar100", "cifar100.tar"):
                print("\\n=== [5] STAGING CIFAR-100 (aria2c + MD5 check) ===", flush=True)
                cf_work = STAGE_ROOT / "cifar100_work"
                shutil.rmtree(cf_work, ignore_errors=True); cf_work.mkdir(parents=True, exist_ok=True)
                cf_u = str(CUSTOM_CIFAR100_URL).strip()
                cf_arc = _aria2_fetch_with_backoff(
                    cf_u or CIFAR100_URL, STAGE_ROOT / "cifar100_dl.tar.gz",
                    checksum="" if cf_u else CIFAR100_MD5, min_bytes=100_000_000,
                )
                _unpack_archive(cf_arc, cf_work); cf_arc.unlink(missing_ok=True)
                RESULTS["cifar100"] = _pack_and_upload_drive_api(cf_work, "cifar100.tar")

            extra_raw = str(EXTRA_DATASET_SPECS_JSON).strip()
            if extra_raw:
                for spec in json.loads(extra_raw):
                    ex_key = str(spec["key"]).strip()
                    ex_tar = str(spec.get("tar") or f"{ex_key}.tar").strip()
                    ex_url = str(spec["url"]).strip()
                    if not _maybe_skip(ex_key, ex_tar):
                        print(f"\\n=== [EXTRA] STAGING {ex_key} ===", flush=True)
                        ex_work = STAGE_ROOT / f"extra_{ex_key}_work"
                        shutil.rmtree(ex_work, ignore_errors=True); ex_work.mkdir(parents=True, exist_ok=True)
                        if ex_url.startswith("hf://"):
                            repo_spec = ex_url[len("hf://"):]
                            repo_id, _, rev = repo_spec.partition("@")
                            _lfs_batch_aria2(repo_id, rev or "main", spec.get("paths"), ex_work)
                        else:
                            ex_arc = _aria2_fetch_with_backoff(ex_url, STAGE_ROOT / f"extra_{ex_key}.dl", min_bytes=int(spec.get("min_bytes", 1024)))
                            _unpack_archive(ex_arc, ex_work); ex_arc.unlink(missing_ok=True)
                        RESULTS[ex_key] = _pack_and_upload_drive_api(ex_work, ex_tar)

            man_local = STAGE_ROOT / "DATASET_ARCHIVES_MANIFEST.json"
            man_local.write_text(json.dumps(RESULTS, indent=2), encoding="utf8")
            old_m = _find_drive_file(man_local.name)
            if old_m:
                DRIVE_SVC.files().delete(fileId=old_m["id"]).execute()
            DRIVE_SVC.files().create(body={"name": man_local.name, "parents": [DRIVE_FOLDER_ID]}, media_body=MediaFileUpload(str(man_local), mimetype="application/json"), fields="id").execute()
            man_local.unlink(missing_ok=True)
            print("[MANIFEST UPLOADED TO DRIVE]", flush=True)
            """,
            cid="stage-cell-3",
        ),
        _cell(
            """
            print("=== COPY-PASTE ARIA2-READY DRIVE LINKS FOR EXPERIMENT NOTEBOOKS ===")
            var_map = {
                "dmd": "DMD_ARCHIVE_URL",
                "incidents_vqa": "INCIDENTS_VQA_ARCHIVE_URL",
                "disaster_m3": "DISASTER_M3_ARCHIVE_URL",
                "crisismmd": "CRISISMMA_TAR_URL",
                "cifar100": "CIFAR100_ARCHIVE_URL",
            }
            for k, info in RESULTS.items():
                var_nm = var_map.get(k, f"{k.upper()}_ARCHIVE_URL")
                rng_tag = "ARIA2 RANGE 206 OK" if info.get("aria2_range_206") else "CHECK SHARE PERMISSION"
                print(f'{var_nm} = "{info["direct_url"]}"  # ({info["bytes"]/1e6:.1f} MB | {rng_tag}) view: {info["view_url"]}')
            """,
            cid="stage-cell-4",
        ),
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "cells": cells,
    }
    nb_json = json.dumps(nb, indent=2, ensure_ascii=False)
    if "$" in nb_json:
        raise AssertionError("Forbidden dollar character in Stage_Disaster_Datasets_To_Drive notebook")
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
