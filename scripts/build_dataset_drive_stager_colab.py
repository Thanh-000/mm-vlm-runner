"""Build standalone high-bandwidth Colab-to-Drive .tar stager notebook for all disaster datasets."""

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
            # High-Bandwidth Disaster Dataset-to-Drive `.tar` Stager (Colab Web -> Google Drive)

            Tối đa hoá băng thông Google Cloud của Colab (**chạy được ngay trên CPU Runtime miễn phí, 0 tốn GPU Compute Units**) để kéo song song các bộ dữ liệu thảm hoạ gốc, làm sạch & loại bỏ mọi phần thừa/rò rỉ, rồi đóng gói thành **`.tar` nguyên bản** đẩy thẳng vào Google Drive:

            - **Hugging Face (`DisasterM3`, `DisasterVQA-Incidents`):** Dùng **Git-LFS Batch API (`us.aws.cdn.hf.co`) + `aria2c -j 32` + xác thực `SHA-256` phần cứng**, vượt qua giới hạn tốc độ / `HTTP 429` của `/resolve/`!
            - **`DMD` (`UCI #456`):** Tải trực tiếp file gốc, bung nén cả outer + inner zip, chuẩn hoá thành `dmd.tar`.
            - **`DisasterVQA` (`Incidents1M`-only):** Tải đầy đủ ảnh gốc độ phân giải cao (`420 MB`), xoá sạch `CrisisMMD/` và `MEDIC/` (`0%` trùng lặp), đóng thành `incidents_vqa.tar`.
            - **`DisasterM3` (`NeurIPS'25` Optical + SAR):** Chỉ lấy `benchmark_release.json` + đúng `2,363` ảnh vệ tinh sau thảm hoạ (`~3.58 GB` thay vì `43 GB`), đóng thành `disaster_m3.tar`.
            - **Zero-Overhead Packing:** Đóng `.tar` không nén tại `/dev/shm` hoặc SSD cục bộ rồi stream tuần tự `16 MB` sang Google Drive FUSE + dọn sạch bộ nhớ tạm sau mỗi tập.
            """,
            kind="markdown",
            cid="stage-cell-0",
        ),
        _cell(
            """
            #@title Dataset Staging Configuration { display-mode: "form" }
            DRIVE_DATASET_DIR = "/content/drive/MyDrive/mnarm_iccv_datasets"  # @param {type:"string"}
            STAGE_DMD = True  # @param {type:"boolean"}
            STAGE_INCIDENTS_VQA = True  # @param {type:"boolean"}
            STAGE_DISASTER_M3 = True  # @param {type:"boolean"}
            STAGE_CRISISMMD = False  # @param {type:"boolean"}
            OVERWRITE_EXISTING = False  # @param {type:"boolean"}
            CUSTOM_DMD_URL = ""  # @param {type:"string"}
            CUSTOM_INCIDENTS_ZIP_URL = ""  # @param {type:"string"}
            CUSTOM_CRISISMMD_TAR_URL = ""  # @param {type:"string"}
            """,
            cid="stage-cell-1",
        ),
        _cell(
            """
            import os, shutil, subprocess
            from pathlib import Path
            from google.colab import drive

            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")

            DRIVE_OUT = Path(str(DRIVE_DATASET_DIR).strip())
            DRIVE_OUT.mkdir(parents=True, exist_ok=True)

            if shutil.which("aria2c") is None and shutil.which("apt-get"):
                subprocess.run(["apt-get", "-qq", "update"], check=False)
                subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz", "unzip"], check=False)

            shm_free_gb = shutil.disk_usage("/dev/shm").free / 1e9 if Path("/dev/shm").exists() else 0.0
            root_free_gb = shutil.disk_usage("/content").free / 1e9
            STAGE_ROOT = Path("/dev/shm/mnarm_stage_tmp") if shm_free_gb >= 10.0 else Path("/content/mnarm_stage_tmp")
            STAGE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[READY] DRIVE_OUT={DRIVE_OUT} | STAGE_ROOT={STAGE_ROOT} | /dev/shm free={shm_free_gb:.1f}GB | /content free={root_free_gb:.1f}GB | aria2c={bool(shutil.which('aria2c'))}", flush=True)
            """,
            cid="stage-cell-2",
        ),
        _cell(
            """
            import concurrent.futures, hashlib, json, os, shutil, subprocess, tarfile, time, urllib.parse, urllib.request, zipfile
            from pathlib import Path

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
            CRISISMMA_REV = "10a5626ba112ab9c50369c473a2062bcf205d708"
            DMD_UCI_URL = "https://archive.ics.uci.edu/static/public/456/multimodal+damage+identification+for+humanitarian+computing.zip"

            def _lfs_batch_aria2(repo_id: str, rev: str, rel_paths: list[str], dest_dir: Path) -> None:
                dest_dir.mkdir(parents=True, exist_ok=True)
                req_hdrs = {"User-Agent": "Mozilla/5.0"}
                if HF_TOKEN:
                    req_hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
                t0 = time.perf_counter()
                with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo_id}?blobs=1", headers=req_hdrs), timeout=30) as r:
                    meta_map = {str(s["rfilename"]): s for s in json.loads(r.read()).get("siblings", [])}

                lfs_items, direct_items = [], []
                for rel in sorted(set(r.replace("\\\\", "/").lstrip("/") for r in rel_paths)):
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

                if direct_items:
                    mf_lines = []
                    for url, hmap, oid, dst in direct_items:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        mf_lines.extend([url, f"  dir={dst.parent}", f"  out={dst.name}"])
                        for hk, hv in hmap.items():
                            mf_lines.append(f"  header={hk}: {hv}")
                        if oid:
                            mf_lines.append(f"  checksum=sha-256={oid}")
                    mf = STAGE_ROOT / f"aria2_{repo_id.replace('/', '_')}.txt"
                    mf.write_text("\\n".join(mf_lines) + "\\n", encoding="utf8")
                    subprocess.run(["aria2c", f"--input-file={mf}", "-j", "32", "-x", "4", "-s", "4", "-k", "1M", "--check-integrity=true", "--retry-wait=3", "--max-tries=6", "--auto-file-renaming=false", "--allow-overwrite=true", "--console-log-level=warn", "--summary-interval=10"], check=False)
                    mf.unlink(missing_ok=True)

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

            def _pack_and_push_tar(src_dir: Path, tar_name: str, drive_dir: Path) -> dict:
                local_tar = STAGE_ROOT / tar_name
                local_tar.unlink(missing_ok=True)
                t0 = time.perf_counter()
                subprocess.run(["tar", "-cf", str(local_tar), "-C", str(src_dir), "."], check=True)
                sz = local_tar.stat().st_size
                dt_pack = max(time.perf_counter() - t0, 0.01)

                dst_part = drive_dir / f"{tar_name}.part"
                dst_final = drive_dir / tar_name
                dst_part.unlink(missing_ok=True)
                t1 = time.perf_counter()
                h = hashlib.sha256()
                with local_tar.open("rb") as fin, dst_part.open("wb") as fout:
                    while True:
                        buf = fin.read(16 * 1024 * 1024)
                        if not buf:
                            break
                        h.update(buf)
                        fout.write(buf)
                os.replace(dst_part, dst_final)
                dt_copy = max(time.perf_counter() - t1, 0.01)
                digest = h.hexdigest()
                local_tar.unlink(missing_ok=True)
                shutil.rmtree(src_dir, ignore_errors=True)
                print(f"[DONE] {tar_name}: {sz/1e6:.1f} MB | tar={(sz/1e6)/dt_pack:.0f} MB/s | drive_stream={(sz/1e6)/dt_copy:.1f} MB/s | sha256={digest[:16]}...", flush=True)
                return {"tar": tar_name, "bytes": sz, "sha256": digest}

            results = {}
            manifest_path = DRIVE_OUT / "DATASET_ARCHIVES_MANIFEST.json"
            if manifest_path.is_file():
                try:
                    results = json.loads(manifest_path.read_text(encoding="utf8"))
                except Exception:
                    results = {}

            if STAGE_DMD:
                target_tar = DRIVE_OUT / "dmd.tar"
                if target_tar.is_file() and not OVERWRITE_EXISTING:
                    print(f"[SKIP] {target_tar} already exists ({target_tar.stat().st_size/1e6:.1f} MB).", flush=True)
                else:
                    print("\\n=== [1/3] STAGING DMD (UCI #456) ===", flush=True)
                    dmd_work = STAGE_ROOT / "dmd_work"
                    shutil.rmtree(dmd_work, ignore_errors=True); dmd_work.mkdir(parents=True, exist_ok=True)
                    dmd_zip = STAGE_ROOT / "dmd_raw.zip"
                    src_url = str(CUSTOM_DMD_URL).strip() or DMD_UCI_URL
                    subprocess.run(["curl", "-fL", "--retry", "4", src_url, "-o", str(dmd_zip)], check=True)
                    subprocess.run(["unzip", "-q", "-o", str(dmd_zip), "-d", str(dmd_work)], check=True)
                    dmd_zip.unlink(missing_ok=True)
                    for sub_zip in list(dmd_work.rglob("*.zip")):
                        subprocess.run(["unzip", "-q", "-o", str(sub_zip), "-d", str(dmd_work)], check=True)
                        sub_zip.unlink(missing_ok=True)
                    results["dmd"] = _pack_and_push_tar(dmd_work, "dmd.tar", DRIVE_OUT)

            if STAGE_INCIDENTS_VQA:
                target_tar = DRIVE_OUT / "incidents_vqa.tar"
                if target_tar.is_file() and not OVERWRITE_EXISTING:
                    print(f"[SKIP] {target_tar} already exists ({target_tar.stat().st_size/1e6:.1f} MB).", flush=True)
                else:
                    print("\\n=== [2/3] STAGING DisasterVQA-Incidents (v3 / Incidents1M-only, 0% CrisisMMD) ===", flush=True)
                    vq_work = STAGE_ROOT / "incidents_vqa_work"
                    shutil.rmtree(vq_work, ignore_errors=True); vq_work.mkdir(parents=True, exist_ok=True)
                    if str(CUSTOM_INCIDENTS_ZIP_URL).strip():
                        vzip = STAGE_ROOT / "vqa_custom.zip"
                        subprocess.run(["aria2c", "-x16", "-s16", "-k1M", str(CUSTOM_INCIDENTS_ZIP_URL).strip(), "-d", str(STAGE_ROOT), "-o", vzip.name], check=True)
                        subprocess.run(["unzip", "-q", "-o", str(vzip), "-d", str(vq_work)], check=True); vzip.unlink(missing_ok=True)
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
                    results["incidents_vqa"] = _pack_and_push_tar(vq_work, "incidents_vqa.tar", DRIVE_OUT)

            if STAGE_DISASTER_M3:
                target_tar = DRIVE_OUT / "disaster_m3.tar"
                if target_tar.is_file() and not OVERWRITE_EXISTING:
                    print(f"[SKIP] {target_tar} already exists ({target_tar.stat().st_size/1e6:.1f} MB).", flush=True)
                else:
                    print("\\n=== [3/3] STAGING DisasterM3 (NeurIPS'25 Optical + SAR, 2363 tiles via LFS Batch CDN) ===", flush=True)
                    m3_work = STAGE_ROOT / "disaster_m3_work"
                    shutil.rmtree(m3_work, ignore_errors=True); m3_work.mkdir(parents=True, exist_ok=True)
                    _lfs_batch_aria2("Kingdrone-Junjue/DisasterM3", M3_REV, ["DisasterM3_Bench/benchmark_release.json"], m3_work)
                    rows_m3 = json.loads((m3_work / "DisasterM3_Bench/benchmark_release.json").read_text(encoding="utf8"))
                    m3_rels = sorted({
                        "DisasterM3_Bench/" + str(r["post_image_path"]).replace("\\\\", "/").lstrip("/")
                        for r in rows_m3 if r.get("task") == "Disaster Report" and r.get("post_image_path")
                    })
                    _lfs_batch_aria2("Kingdrone-Junjue/DisasterM3", M3_REV, m3_rels, m3_work)
                    results["disaster_m3"] = _pack_and_push_tar(m3_work, "disaster_m3.tar", DRIVE_OUT)

            if STAGE_CRISISMMD:
                target_tar = DRIVE_OUT / "crisismmd_v2.tar"
                if target_tar.is_file() and not OVERWRITE_EXISTING:
                    print(f"[SKIP] {target_tar} already exists ({target_tar.stat().st_size/1e6:.1f} MB).", flush=True)
                else:
                    print("\\n=== [OPT] STAGING CrisisMMD_v2 ===", flush=True)
                    cm_work = STAGE_ROOT / "crisismmd_work"
                    shutil.rmtree(cm_work, ignore_errors=True); cm_work.mkdir(parents=True, exist_ok=True)
                    cm_tgz = STAGE_ROOT / "CrisisMMD_v2.0.tar.gz"
                    src_cm = str(CUSTOM_CRISISMMD_TAR_URL).strip() or "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"
                    subprocess.run(["aria2c", "-x16", "-s16", "-k1M", src_cm, "-d", str(STAGE_ROOT), "-o", cm_tgz.name], check=True)
                    subprocess.run(["tar", "-I", "pigz", "-xf", str(cm_tgz), "-C", str(cm_work)], check=True); cm_tgz.unlink(missing_ok=True)
                    results["crisismmd"] = _pack_and_push_tar(cm_work, "crisismmd_v2.tar", DRIVE_OUT)

            manifest_path.write_text(json.dumps(results, indent=2), encoding="utf8")
            print("\\n[ALL DONE] Saved manifest ->", manifest_path, flush=True)
            """,
            cid="stage-cell-3",
        ),
        _cell(
            """
            import json, subprocess
            from pathlib import Path

            DRIVE_OUT = Path(str(DRIVE_DATASET_DIR).strip())
            print("=== DRIVE DATASET ARCHIVES SUMMARY ===", DRIVE_OUT)
            for p in sorted(DRIVE_OUT.glob("*.tar")):
                head = subprocess.check_output(["bash", "-c", f'tar -tf "{p}" | head -n 5'], text=True).strip().splitlines()
                print(f"  - {p.name:20s} | {p.stat().st_size/1e9:.2f} GB ({p.stat().st_size/1e6:.1f} MB) | sample={head[:3]}")
            mp = DRIVE_OUT / "DATASET_ARCHIVES_MANIFEST.json"
            if mp.is_file():
                print("\\nManifest:\\n", mp.read_text(encoding="utf8"))
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
