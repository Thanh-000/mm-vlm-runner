"""Build standalone Colab verification notebook for author baseline parity and confound audit."""

from __future__ import annotations

import ast
import json
import textwrap
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "colab" / "ICCV_Author_Baseline_Parity_Verification.ipynb"
RUNNER_REPO_URL = "https://github.com/Thanh-000/mm-vlm-runner.git"


def _cell(src: str, kind: str = "code", cid: str = "") -> dict:
    clean = textwrap.dedent(src).lstrip("\n")
    if kind == "markdown":
        return {"id": cid, "cell_type": "markdown", "metadata": {}, "source": clean.splitlines(keepends=True)}
    return {
        "id": cid,
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": clean.splitlines(keepends=True),
    }


def build() -> None:
    cells = [
        _cell(
            """
            # ICCV Baseline Parity & Confound Audit (Standalone Native-Paper Verification)

            This standalone Colab notebook reproduces **published author baselines on their own native benchmarks (`CIFAR-100`, 10 tasks x 10 classes, OpenAI CLIP `ViT-B/16`)** alongside our `mnarm` operators, isolating both:
            1. **Direct Tensor Parity (Exp 1):** Elementwise equivalence of `mnarm.methods.merging` (`mean`, `ta_sum`, `magmax`, `magmax_s1`, `ties20`, `iso_c`, `iso_cts`, `tsvm`) against cloned `danielm1405/magmax` and `danielm1405/iso-merging`.
            2. **MG-CLIP (ICCV 2025 Highlight, `arXiv:2507.09118`, `linlany/MindtheGap`) (Exp 2):** Verbatim author execution (`order0`, `lr=1e-3`, dynamic text LoRA, MGC `3` epochs) vs. `mnarm` current YAML config (`lr=3e-5`, frozen `theta_0` prompts) vs. author-matched `mnarm` config (`lr=1e-3`, dynamic prompts).
            3. **MagMax (ECCV 2024, `arXiv:2407.06322`, `danielm1405/magmax`) + TSV-M + Iso-C (Exp 3):** Verbatim `default_class_order`, `18`-template head, `lr=1e-5`, `epochs=10`, `Ind-FT (theta_0)` vs `Seq-FT (theta_{t-1})`, verifying published Table 1 & Table 3 (`Zero-Shot = 66.91%`, `Seq-FT MagMax = 80.41%`).
            4. **LoRA Learning Rate & Readout Head Confound Matrix (Exp 4):** Quantifies how `Readout Head` (`Prompt Raw` vs `NCM none_vmf` vs `CSS-CT`) and `LoRA lr` (`3e-5` vs `3e-4`) behave on standard `P1`-like streams without MNAR missingness.
            - **Zero Manual Upload / Max-Speed `aria2c` Fetch:** Dataset (`CIFAR-100`, ~169 MB) and OpenAI `ViT-B-16.pt` (~351 MB) / HF `CLIP ViT-B/16` weights are downloaded directly into `/dev/shm` via **16-connection `aria2c -x16 -s16` + `pigz`** (with automatic web-overload backoff `-x4 -s4` and optional Google Drive `.tar` support via `CIFAR100_ARCHIVE_URL`).
            """,
            kind="markdown",
            cid="parity-cell-0",
        ),
        _cell(
            """
            #@title Audit Switches { display-mode: "form" }
            RUN_TENSOR_PARITY = True  # @param {type:"boolean"}
            RUN_MG_CLIP_CIFAR100 = True  # @param {type:"boolean"}
            RUN_MAGMAX_CIFAR100 = True  # @param {type:"boolean"}
            RUN_LORA_CONFOUND_MATRIX = True  # @param {type:"boolean"}
            CIFAR100_ARCHIVE_URL = ""  # @param {type:"string"}
            AUTO_DOWNLOAD = True  # @param {type:"boolean"}
            """,
            cid="parity-cell-1",
        ),
        _cell(
            f"""
            import concurrent.futures, html, importlib, os, re, shutil, site, subprocess, sys, urllib.parse
            from pathlib import Path
            import requests
            import torch
            import torchvision

            if not torch.cuda.is_available():
                raise RuntimeError("Enable a GPU runtime in Colab before running this verification notebook.")

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            WS = Path("/content/parity_ws")
            WS.mkdir(parents=True, exist_ok=True)
            shm_free = (shutil.disk_usage("/dev/shm").free / 1e9) if Path("/dev/shm").exists() else 0.0
            DATA_DIR = Path("/dev/shm/cifar100_data") if shm_free >= 2.0 else (WS / "cifar100_data")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            sys_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9) if hasattr(os, "sysconf") else 16.0
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            MAX_CACHE_IMGS = 60000 if (sys_ram_gb >= 36.0 and vram_gb >= 36.0) else 12000

            print(f"[ENV] GPU={{torch.cuda.get_device_name(0)}} VRAM={{vram_gb:.1f}}GB RAM={{sys_ram_gb:.1f}}GB SHM_FREE={{shm_free:.1f}}GB MAX_CACHE={{MAX_CACHE_IMGS}}", flush=True)

            pkgs = [
                "setuptools<81", "packaging", "open_clip_torch", "ftfy", "regex",
                "transformers==4.57.6", "scipy", "PyYAML", "omegaconf", "hydra-core",
                "continuum", "gdown", "hf_transfer",
            ]
            def _pip_and_repos():
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "git+https://github.com/openai/CLIP.git"], check=True)
                REPOS = {{
                    "mm_vlm_runner": {RUNNER_REPO_URL!r},
                    "MindtheGap": "https://github.com/linlany/MindtheGap.git",
                    "magmax": "https://github.com/danielm1405/magmax.git",
                    "iso_merging": "https://github.com/danielm1405/iso-merging.git",
                }}
                for name, url in REPOS.items():
                    dst = WS / name
                    if (dst / ".git").is_dir():
                        subprocess.run(["git", "-C", str(dst), "fetch", "--depth", "1", "origin"], check=True)
                        subprocess.run(["git", "-C", str(dst), "reset", "--hard", "FETCH_HEAD"], check=True)
                    else:
                        shutil.rmtree(dst, ignore_errors=True)
                        subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(WS / "mm_vlm_runner")], check=True)

            def _apt_aria2():
                if any(shutil.which(b) is None for b in ("aria2c", "pigz", "unzip")) and shutil.which("apt-get"):
                    subprocess.run(["apt-get", "-qq", "update"], check=False)
                    subprocess.run(["apt-get", "-qq", "install", "-y", "aria2", "pigz", "unzip"], check=False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f_pip, f_apt = pool.submit(_pip_and_repos), pool.submit(_apt_aria2)
                f_pip.result(); f_apt.result()

            site.addsitedir(str(WS / "mm_vlm_runner" / "src"))
            importlib.invalidate_caches()
            import mnarm
            from mnarm.data import aria2_download_file, extract_archive_safe, is_valid_binary_archive
            print("[READY] mnarm:", mnarm.__file__, flush=True)

            def _resolve_drive_direct_url(url_in: str) -> tuple[str, dict[str, str]]:
                u = url_in.strip()
                host = urllib.parse.urlparse(u).netloc.lower()
                if "drive" not in host or not host.endswith("google.com"):
                    return u, {{}}
                uc_host = "drive." + "usercontent.google.com"
                if uc_host in host and "confirm=" in u and ("uuid=" in u or "at=" in u):
                    return u, {{}}
                m = re.search(r"/file/d/([A-Za-z0-9_-]+)", u)
                fid = m.group(1) if m else urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("id", [""])[0]
                if not fid:
                    return u, {{}}
                sess = requests.Session()
                sess.headers["User-Agent"] = "Mozilla/5.0"
                probe = f"https://{{uc_host}}/download?id={{fid}}&export=download&confirm=t"
                with sess.get(probe, stream=True, timeout=30, allow_redirects=True) as r:
                    ck = "; ".join(f"{{k}}={{v}}" for k, v in sess.cookies.get_dict().items())
                    hdr = {{"Cookie": ck}} if ck else {{}}
                    if "text/html" not in (r.headers.get("Content-Type") or "").lower():
                        return r.url, hdr
                    body = r.text
                inputs = {{}}
                for tag in re.findall(r"<input[^>]+>", body, flags=re.I):
                    mn = re.search(r'name=["\\\']([^"\\\']+)["\\\']', tag, flags=re.I)
                    mv = re.search(r'value=["\\\']([^"\\\']*)["\\\']', tag, flags=re.I)
                    if mn and mv:
                        inputs[mn.group(1)] = html.unescape(mv.group(1))
                return (f"https://{{uc_host}}/download?" + urllib.parse.urlencode(inputs) if inputs else probe), hdr

            def _ensure_cifar100_aria2(data_dir: Path, custom_url: str) -> None:
                py_dir = data_dir / "cifar-100-python"
                if (py_dir / "train").is_file() and (py_dir / "test").is_file():
                    print("[REUSE] CIFAR-100 already cached in", data_dir, flush=True)
                    return
                u_custom = custom_url.strip()
                if u_custom:
                    direct_u, hdrs = _resolve_drive_direct_url(u_custom)
                    dl_p = data_dir / "cifar100_stage.dl"
                    try:
                        aria2_download_file(direct_u, dl_p, headers=hdrs, min_bytes=100_000_000, max_conn=16)
                    except Exception:
                        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", u_custom, "-O", str(dl_p)], check=True)
                    extract_archive_safe(dl_p, data_dir)
                    dl_p.unlink(missing_ok=True)
                else:
                    tgz_mirrors = (
                        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz",
                        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                    )
                    tgz_p = data_dir / "cifar-100-python.tar.gz"
                    fetched_ok, exc_web = False, None
                    for tgz_url in tgz_mirrors:
                        try:
                            aria2_download_file(
                                tgz_url, tgz_p, checksum="md5=eb9058c3a382ffc7106e4002c42a8d85",
                                min_bytes=160_000_000, max_conn=16,
                            )
                            extract_archive_safe(tgz_p, data_dir)
                            tgz_p.unlink(missing_ok=True)
                            fetched_ok = True
                            break
                        except Exception as exc_try:
                            exc_web = exc_try
                            tgz_p.unlink(missing_ok=True)
                    if not fetched_ok:
                        print(f"[WARN] Direct aria2c web pull throttled/failed ({{exc_web}})", flush=True)
                        fb_url = input("Web source overloaded; paste Google Drive CIFAR100_ARCHIVE_URL from Stage_Disaster_Datasets_To_Drive.ipynb (or Enter to retry torchvision): ").strip()
                        if fb_url:
                            _ensure_cifar100_aria2(data_dir, fb_url)
                            return
                        torchvision.datasets.CIFAR100(str(data_dir), train=True, download=True)
                        torchvision.datasets.CIFAR100(str(data_dir), train=False, download=True)
                torchvision.datasets.CIFAR100(str(data_dir), train=True, download=False)
                torchvision.datasets.CIFAR100(str(data_dir), train=False, download=False)

            _ensure_cifar100_aria2(DATA_DIR, str(CIFAR100_ARCHIVE_URL))
            clip_pt = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
            aria2_download_file(
                "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
                clip_pt,
                checksum="sha-256=5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
                min_bytes=340_000_000, max_conn=16,
            )
            try:
                from huggingface_hub import snapshot_download
                snapshot_download("openai/clip-vit-base-patch16", revision="57c216476eefef5ab752ec549e440a49ae4ae5f3", ignore_patterns=["*.msgpack", "*.h5", "*.ot", "tf_model*"])
            except Exception as exc_hf:
                print(f"[WARN] HF CLIP snapshot pre-warm skipped ({{exc_hf}})", flush=True)
            print("[READY] CIFAR-100 & CLIP ViT-B-16.pt cached via aria2c at", DATA_DIR, flush=True)

            AUDIT_RESULTS = {{}}
            """,
            cid="parity-cell-2",
        ),
        _cell(
            """
            # Experiment 1: Direct Operator Tensor Parity vs Official Author Code (magmax & iso-merging)
            import importlib.util, sys
            from types import SimpleNamespace
            import torch
            import torch.nn.functional as F
            from mnarm.methods.merging import merge_task_deltas

            def _load_isolated_module(mod_name: str, file_path: Path, repo_root: Path):
                saved_path = list(sys.path)
                for k in list(sys.modules):
                    if k == "src" or k.startswith("src."):
                        del sys.modules[k]
                sys.path.insert(0, str(repo_root))
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
                    mod = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(mod)
                    return mod
                finally:
                    for k in list(sys.modules):
                        if k == "src" or k.startswith("src."):
                            del sys.modules[k]
                    sys.path[:] = saved_path

            if RUN_TENSOR_PARITY:
                mm_tv = _load_isolated_module("mm_tv", WS / "magmax" / "src" / "merging" / "task_vectors.py", WS / "magmax")
                mm_ties = _load_isolated_module("mm_ties", WS / "magmax" / "src" / "merging" / "ties.py", WS / "magmax")
                iso_mod = _load_isolated_module("iso_utils", WS / "iso_merging" / "src" / "utils" / "iso.py", WS / "iso_merging")

                def _compare_pair(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
                    af, bf = a.detach().float().cpu().reshape(-1), b.detach().float().cpu().reshape(-1)
                    mad = float((af - bf).abs().max().item())
                    cos = float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())
                    return {"operator": name, "max_abs_diff": mad, "cosine": cos}

                torch.manual_seed(17)
                # Case A: 4 full-rank (64 x 64) task matrices
                full_mats = [torch.randn(64, 64) * 0.05 for _ in range(4)]
                full_deltas = [{"k": m} for m in full_mats]
                mm_vecs = [mm_tv.TaskVector(vector={"k": m.clone()}) for m in full_mats]
                iso_cfg = SimpleNamespace(device="cpu", DATASETS=[f"T{i}" for i in range(4)], method=SimpleNamespace(common_space_fraction=0.8))

                rows_full = []
                # Mean & TA-Sum
                rows_full.append(_compare_pair("mean (full-rank)", merge_task_deltas(full_deltas, "mean")["k"], torch.stack(full_mats).mean(0)))
                rows_full.append(_compare_pair("ta_sum (full-rank)", merge_task_deltas(full_deltas, "ta_sum")["k"], torch.stack(full_mats).sum(0)))
                # MagMax (s=0.5 & s=1.0)
                auth_maxabs = mm_tv.merge_max_abs(mm_vecs).vector["k"]
                rows_full.append(_compare_pair("magmax s=0.5 (full-rank)", merge_task_deltas(full_deltas, "magmax")["k"], 0.5 * auth_maxabs))
                rows_full.append(_compare_pair("magmax_s1 s=1.0 (full-rank)", merge_task_deltas(full_deltas, "magmax_s1")["k"], auth_maxabs))
                # TIES-20
                auth_ties = mm_ties.merge_methods("topk", torch.vstack([m.reshape(-1) for m in full_mats]), reset_thresh=20, resolve_method="mass", merge_func="dis-mean").reshape(64, 64)
                rows_full.append(_compare_pair("ties20 (full-rank)", merge_task_deltas(full_deltas, "ties20")["k"], auth_ties))
                # Iso-C & Iso-CTS at rank_per_task=64 (full-rank matrix parity)
                auth_isoc = iso_mod.iso_c(mm_vecs, iso_cfg)["k"]
                auth_isocts = iso_mod.iso_cts(mm_vecs, iso_cfg)["k"]
                rows_full.append(_compare_pair("iso_c (rank_cap=64)", merge_task_deltas(full_deltas, "iso_c", rank_per_task=64)["k"], auth_isoc))
                rows_full.append(_compare_pair("iso_cts (rank_cap=64)", merge_task_deltas(full_deltas, "iso_cts", rank_per_task=64)["k"], auth_isocts))

                # TSV-M reference math (Gargiulo et al. CVPR 2025 Eq. 5-7)
                rk_f = 64 // 4
                up, sp, vp = [], [], []
                for m in full_mats:
                    u, s, vh = torch.linalg.svd(m, full_matrices=False)
                    up.append(u[:, :rk_f]); sp.append(s[:rk_f]); vp.append(vh[:rk_f, :].T)
                uu, _, uhu = torch.linalg.svd(torch.cat(up, dim=1), full_matrices=False)
                uv, _, uhv = torch.linalg.svd(torch.cat(vp, dim=1), full_matrices=False)
                ref_tsvm_full = ((uu @ uhu) * torch.cat(sp).unsqueeze(0)) @ (uv @ uhv).T
                rows_full.append(_compare_pair("tsvm (full-rank, rk=16)", merge_task_deltas(full_deltas, "tsvm", rank_per_task=64)["k"], ref_tsvm_full))

                # Case B: LoRA rank-8 delta behavior (same-bank config vs naive full-spectrum iso_c)
                lora_mats = [(torch.randn(64, 8) @ torch.randn(8, 64)) * 0.05 for _ in range(4)]
                lora_deltas = [{"k": m} for m in lora_mats]
                lora_vecs = [mm_tv.TaskVector(vector={"k": m.clone()}) for m in lora_mats]
                auth_isoc_lora_untruncated = iso_mod.iso_c(lora_vecs, iso_cfg)["k"]
                mnarm_isoc_lora = merge_task_deltas(lora_deltas, "iso_c", rank_per_task=8)["k"]
                rows_full.append(_compare_pair("iso_c (LoRA r=8: active-rank vs naive_64)", mnarm_isoc_lora, auth_isoc_lora_untruncated))

                print("=== EXPERIMENT 1: OPERATOR TENSOR PARITY ===", flush=True)
                for row in rows_full:
                    print(f"  {row['operator']:42s} | max_abs_diff={row['max_abs_diff']:.2e} | cos={row['cosine']:.6f}", flush=True)
                AUDIT_RESULTS["tensor_parity"] = rows_full
            """,
            cid="parity-cell-3",
        ),
        _cell(
            """
            # Experiment 2: MG-CLIP (ICCV 2025 Highlight, linlany/MindtheGap) on Native CIFAR-100 10-10
            import math, random, types
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, Subset
            import torchvision, yaml
            from mnarm.data import PreprocessedTensorStore
            from mnarm.methods.vlm_cl_sota import IntraModalCosineClassifier, mg_clip_eval_logits, train_mg_clip_stage
            from mnarm.models import MultimodalVLM, l2_normalize, proto_logits

            def _install_pkg_resources_shim_if_needed():
                if "pkg_resources" in sys.modules:
                    return
                try:
                    import pkg_resources  # noqa: F401
                except Exception:
                    import packaging
                    sys.modules["pkg_resources"] = types.SimpleNamespace(packaging=packaging)

            def run_exp2_mg_clip():
                _install_pkg_resources_shim_if_needed()
                device = torch.device("cuda")
                mt_root = WS / "MindtheGap"
                order = list(yaml.safe_load((mt_root / "class_orders" / "cifar100.yaml").read_text(encoding="utf8"))["class_order"])
                raw_tr = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True)
                class_names = [c.replace("_", " ") for c in raw_tr.classes]
                prompts_ord = [f"a good photo of a {class_names[c]}." for c in order]

                # --- (2A) Verbatim Official linlany/MindtheGap Execution ---
                saved_path = list(sys.path)
                sys.path.insert(0, str(mt_root))
                for k in list(sys.modules):
                    if k == "loraclip" or k.startswith("loraclip.") or k == "continual_clip" or k.startswith("continual_clip."):
                        del sys.modules[k]
                try:
                    from loraclip import lora_clip as auth_lora_clip
                    from continual_clip.models import VisionClassifier as AuthVisionClassifier, forward_clip as auth_forward_clip
                finally:
                    sys.path[:] = saved_path

                random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                m_auth, tf_auth = auth_lora_clip.load("ViT-B/16", device=device, jit=False, r=8, lora_mode="vision+only_kv+text")
                m_auth = m_auth.float()
                m_auth.forward = types.MethodType(auth_forward_clip, m_auth)
                for n_p, p_p in m_auth.named_parameters():
                    if "lora_text_projection" in n_p:
                        p_p.requires_grad_(False)

                tr_ds_a = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True, transform=tf_auth)
                te_ds_a = torchvision.datasets.CIFAR100(str(DATA_DIR), train=False, transform=tf_auth)
                tr_ya, te_ya = np.array(tr_ds_a.targets), np.array(te_ds_a.targets)
                tok_all_a = auth_lora_clip.tokenize(prompts_ord).to(device)

                # Zero-shot check on all 100 classes
                m_auth.eval()
                zs_corr_a = 0
                inv_ord = {c: pos for pos, c in enumerate(order)}
                with torch.no_grad():
                    for imgs, ys in DataLoader(te_ds_a, batch_size=256, shuffle=False, num_workers=2):
                        z_a, _ = m_auth(imgs.to(device), tok_all_a)
                        preds = z_a.argmax(dim=1).cpu().tolist()
                        zs_corr_a += sum(int(pr == inv_ord[int(yv)]) for pr, yv in zip(preds, ys))
                zs_acc_a = 100.0 * zs_corr_a / len(te_ds_a)

                vis_clf = AuthVisionClassifier(512, 10, activation=None).to(device)
                accs_a = []
                for tid in range(10):
                    cur_c = order[tid * 10 : (tid + 1) * 10]
                    seen_c = order[: (tid + 1) * 10]
                    seen_map = {c: i for i, c in enumerate(seen_c)}
                    old_cnt = tid * 10
                    tok_seen = tok_all_a[: len(seen_c)]
                    idx_tr = np.where(np.isin(tr_ya, cur_c))[0]
                    tr_ldr = DataLoader(Subset(tr_ds_a, idx_tr), batch_size=128, shuffle=True, num_workers=2)
                    opt_a = torch.optim.Adam([p for p in m_auth.parameters() if p.requires_grad], lr=1e-3)
                    m_auth.train()
                    for imgs, ys in tr_ldr:
                        yl = torch.tensor([seen_map[int(v)] for v in ys], dtype=torch.long, device=device)
                        out_a, _ = m_auth(imgs.to(device), tok_seen)
                        loss_a = F.cross_entropy(out_a[:, old_cnt:], yl - old_cnt) if tid > 0 else F.cross_entropy(out_a, yl)
                        opt_a.zero_grad(); loss_a.backward(); opt_a.step()

                    m_auth.eval()
                    mgc_ldr = DataLoader(Subset(tr_ds_a, idx_tr), batch_size=64, shuffle=True, num_workers=2)
                    fb = {seen_map[c]: [] for c in cur_c}
                    with torch.no_grad():
                        for imgs, ys in mgc_ldr:
                            _, fi, _ = m_auth(imgs.to(device), tok_seen, return_feature=True)
                            for fv, yv in zip(fi.cpu(), ys):
                                fb[seen_map[int(yv)]].append(fv)
                    means = torch.stack([torch.stack(fb[i]).mean(0) for i in range(old_cnt, old_cnt + 10)]).to(device)
                    if tid > 0:
                        vis_clf.add_weight(means)
                    else:
                        vis_clf.set_weight(means)
                    opt_v = torch.optim.Adam(vis_clf.parameters(), lr=5e-4)
                    sched_v = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=3 * len(mgc_ldr), eta_min=5e-6)
                    for _ in range(3):
                        for imgs, ys in mgc_ldr:
                            yl = torch.tensor([seen_map[int(v)] for v in ys], dtype=torch.long, device=device)
                            with torch.no_grad():
                                _, fi, _ = m_auth(imgs.to(device), tok_seen, return_feature=True)
                            vo = vis_clf(fi)
                            lv = F.cross_entropy(vo[:, old_cnt:], yl - old_cnt) if tid > 0 else F.cross_entropy(vo, yl)
                            opt_v.zero_grad(); lv.backward()
                            if tid > 0 and vis_clf.fc.grad is not None:
                                vis_clf.fc.grad[:old_cnt] = 0.0
                            opt_v.step(); sched_v.step()

                    idx_te = np.where(np.isin(te_ya, seen_c))[0]
                    corr_a = 0
                    with torch.no_grad():
                        for imgs, ys in DataLoader(Subset(te_ds_a, idx_te), batch_size=256, shuffle=False, num_workers=2):
                            z_c, fi, _ = m_auth(imgs.to(device), tok_seen, return_feature=True)
                            z_v = vis_clf(fi)
                            comb = (1.0 * F.softmax(z_c, dim=1) + 4.0 * F.softmax(z_v, dim=1)) / 5.0
                            corr_a += sum(int(pr == seen_map[int(yv)]) for pr, yv in zip(comb.argmax(1).cpu().tolist(), ys))
                    accs_a.append(100.0 * corr_a / len(idx_te))
                print(f"[2A: Official MindtheGap (lr=1e-3)] Zero-Shot={zs_acc_a:.2f}% | Avg={np.mean(accs_a):.2f}% | Last={accs_a[-1]:.2f}%", flush=True)
                del m_auth, vis_clf; torch.cuda.empty_cache()

                # --- Shared PreprocessedTensorStore in mnarm for (2B) and (2C) ---
                tr_pil = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True)
                te_pil = torchvision.datasets.CIFAR100(str(DATA_DIR), train=False)
                row_lookup = {}
                recs_tr_by_c = {c: [] for c in range(100)}
                recs_te_by_c = {c: [] for c in range(100)}
                for idx_r, (im_r, y_r) in enumerate(tr_pil):
                    uid = f"tr_{idx_r}"
                    row_lookup[uid] = {"image": im_r, "text": ""}
                    recs_tr_by_c[int(y_r)].append({"uid": uid, "label": int(y_r), "mask": ("image",)})
                for idx_r, (im_r, y_r) in enumerate(te_pil):
                    uid = f"te_{idx_r}"
                    row_lookup[uid] = {"image": im_r, "text": ""}
                    recs_te_by_c[int(y_r)].append({"uid": uid, "label": int(y_r), "mask": ("image",)})
                prompts_100 = [f"a good photo of a {nm}." for nm in class_names]

                # --- (2B) mnarm Current Config (lr=3e-5, alpha=16, frozen theta_0 prompts) ---
                vlm_b = MultimodalVLM("clip_b16", "openai/clip-vit-base-patch16", device="cuda", rank=8, alpha=16, revision="57c216476eefef5ab752ec549e440a49ae4ae5f3")
                store_b = PreprocessedTensorStore(row_lookup, vlm_b, max_cached_images=MAX_CACHE_IMGS, workers=4, gpu_cache=None)
                p0_b = vlm_b.encode_prompts(prompts_100)
                mg_state_b, mg_heads_b = None, {}
                accs_b = []
                for tid in range(10):
                    old_c = order[: tid * 10]
                    new_c = order[tid * 10 : (tid + 1) * 10]
                    seen_c = old_c + new_c
                    st_recs = [r for c in new_c for r in recs_tr_by_c[c]]
                    mg_state_b, mg_heads_b = train_mg_clip_stage(
                        vlm=vlm_b, prev_state=mg_state_b, mg_heads=mg_heads_b, records=st_recs,
                        protos_full=p0_b, old_classes=old_c, new_classes=new_c, stage_idx=tid + 1,
                        epochs=1, effective_batch_size=128, micro_batch_size=128, lr=3e-5, weight_decay=0.0,
                        seed=42, use_amp=True, tensor_store=store_b, eval_batch_size=256,
                    )
                    te_recs = [r for c in seen_c for r in recs_te_by_c[c]]
                    vlm_b.restore_base_weights(); vlm_b.load_lora_state(mg_state_b); vlm_b.clip.eval()
                    fe_list, ye_list = [], []
                    with torch.inference_mode():
                        for _, batch in store_b.iter_batches(te_recs, batch_size=256, shuffle=False):
                            dev_i = {k: v.to(device) for k, v in batch.items() if k not in ("label", "sample_id")}
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                fe_list.append(vlm_b.encode_image(dev_i).float().cpu())
                            ye_list.append(batch["label"])
                    ui_e, ye = torch.cat(fe_list), torch.cat(ye_list)
                    zi = proto_logits(ui_e, p0_b, seen_c, vlm_b.logit_scale)
                    ev_l = mg_clip_eval_logits(None, ui_e, {"image": zi}, mg_heads_b, seen_c, vlm_b.logit_scale)["image"]
                    pred_g = [seen_c[int(p)] for p in ev_l.argmax(dim=1).tolist()]
                    accs_b.append(100.0 * sum(int(p == int(y)) for p, y in zip(pred_g, ye.tolist())) / len(ye))
                print(f"[2B: mnarm Current Config (lr=3e-5, alpha=16, frozen_t0)] Avg={np.mean(accs_b):.2f}% | Last={accs_b[-1]:.2f}%", flush=True)
                del vlm_b, store_b, mg_heads_b; torch.cuda.empty_cache()

                # --- (2C) mnarm Author-Matched (lr=1e-3, alpha=1, dynamic prompt features) ---
                random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                vlm_c = MultimodalVLM("clip_b16", "openai/clip-vit-base-patch16", device="cuda", rank=8, alpha=1, revision="57c216476eefef5ab752ec549e440a49ae4ae5f3")
                store_c = PreprocessedTensorStore(row_lookup, vlm_c, max_cached_images=MAX_CACHE_IMGS, workers=4, gpu_cache=None)
                tok_100 = vlm_c.tokenize_texts(prompts_100)
                tok_100_dev = {k: v.to(device) for k, v in tok_100.items()}
                head_c = IntraModalCosineClassifier(vlm_c.embed_dim, 100).to(device)
                accs_c = []
                for tid in range(10):
                    old_c = order[: tid * 10]
                    new_c = order[tid * 10 : (tid + 1) * 10]
                    seen_c = old_c + new_c
                    seen_map = {c: i for i, c in enumerate(seen_c)}
                    old_cnt = len(old_c)
                    st_recs = [r for c in new_c for r in recs_tr_by_c[c]]
                    opt_l = torch.optim.Adam(vlm_c.lora_parameters(), lr=1e-3)
                    seen_idx_t = torch.tensor(seen_c, dtype=torch.long, device=device)
                    seen_tok = {k: v.index_select(0, seen_idx_t) for k, v in tok_100_dev.items()}
                    vlm_c.clip.train()
                    for _, batch in store_c.iter_batches(st_recs, batch_size=128, shuffle=True, seed=42 + tid):
                        pv = batch["pixel_values"].to(device)
                        yl = torch.tensor([seen_map[int(v)] for v in batch["label"].tolist()], dtype=torch.long, device=device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            fi = F.normalize(vlm_c.encode_image({"pixel_values": pv}).float(), dim=-1)
                            ft = F.normalize(vlm_c.encode_text(seen_tok).float(), dim=-1)
                            zc = vlm_c.logit_scale * (fi @ ft.T)
                            loss_c = F.cross_entropy(zc[:, old_cnt:], yl - old_cnt) if tid > 0 else F.cross_entropy(zc, yl)
                        opt_l.zero_grad(); loss_c.backward(); opt_l.step()

                    vlm_c.clip.eval()
                    fb_c = {c: [] for c in new_c}
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        for _, batch in store_c.iter_batches(st_recs, batch_size=64, shuffle=False):
                            ui = F.normalize(vlm_c.encode_image({"pixel_values": batch["pixel_values"].to(device)}).float(), dim=-1).cpu()
                            for fv, yv in zip(ui, batch["label"].tolist()):
                                fb_c[int(yv)].append(fv)
                    with torch.no_grad():
                        for c in new_c:
                            head_c.weight.data[c].copy_(torch.stack(fb_c[c]).mean(0).to(device))
                            head_c.initialized[c] = True
                    x_mgc = torch.stack([v for c in new_c for v in fb_c[c]]).to(device)
                    y_mgc = torch.tensor([seen_map[c] for c in new_c for _ in fb_c[c]], dtype=torch.long, device=device)
                    nb = max(1, math.ceil(x_mgc.shape[0] / 64))
                    opt_vc = torch.optim.Adam([head_c.weight], lr=5e-4)
                    sch_vc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vc, T_max=3 * nb, eta_min=5e-6)
                    for _ in range(3):
                        perm = torch.randperm(x_mgc.shape[0], device=device)
                        for off in range(0, x_mgc.shape[0], 64):
                            bi = perm[off : off + 64]
                            vo = head_c(x_mgc[bi], seen_c)
                            lv = F.cross_entropy(vo[:, old_cnt:], y_mgc[bi] - old_cnt) if tid > 0 else F.cross_entropy(vo, y_mgc[bi])
                            opt_vc.zero_grad(); lv.backward()
                            if old_c and head_c.weight.grad is not None:
                                head_c.weight.grad[old_c] = 0.0
                            opt_vc.step(); sch_vc.step()

                    te_recs = [r for c in seen_c for r in recs_te_by_c[c]]
                    with torch.no_grad():
                        ft_cur = F.normalize(vlm_c.encode_text(seen_tok).float(), dim=-1).cpu()
                    fe_list, ye_list = [], []
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        for _, batch in store_c.iter_batches(te_recs, batch_size=256, shuffle=False):
                            fe_list.append(F.normalize(vlm_c.encode_image({"pixel_values": batch["pixel_values"].to(device)}).float(), dim=-1).cpu())
                            ye_list.append(batch["label"])
                    ui_e, ye = torch.cat(fe_list), torch.cat(ye_list)
                    zc_e = vlm_c.logit_scale * (ui_e @ ft_cur.T)
                    ev_l = mg_clip_eval_logits(None, ui_e, {"image": zc_e}, {"image": head_c}, seen_c, vlm_c.logit_scale)["image"]
                    pred_g = [seen_c[int(p)] for p in ev_l.argmax(dim=1).tolist()]
                    accs_c.append(100.0 * sum(int(p == int(y)) for p, y in zip(pred_g, ye.tolist())) / len(ye))
                print(f"[2C: mnarm Author-Matched (lr=1e-3, alpha=1, dyn_prompt)] Avg={np.mean(accs_c):.2f}% | Last={accs_c[-1]:.2f}%", flush=True)
                del vlm_c, store_c, head_c; torch.cuda.empty_cache()

                AUDIT_RESULTS["mg_clip_cifar100"] = {
                    "paper_table1_3order": {"avg": 87.00, "last": 80.57},
                    "paper_tables1_order0": {"avg": 86.79, "last": 80.40, "zeroshot_last": 66.68},
                    "official_mindthegap_exec": {"zeroshot": zs_acc_a, "avg": float(np.mean(accs_a)), "last": float(accs_a[-1]), "per_task": accs_a},
                    "mnarm_current_lr3e5_frozen": {"avg": float(np.mean(accs_b)), "last": float(accs_b[-1]), "per_task": accs_b},
                    "mnarm_matched_lr1e3_dyn": {"avg": float(np.mean(accs_c)), "last": float(accs_c[-1]), "per_task": accs_c},
                }

            if RUN_MG_CLIP_CIFAR100:
                run_exp2_mg_clip()
            """,
            cid="parity-cell-4",
        ),
        _cell(
            """
            # Experiment 3: Official MagMax (ECCV 2024, danielm1405/magmax) + TSV-M + Iso-C on Native CIFAR-100 / 10
            import copy, importlib.util, math, random, sys
            import numpy as np
            import torch
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, Subset
            import torchvision, open_clip
            from mnarm.methods.subspace_merging import _iso_c_matrix, _tsvm_merge_matrix

            def _load_mod_clean(mod_name: str, fp: Path, root: Path):
                sp = list(sys.path)
                for k in list(sys.modules):
                    if k == "src" or k.startswith("src."):
                        del sys.modules[k]
                sys.path.insert(0, str(root))
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, str(fp))
                    mod = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(mod)
                    return mod
                finally:
                    for k in list(sys.modules):
                        if k == "src" or k.startswith("src."):
                            del sys.modules[k]
                    sys.path[:] = sp

            def run_exp3_magmax():
                device = torch.device("cuda")
                mm_root = WS / "magmax"
                mm_cf = _load_mod_clean("mm_cf100", mm_root / "src" / "datasets" / "cifar100.py", mm_root)
                mm_tmpl = _load_mod_clean("mm_tmpl", mm_root / "src" / "datasets" / "templates.py", mm_root)
                mm_tv = _load_mod_clean("mm_tv3", mm_root / "src" / "merging" / "task_vectors.py", mm_root)
                mm_ties = _load_mod_clean("mm_ties3", mm_root / "src" / "merging" / "ties.py", mm_root)

                # Inspect default_class_order directly from cloned magmax repo
                raw_tr = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True)
                ds_inst = mm_cf.CIFAR100(preprocess=None, location=str(DATA_DIR), batch_size=128, num_workers=0)
                class_order = list(ds_inst.default_class_order)
                splits = [sorted(class_order[i * 10 : (i + 1) * 10]) for i in range(10)]
                templates = list(mm_tmpl.cifar100_template)

                base_m, tr_tf, val_tf = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", device=device, cache_dir=str(Path.home() / ".cache" / "clip"))
                base_m = base_m.float().eval()
                with torch.no_grad():
                    zs_w = []
                    for cname in ds_inst.classnames:
                        tok = open_clip.tokenize([t(cname) for t in templates]).to(device)
                        emb = F.normalize(base_m.encode_text(tok).float(), dim=-1).mean(dim=0)
                        zs_w.append(F.normalize(emb, dim=0) * base_m.logit_scale.exp())
                    W_head = torch.stack(zs_w, dim=0)  # [100, 512] normalized*logit_scale

                tr_ds = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True, transform=tr_tf)
                te_ds = torchvision.datasets.CIFAR100(str(DATA_DIR), train=False, transform=val_tf)
                tr_y = np.array(tr_ds.targets)
                te_ldr = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=2)
                ptm_sd = {k: v.detach().cpu().clone() for k, v in base_m.visual.state_dict().items()}

                def _eval_sd(vis_sd: dict[str, torch.Tensor]) -> float:
                    base_m.visual.load_state_dict(vis_sd)
                    base_m.visual.eval()
                    corr = 0
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        for imgs, ys in te_ldr:
                            fi = F.normalize(base_m.encode_image(imgs.to(device)).float(), dim=-1)
                            corr += int(((fi @ W_head.T).argmax(dim=1).cpu() == ys).sum().item())
                    return 100.0 * corr / len(te_ds)

                zs_acc = _eval_sd(ptm_sd)
                print(f"[MagMax Exact Setup] Zero-Shot (18 templates) = {zs_acc:.2f}% (Paper Table 1: 66.91%)", flush=True)

                out_modes = {"zeroshot": zs_acc}
                for seq_mode in (False, True):
                    mode_name = "Seq-FT" if seq_mode else "Ind-FT"
                    random.seed(5); np.random.seed(5); torch.manual_seed(5); torch.cuda.manual_seed_all(5)
                    cur_sd = copy.deepcopy(ptm_sd)
                    tvs = []
                    for sp_i, sp_classes in enumerate(splits):
                        base_m.visual.load_state_dict(cur_sd if seq_mode else ptm_sd)
                        for p in base_m.parameters():
                            p.requires_grad_(False)
                        for p in base_m.visual.parameters():
                            p.requires_grad_(True)
                        sp_map = {c: i for i, c in enumerate(sp_classes)}
                        w_sub = W_head[sp_classes]
                        idx_sp = np.where(np.isin(tr_y, sp_classes))[0]
                        dl = DataLoader(Subset(tr_ds, idx_sp), batch_size=128, shuffle=True, num_workers=2)
                        params = [p for p in base_m.visual.parameters() if p.requires_grad]
                        opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.1)
                        tot_steps = 10 * len(dl)
                        step = 0
                        base_m.visual.train()
                        for _ in range(10):
                            for imgs, ys in dl:
                                lr_now = (1e-5 * ((step + 1) / 500.0)) if step < 500 else (0.5 * 1e-5 * (1.0 + math.cos(math.pi * (step - 500) / max(1, tot_steps - 500))))
                                for pg in opt.param_groups:
                                    pg["lr"] = lr_now
                                step += 1
                                yl = torch.tensor([sp_map[int(v)] for v in ys], dtype=torch.long, device=device)
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                    fi = F.normalize(base_m.encode_image(imgs.to(device)).float(), dim=-1)
                                    loss = F.cross_entropy(fi @ w_sub.T, yl)
                                opt.zero_grad(); loss.backward()
                                torch.nn.utils.clip_grad_norm_(params, 1.0)
                                opt.step()
                        cur_sd = {k: v.detach().cpu().clone() for k, v in base_m.visual.state_dict().items()}
                        tvs.append({k: (cur_sd[k] - ptm_sd[k]).float() for k in ptm_sd})

                    keys = sorted(ptm_sd)
                    # 1. Avg (alpha = 1/10)
                    acc_avg = _eval_sd({k: ptm_sd[k] + torch.stack([t[k] for t in tvs]).mean(0) for k in keys})
                    # 2. TIES (topk=20, scaling_coef=0.55 per magmax/merge_splitted.py line 311)
                    flat_tv = torch.vstack([torch.cat([t[k].reshape(-1) for k in keys]) for t in tvs])
                    ties_flat = mm_ties.merge_methods("topk", flat_tv, reset_thresh=20, resolve_method="mass", merge_func="dis-mean")
                    sizes = [ptm_sd[k].numel() for k in keys]
                    ties_dict = {k: ptm_sd[k] + 0.55 * ch.reshape_as(ptm_sd[k]) for k, ch in zip(keys, ties_flat.split(sizes))}
                    acc_ties = _eval_sd(ties_dict)
                    # 3. Official MaxAbs / MagMax (alpha = 0.5)
                    mm_objs = [mm_tv.TaskVector(vector={k: t[k].clone() for k in keys}) for t in tvs]
                    maxabs_dict = mm_tv.merge_max_abs(mm_objs).vector
                    acc_magmax = _eval_sd({k: ptm_sd[k] + 0.5 * maxabs_dict[k] for k in keys})
                    # 4. mnarm TSV-M & Iso-C on 2D visual weight matrices (rank_per_task=64 full-rank FT)
                    tsvm_d = {k: (_tsvm_merge_matrix([t[k] for t in tvs], rank_per_task=64) if (ptm_sd[k].ndim == 2 and "proj" not in k) else torch.stack([t[k] for t in tvs]).mean(0)) for k in keys}
                    isoc_d = {k: (_iso_c_matrix([t[k] for t in tvs], rank_per_task=768) if (ptm_sd[k].ndim == 2 and "proj" not in k) else torch.stack([t[k] for t in tvs]).mean(0)) for k in keys}
                    acc_tsvm_s1 = _eval_sd({k: ptm_sd[k] + tsvm_d[k] for k in keys})
                    acc_tsvm_s05 = _eval_sd({k: ptm_sd[k] + 0.5 * tsvm_d[k] for k in keys})
                    acc_isoc_s05 = _eval_sd({k: ptm_sd[k] + 0.5 * isoc_d[k] for k in keys})
                    out_modes[mode_name] = {
                        "avg": acc_avg, "ties_0.55": acc_ties, "magmax_0.5": acc_magmax,
                        "tsvm_1.0": acc_tsvm_s1, "tsvm_0.5": acc_tsvm_s05, "iso_c_0.5": acc_isoc_s05,
                    }
                    print(f"[{mode_name}] Avg={acc_avg:.2f}% | TIES={acc_ties:.2f}% | MaxAbs/MagMax={acc_magmax:.2f}% | TSV-M(s=1.0/0.5)={acc_tsvm_s1:.2f}/{acc_tsvm_s05:.2f}% | Iso-C(0.5)={acc_isoc_s05:.2f}%", flush=True)

                AUDIT_RESULTS["magmax_cifar100"] = {
                    "paper_table1_table3": {
                        "zeroshot": 66.91, "joint": 90.94,
                        "Ind-FT": {"avg": 77.04, "ties": 77.23, "maxabs": 76.75},
                        "Seq-FT": {"avg": 79.12, "ties": 80.04, "magmax": 80.41},
                    },
                    "reproduced": out_modes,
                }
                del base_m; torch.cuda.empty_cache()

            if RUN_MAGMAX_CIFAR100:
                run_exp3_magmax()
            """,
            cid="parity-cell-5",
        ),
        _cell(
            r"""
            # Experiment 4: Confound Matrix in zero_mnar (LoRA lr=3e-5 vs 3e-4 & Prompt vs NCM vs CSS-CT Head) + Report
            import copy, json, zipfile
            import numpy as np
            import torch
            import torch.nn.functional as F
            import torchvision
            from IPython.display import Markdown, display
            try:
                from google.colab import files as colab_files
            except ImportError:
                colab_files = None

            from mnarm.data import PreprocessedTensorStore
            from mnarm.methods import SelfNormalizedBayesTransportHead, merge_task_deltas
            from mnarm.models import MultimodalVLM, l2_normalize, proto_logits
            from mnarm.runners.stage_ops import collect_train_features_by_modality, train_lora_stage

            def _eval_acc(logits: torch.Tensor, y_true: torch.Tensor, seen: list[int]) -> float:
                preds = torch.tensor([seen[int(i)] for i in logits.argmax(dim=-1).tolist()], dtype=torch.long)
                return 100.0 * float((preds == y_true.cpu()).float().mean().item())

            def run_exp4_confound_matrix():
                raw_tr = torchvision.datasets.CIFAR100(str(DATA_DIR), train=True)
                raw_te = torchvision.datasets.CIFAR100(str(DATA_DIR), train=False)
                # Identical MagMax CIFAR-100 / 10 splits
                cls_order = [70, 89, 11, 13, 63, 53, 86, 57, 41, 43, 14, 98, 52, 73, 95, 96, 33, 16, 39, 74, 25, 88, 35, 28, 79, 82, 72, 4, 30, 17, 59, 97, 36, 38, 29, 55, 83, 7, 22, 48, 19, 47, 2, 44, 67, 71, 34, 84, 6, 46, 61, 8, 80, 10, 49, 15, 68, 9, 99, 40, 27, 45, 51, 37, 21, 64, 92, 24, 60, 31, 5, 91, 93, 90, 65, 66, 77, 20, 58, 62, 23, 76, 75, 42, 0, 26, 87, 50, 3, 56, 81, 1, 94, 69, 18, 78, 54, 12, 85, 32]
                splits = [sorted(cls_order[i * 10 : (i + 1) * 10]) for i in range(10)]
                row_lookup = {}
                recs_tr, recs_te = {c: [] for c in range(100)}, []
                for idx_r, (im_r, y_r) in enumerate(raw_tr):
                    uid = f"tr_{idx_r}"; row_lookup[uid] = {"image": im_r, "text": ""}
                    recs_tr[int(y_r)].append({"uid": uid, "label": int(y_r), "mask": ("image",)})
                for idx_r, (im_r, y_r) in enumerate(raw_te):
                    uid = f"te_{idx_r}"; row_lookup[uid] = {"image": im_r, "text": ""}
                    recs_te.append({"uid": uid, "label": int(y_r), "mask": ("image",)})

                vlm = MultimodalVLM("clip_b16", "openai/clip-vit-base-patch16", device="cuda", rank=8, alpha=16, revision="57c216476eefef5ab752ec549e440a49ae4ae5f3")
                store = PreprocessedTensorStore(row_lookup, vlm, max_cached_images=MAX_CACHE_IMGS, workers=4, gpu_cache=None)
                prompts_single = [f"a photo of a {c.replace('_', ' ')}." for c in raw_tr.classes]
                p_single = vlm.encode_prompts(prompts_single)

                templates18 = [
                    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
                    "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
                    "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.",
                    "a photo of the {}.", "a blurry photo of the {}.", "a black and white photo of the {}.",
                    "a low contrast photo of the {}.", "a high contrast photo of the {}.", "a bad photo of the {}.",
                    "a good photo of the {}.", "a photo of the small {}.", "a photo of the big {}.",
                ]
                with torch.no_grad():
                    p18_rows = [l2_normalize(vlm.encode_prompts([t.format(c) for t in templates18]).mean(dim=0, keepdim=True)).squeeze(0) for c in raw_tr.classes]
                    p_18 = torch.stack(p18_rows, dim=0)

                all_tr_recs = [r for c in range(100) for r in recs_tr[c]]
                seen_100 = list(range(100))

                def _run_recipe(tag: str, lr_val: float, seq_ft: bool, protos_use: torch.Tensor):
                    deltas = []
                    cur_state = None
                    for sp_i, sp_c in enumerate(splits):
                        sp_recs = [r for c in sp_c for r in recs_tr[c]]
                        st_out, _ = train_lora_stage(
                            vlm=vlm, init_lora=cur_state if seq_ft else None, records=sp_recs,
                            protos_full=protos_use, seen_classes=sp_c, objective="std_ce", stage_idx=sp_i + 1,
                            epochs=5, effective_batch_size=128, micro_batch_size=128, lr=lr_val, weight_decay=1e-4,
                            seed=11, use_amp=True, tensor_store=store,
                        )
                        cur_state = st_out
                        deltas.append(vlm.extract_lora_delta(st_out))

                    recipe_out = {}
                    for m in ("mean", "ties20", "magmax", "magmax_s1", "tsvm", "iso_c", "knots"):
                        dm = merge_task_deltas(deltas, m, rank_per_task=8)
                        tf_tr = collect_train_features_by_modality(vlm=vlm, kind="delta", payload=dm, records=all_tr_recs, tensor_store=store, batch_size=256, use_amp=True)
                        tf_te = collect_train_features_by_modality(vlm=vlm, kind="delta", payload=dm, records=recs_te, tensor_store=store, batch_size=256, use_amp=True)
                        xi_tr, y_tr, xi_te, y_te = tf_tr["image_x"], tf_tr["image_y"], tf_te["image_x"], tf_te["image_y"]

                        acc_raw = _eval_acc(proto_logits(xi_te, protos_use, seen_100, vlm.logit_scale), y_te, seen_100)
                        ncm_h = SelfNormalizedBayesTransportHead(bridge_mode="none_zero_shot", partial_cov_mode="isotropic_vmf")
                        ncm_h.fit(None, None, xi_tr, y_tr, None, None, protos_use, seen_100, fallback_scale=vlm.logit_scale)
                        acc_ncm = _eval_acc(ncm_h.score_tower(xi_te, "image"), y_te, seen_100)

                        css_h = SelfNormalizedBayesTransportHead(bridge_mode="conjugate_bayes", partial_cov_mode="marginal_epistemic")
                        css_h.fit(None, None, xi_tr, y_tr, None, None, protos_use, seen_100, fallback_scale=vlm.logit_scale)
                        acc_css = _eval_acc(css_h.score_tower(xi_te, "image"), y_te, seen_100)
                        recipe_out[m] = {"prompt_raw": acc_raw, "ncm_head": acc_ncm, "css_ct_head": acc_css}
                    print(f"[{tag}] " + " | ".join(f"{k}:{v['prompt_raw']:.1f}/{v['ncm_head']:.1f}/{v['css_ct_head']:.1f}" for k, v in recipe_out.items()), flush=True)
                    return recipe_out

                res_e4 = {
                    "ind_lr3e5_1prompt_current": _run_recipe("Ind-FT lr=3e-5 (zero_mnar current)", 3e-5, False, p_single),
                    "ind_lr3e4_18tmpl_fair": _run_recipe("Ind-FT lr=3e-4 (fair LoRA)", 3e-4, False, p_18),
                    "seq_lr3e4_18tmpl_fair": _run_recipe("Seq-FT lr=3e-4 (MagMax-style LoRA)", 3e-4, True, p_18),
                }
                AUDIT_RESULTS["lora_confound_cifar100"] = res_e4
                del vlm, store; torch.cuda.empty_cache()

            if RUN_LORA_CONFOUND_MATRIX:
                run_exp4_confound_matrix()

            # Build & display Markdown parity audit report (strictly 0 dollar characters)
            lines = [
                "# ICCV Author Baseline Parity & Confound Audit Report (`CIFAR-100` / 10 splits, CLIP `ViT-B/16`)\n",
            ]
            if "tensor_parity" in AUDIT_RESULTS:
                lines.append("## 1. Direct Operator Tensor Parity (`mnarm` vs Official Author Repos)\n")
                lines.append("| Operator | Max Abs Diff | Cosine Similarity |")
                lines.append("|---|---:|---:|")
                for r in AUDIT_RESULTS["tensor_parity"]:
                    lines.append(f"| `{r['operator']}` | `{r['max_abs_diff']:.2e}` | `{r['cosine']:.6f}` |")
                lines.append("")

            if "mg_clip_cifar100" in AUDIT_RESULTS:
                mg = AUDIT_RESULTS["mg_clip_cifar100"]
                lines.append("## 2. `MG-CLIP` (ICCV 2025 Highlight) Native `CIFAR-100 10-10` Verification\n")
                lines.append("| Run | `Avg` (%) | `Last` (%) | Notes |")
                lines.append("|---|---:|---:|---|")
                lines.append(f"| Published Paper (Order 0 / 3-Order Mean) | 86.79 / 87.00 | 80.40 / 80.57 | `arXiv:2507.09118` Table 1 & Table S1 |")
                lines.append(f"| Official `linlany/MindtheGap` (`lr=1e-3`, dyn) | {mg['official_mindthegap_exec']['avg']:.2f} | {mg['official_mindthegap_exec']['last']:.2f} | Zero-Shot = {mg['official_mindthegap_exec']['zeroshot']:.2f}% (Paper 66.68%) |")
                lines.append(f"| `mnarm` Author-Matched (`lr=1e-3`, `alpha=1`, dyn) | {mg['mnarm_matched_lr1e3_dyn']['avg']:.2f} | {mg['mnarm_matched_lr1e3_dyn']['last']:.2f} | Same HF `ViT-B/16` Q/V LoRA |")
                lines.append(f"| `mnarm` Current YAML (`lr=3e-5`, frozen `theta_0`) | {mg['mnarm_current_lr3e5_frozen']['avg']:.2f} | {mg['mnarm_current_lr3e5_frozen']['last']:.2f} | Quantifies `lr=3e-5` LoRA handicap |")
                lines.append("")

            if "magmax_cifar100" in AUDIT_RESULTS:
                mm = AUDIT_RESULTS["magmax_cifar100"]
                p_mm, r_mm = mm["paper_table1_table3"], mm["reproduced"]
                lines.append("## 3. `MagMax` (ECCV 2024) + `TSV-M` + `Iso-C` Native `CIFAR-100 / 10` Verification\n")
                lines.append(f"- Zero-Shot (18 templates): Published = **{p_mm['zeroshot']:.2f}%** | Reproduced = **{r_mm['zeroshot']:.2f}%**\n")
                lines.append("| Training Mode | Method | Published (`arXiv:2407.06322`) | Reproduced Native Author Code |")
                lines.append("|---|---|---:|---:|")
                lines.append(f"| `Ind-FT` (`theta_0`) | `Avg` (`1/10`) | {p_mm['Ind-FT']['avg']:.2f} | {r_mm['Ind-FT']['avg']:.2f} |")
                lines.append(f"| `Ind-FT` (`theta_0`) | `TIES` (`s=0.55`) | {p_mm['Ind-FT']['ties']:.2f} | {r_mm['Ind-FT']['ties_0.55']:.2f} |")
                lines.append(f"| `Ind-FT` (`theta_0`) | `MaxAbs` (`s=0.5`) | {p_mm['Ind-FT']['maxabs']:.2f} | {r_mm['Ind-FT']['magmax_0.5']:.2f} |")
                lines.append(f"| `Seq-FT` (`theta_{{t-1}}`) | `Avg` (`1/10`) | {p_mm['Seq-FT']['avg']:.2f} | {r_mm['Seq-FT']['avg']:.2f} |")
                lines.append(f"| `Seq-FT` (`theta_{{t-1}}`) | `TIES` (`s=0.55`) | {p_mm['Seq-FT']['ties']:.2f} | {r_mm['Seq-FT']['ties_0.55']:.2f} |")
                lines.append(f"| `Seq-FT` (`theta_{{t-1}}`) | **`MagMax`** (`s=0.5`) | **{p_mm['Seq-FT']['magmax']:.2f}** | **{r_mm['Seq-FT']['magmax_0.5']:.2f}** |")
                lines.append(f"| `Seq-FT` (`theta_{{t-1}}`) | `TSV-M` (`s=1.0 / 0.5`) | -- | {r_mm['Seq-FT']['tsvm_1.0']:.2f} / {r_mm['Seq-FT']['tsvm_0.5']:.2f} |")
                lines.append(f"| `Seq-FT` (`theta_{{t-1}}`) | `Iso-C` (`s=0.5`) | -- | {r_mm['Seq-FT']['iso_c_0.5']:.2f} |")
                lines.append("")

            if "lora_confound_cifar100" in AUDIT_RESULTS:
                lines.append("## 4. `zero_mnar` Confound Matrix: `Prompt Head` vs `NCM Head` vs `CSS-CT` & `LoRA lr` (`CIFAR-100 / 10`, Image-Only)\n")
                lines.append("| Recipe | Merger | Prompt Head (`Raw`) | NCM Head (`none_vmf`) | `CSS-CT` Head |")
                lines.append("|---|---|---:|---:|---:|")
                for rec_k, rec_v in AUDIT_RESULTS["lora_confound_cifar100"].items():
                    for m_k, vals in rec_v.items():
                        lines.append(f"| `{rec_k}` | `{m_k}` | {vals['prompt_raw']:.2f} | {vals['ncm_head']:.2f} | {vals['css_ct_head']:.2f} |")
                lines.append("")

            rep_md = "\n".join(lines)
            assert chr(36) not in rep_md, "Forbidden dollar character in parity report!"
            rp_md = Path("/content/ICCV_AUTHOR_PARITY_REPORT.md")
            rp_js = Path("/content/ICCV_AUTHOR_PARITY_RESULTS.json")
            rp_zip = Path("/content/ICCV_AUTHOR_PARITY_VERIFICATION.zip")
            rp_md.write_text(rep_md, encoding="utf8")
            rp_js.write_text(json.dumps(AUDIT_RESULTS, indent=2), encoding="utf8")
            with zipfile.ZipFile(rp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(rp_md, rp_md.name)
                zf.write(rp_js, rp_js.name)
            display(Markdown(rep_md))
            print("[PACKAGED]", rp_zip, rp_zip.stat().st_size, "bytes", flush=True)
            if AUTO_DOWNLOAD and colab_files is not None:
                colab_files.download(str(rp_zip))
            """,
            cid="parity-cell-6",
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
    if chr(36) in nb_json:
        raise AssertionError("Forbidden dollar character detected in parity notebook JSON")
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", SyntaxWarning)
                ast.parse("".join(c["source"]))
                if w:
                    raise SyntaxError(f"SyntaxWarning in cell {idx}: {w[0].message}")

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(nb_json, encoding="utf8")
    print(f"Built {OUT_NB} ({len(nb_json)} B, {len(cells)} cells)")


if __name__ == "__main__":
    build()
