# Canonical Spherical Sufficiency & Conjugate Transport (`CSS-CT` / `mnarm`)

Official clean-room research repository for **`CSS-CT` (Canonical Spherical Sufficiency & Conjugate Transport)** — a zero-exemplar, zero-calibration-label plug-and-play framework solving **Zero-Support Missing-Not-At-Random (`P_train(M | y) = 0`)** continual multimodal VLM merging and adaptation.

For full mathematical contracts, see [docs/FRAMEWORK_SPECIFICATION.md](docs/FRAMEWORK_SPECIFICATION.md), [docs/METHOD_FREEZE_ICCV.md](docs/METHOD_FREEZE_ICCV.md), and [docs/AUTHOR_CODE_PARITY_ICCV.md](docs/AUTHOR_CODE_PARITY_ICCV.md).

## Core Architecture (`CSS-CT` v3)

1. **Canonical `O(|Y|d + d^2)` Sufficient Bank (`Pillar 1`):** Aligns each stage adapter isometrically (`O(d)`) onto the frozen pretrained reference frame `theta_0` on `S^{d-1}`, maintaining exact Chan-Golub-LeVeque class centroids, pooled scatter, and `O(d^2)` paired cross-moments `(N_TI, s_T, s_I, C_TI)` with **`0` stored exemplars**.
2. **Single Isometric `O(d)` Frame Re-Anchoring (`Pillar 2`):** Solves one uncentered orthogonal Procrustes map `Q*_{m->0} in O(d)` from unlabeled anchors (`P U U_cal`, `0` labels) to map adapted/merged towers back to `theta_0` with isometric norm preservation (avoiding hubness collapse), zero extra inference weights (`weight folding`), and partial cross-modal `I->T R@1` recovery without retraining.
3. **Conjugate Empirical-Bayes Transport & All-`|Y|` Stat-OOF `tau_m` (`Pillar 3`):** Synthesizes missing `(m, y)` centroids from unbiased cone-adjusted prompt and cross-modal moment bridges, applies trace-preserving epistemic covariance shrinkage `alpha_m` on partial towers plus unconditional spherical Mahalanobis norm equalization, and calibrates `tau_m` against all `|Y|` centroids purely from canonical bank sufficient statistics.
4. **Strict `dev`-Only Falsification Hygiene:** `allow_test=false` is enforced across all configs, runners, and Colab bundles prior to final pre-registered confirmatory evaluation.

## Quickstart (Local Verification)

```bash
pip install --no-build-isolation --no-deps -e .
pytest -v
```

## Running Unified Same-Bank ICCV Evaluation (`dev`-Only)

```bash
python -m mnarm.runners.iccv_same_bank \
  --config configs/iccv_crisismmd_clip.yaml \
  --seq-inputs path/to/crisismmd_seq_inputs \
  --images path/to/CrisisMMD_v2.0 \
  --out outputs/iccv_same_bank_clip \
  --protocols P4 P3 P2 P1

python -m mnarm.runners.iccv_same_bank \
  --config configs/iccv_crisismmd_siglip.yaml \
  --seq-inputs path/to/crisismmd_seq_inputs \
  --images path/to/CrisisMMD_v2.0 \
  --out outputs/iccv_same_bank_siglip \
  --protocols P4 P3 P2 P1
```

For Colab Web (A100 / G4 / L4), rebuild and open [colab/ICCV_SameBank_ZeroMNAR_ColabWeb.ipynb](colab/ICCV_SameBank_ZeroMNAR_ColabWeb.ipynb) via `python scripts/build_iccv_same_bank_colab.py`.

## Repository Layout

- `src/mnarm/core/` — Determinism, SHA-256 provenance (`RUN_LOCK.json`), atomic I/O.
- `src/mnarm/data/` — Strict split-overlap guards (`0` uid/image/text leak), high-VRAM (`>= 36 GB`) GPU tensor cache, and 4 zero-overlap Multi-Altitude Disaster Benchmarks (`crisismmd`, `dmd`, `incidents_vqa`, `disaster_m3`).
- `src/mnarm/models/` — LoRA Q/V VLM wrappers (`CLIP` / `SigLIP`, fast processors) and collision-free `blake2b` tower digests.
- `src/mnarm/methods/` — Canonical `CSS-CT` (`StreamingClassBank`, `reanchor`, `spherical_stats`, `bayes_transport`), batched GPU/CPU `14` verbatim 2023–2026 mergers (`subspace_merging`, `iterative_merging`), `GR-CLIP`, `MG-CLIP`, `PAL-RLS`, and counterfactual `L_MNAR`.
- `src/mnarm/eval/` — Macro-F1, `ECE/NLL/Brier`, Sato `Excess Gini`, cone drift `theta_deg`, paired bootstrap CIs, profiler, and 4 publication diagnostic figures (`PNG` + `PDF`).
- `src/mnarm/runners/` — Unified `iccv_same_bank`, `iccv_eval`, `iccv_report`, `crisismmd_p1_p4`, and `pilot_f1` runners.
- `tests/` — Deterministic CPU test suite verifying mathematical invariants, `0` data leakage, verbatim author parity, and double-blind anonymity.
