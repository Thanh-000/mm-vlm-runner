# Zero-Support MNAR Multimodal Learning & Cross-Modal Density Transport (`mnarm`)

Official clean-room research repository for testing **Support-Aware MNAR Propensity Risk (`L_MNAR`)** paired with **Self-Normalized Class-Conditional Density & Modality-Gap Transport** under zero-support structural missingness (`P_train(M | y) = 0`) — using **zero calibration labels** from `D_cal`.

## Key Scientific Principles

1. **Co-Observation Graph & Support-Aware Propensity Risk (`L_MNAR`):** Replaces global cross-entropy over unsupported classes (`c ∉ S_M`) with propensity-adjusted support-restricted risk over `S_M`, eliminating the `-∞` collider gradient and Rank-1 Cone Drift during training.
2. **Self-Normalized Shrinkage-LDA Density + Modality-Gap Transport:** Estimates OAS-regularized class-conditional Gaussian/spherical log-densities on observed towers, transports missing `(m, y)` prototypes via closed-form orthogonal Procrustes / prompt bridge across the modality gap, and calibrates scalar nat-scales purely via train-only out-of-fold validation (`0` `D_cal` labels).
3. **Strict `dev`-Only Falsification Hygiene:** `allow_test=false` is enforced across Pilot F1 configs and data loaders until architecture & hyperparameters are frozen.

## Quickstart (Local CPU Verification)

```bash
pip install --no-build-isolation --no-deps -e .
pytest -v
```

## Running Pilot F1 (`dev`-Only, CLIP P4 & SigLIP P4)

```bash
python -m mnarm.runners.pilot_f1 \
  --config configs/pilot_f1_clip_p4.yaml \
  --seq-inputs path/to/crisismmd_seq_inputs \
  --images path/to/CrisisMMD_v2.0 \
  --out outputs/pilot_f1_clip_p4

python -m mnarm.runners.pilot_f1 \
  --config configs/pilot_f1_siglip_p4.yaml \
  --seq-inputs path/to/crisismmd_seq_inputs \
  --images path/to/CrisisMMD_v2.0 \
  --out outputs/pilot_f1_siglip_p4
```

For Google Colab GPU execution, build and open `colab/Pilot_F1_ZeroMNAR_ColabWeb.ipynb` via `make build-colab`.

## Repository Layout

- `src/mnarm/core/` — Determinism, SHA-256 provenance (`RUN_LOCK.json`), atomic I/O.
- `src/mnarm/data/` — Strict split-overlap guards (`0` uid/image/text leak) and CrisisMMD contract loaders.
- `src/mnarm/models/` — LoRA Q/V VLM wrappers (`CLIP` / `SigLIP`) and collision-free `blake2b` tower digests.
- `src/mnarm/methods/` — `L_MNAR`, `SelfNormalizedBayesTransportHead`, and task-vector merging (`TA`, `TIES`, `MagMax`, `TSV-M`).
- `src/mnarm/eval/` — Macro-F1 (`8-cls` & `7-cls`), `ECE`, `NLL`, `Brier`, cone drift `θ_deg`, paired bootstrap CIs, and VRAM/latency profiling.
- `src/mnarm/runners/` — `pilot_f1` runner emitting pre-committed `PASS / PARTIAL / FAIL` falsification verdicts.
- `tests/` — Deterministic CPU test suite verifying mathematical invariants, zero data leakage, and double-blind anonymity.
