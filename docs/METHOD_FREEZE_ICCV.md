# Method Freeze & Unified Same-Bank ICCV Evaluation Protocol (`Zero-MNAR`)

> **Status:** FROZEN at commit `efefb03` (`METHOD_FREEZE`). No hyperparameter sweeps, post-hoc dataset patches, or `test` split reads are permitted prior to final confirmatory reporting.

## 1. Four-Layer Scientific Separation & Information Regimes

Every comparison in this benchmark isolates **exactly one layer** at a time while holding the remaining three layers strictly identical:

1. **Contract Layer:** Fixed split manifests, zero UID/image/text overlap across `train` / `cal` / `dev` / `test`, and exact modality masks across `P1–P4`.
2. **Source Layer:** One canonical 4-stage `std_ce` LoRA Q/V (`r=8, alpha=16`) expert bank trained per `(dataset, backbone, protocol, seed)`. All 9 merging operators consume the **exact same 4 expert deltas**. Separate trajectory tracks (`seq_ft`, `seq_replay256`, `joint_oracle`) train sequentially/jointly under the exact same `std_ce` recipe.
3. **Merge Layer:** Parameter-space consolidation via 9 verbatim-parity author operators:
   - `mean`, `exposure`, `ta_sum`, `ties20`, `magmax` (s=0.5), `magmax_s1` (s=1.0), `tsvm`, `opcm` (alpha=0.5), `dc_lora` (r=8, avg smoothing).
4. **Readout Layer (Strictly Partitioned by Data/Label Privileges):**
   - **Regime 0 — Weight-Only (`raw`):** Cosine prompt logits at native model scale (`0` calibration samples, `0` labels).
   - **Regime 1A — Zero-Label Unlabeled Centering (`unlab_center`):** Subtracts unlabeled mean logits estimated on `calibration_latest_views` (`0` labels).
   - **Regime 1B — Zero-Label, Zero-Exemplar `Zero-MNAR` (`zero_mnar`, Ours):** Accumulates canonical `StreamingClassBank` sufficient statistics during training (`0` stored raw exemplars), bridges missing `(m, y)` centroids via cross-modal orthogonal Procrustes `R_{m'->m}` from co-observed pairs, fits OAS Shrinkage-LDA on fully observed towers and isotropic vMF (`prec = I`) with Mahalanobis norm equalization on partially observed towers, and calibrates scalar nat-scales via unweighted proper-scoring train-only OOF NLL (`0` `D_cal` labels).
   - **Regime 2 — Labeled Calibration (`cal_bias`):** Fits convex class-balanced per-class logit offsets via L-BFGS (`0.01 * ||b||^2`) on `100%` labeled `calibration_latest_views`. (Also `seq_replay256` with `<= 256` labeled replay exemplars.)
   - **Regime 3 — Offline Diagnostic (`joint_oracle`):** Offline joint multi-stage training reported on its own separated rows.

*(Diagnostic note: `L_MNAR` support-aware propensity risk remains available for training-time collider-shift ablations on `seq_replay` / `joint`, and is NOT claimed as part of the frozen `Zero-MNAR` merging readout.)*

## 2. Evaluation Scope & Metrics

- **Main Same-Bank Metrics:** `Paired Macro-F1` (`8-cls` & `7-cls`), `Text Macro-F1`, `Image Macro-F1`, `3-View Mean Macro-F1`, `Early Recall`, `Late Recall`, and Expected Calibration Error `ECE (P)`.
- **Encoder Preservation Metrics:** Zero-shot `Unseen-Prompt OOD Macro-F1` and cross-modal retrieval `I->T R@1` / `T->I R@1`.
- **Statistical Rigor:** Paired bootstrap `1,000`-resample `95%` CIs and two-sided p-values; multi-seed `mean +/- SD` across seeds `11, 23, 37`.
- **Component Ablation (Fixed Bank + `tsvm`):** Bridge ablation (`none_zero_shot` vs `prompt_bridge` vs `cross_modal_transport`) crossed with partial-tower precision geometry (`raw_oas` vs `spectral_floor` vs `isotropic_vmf`).

## 3. Confirmatory Generalization Roadmap

1. **Same-Bank CrisisMMD `P1–P4` (`CLIP ViT-B/16`, `dev`, seed 11):** Direct apples-to-apples comparison of all 9 merging operators + 3 sequential/oracle arms across Regimes 0–3.
2. **Backbone Transfer:** Repeat identically on `SigLIP-B/16` (`dev`).
3. **Dataset Transfer:** Expand cleanly without algorithm changes to `N24News` and `UPMC Food-101` (`5-stage` & `10-stage`).
4. **Multi-Seed Stability:** Aggregate `seeds 11, 23, 37` (`mean +/- SD`) and unlock Locked `test` exactly once after pre-registration checks pass.
