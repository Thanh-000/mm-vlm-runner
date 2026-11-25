# Method Freeze & Unified Same-Bank ICCV Evaluation Protocol (`CSS-CT`)

> **Status:** FROZEN (`METHOD_FREEZE`). No hyperparameter sweeps, post-hoc dataset patches, or `test` split reads are permitted prior to final confirmatory reporting.

## 1. Four-Layer Scientific Separation & Information Regimes

Every comparison in this benchmark isolates **exactly one layer** at a time while holding the remaining three layers strictly identical:

1. **Contract Layer:** Fixed split manifests, zero UID/image/text overlap across `train` / `cal` / `dev` / `test`, and exact modality masks across `P1–P4`.
2. **Source Layer:** One canonical 4-stage `std_ce` LoRA Q/V (`r=8, alpha=16`) expert bank trained per `(dataset, backbone, protocol, seed)`. All 14 merging operators consume the **exact same 4 expert deltas**. Separate trajectory tracks (`seq_ft`, `seq_replay256`, `mg_clip`, `pal_rls`, `joint_oracle`) follow the exact same split contract.
3. **Merge Layer:** Parameter-space consolidation via 14 verbatim-parity author operators:
   - `mean`, `exposure`, `ta_sum`, `ties20`, `magmax` (s=0.5), `magmax_s1` (s=1.0), `knots` (ICLR'25), `tsvm` (CVPR'25), `iso_c` (ICML'25), `iso_cts` (ICML'25), `wudi` (ICML'25), `opcm` (ICLR'25, alpha=0.5), `dop` (NeurIPS'25, dual-SVD MGDA), `dc_lora` (CVPR'26, r=8).
4. **Readout Layer (Strictly Partitioned by Data/Label Privileges):**
   - **Regime 0 — Weight-Only / Baseline SOTA (`raw`):** Cosine prompt logits at native model scale (and native VLM-CL/Analytic output for `mg_clip` [ICCV'25 Highlight] and `pal_rls` [Analytic MMCIL'25]).
   - **Regime 1A — Zero-Label Post-Hoc Gap/Mean Controls (`gr_clip`, `unlab_center`):** `gr_clip` (NeurIPS'25 linear modality-gap centering) and `unlab_center` using `0` calibration labels.
   - **Regime 1B — `CSS-CT` (`zero_mnar`, Ours — `0` `D_cal` labels, `0` stored exemplars):**
     1. *Pillar 1 (`O(|Y|d + d^2)` Canonical Sufficiency):* Accumulates per-class tower moments and exact `O(d^2)` paired cross-moments `(N_TI, s_T, s_I, C_TI)` in `theta_0` with zero stored sample vectors.
     2. *Pillar 2 (Isometric `O(d)` Frame Re-Anchoring):* Solves uncentered orthogonal Procrustes `Q*_{m->0} in O(d)` to pull adapted towers back onto canonical `theta_0` with zero Sato et al. hubness bias, restoring open-vocab `Unseen-Prompt OOD` and `I->T R@1`.
     3. *Pillar 3 (Conjugate Empirical-Bayes & Marginal Epistemic Precision):* Estimates cone-baseline-adjusted leave-one-class-out sensor weights `(w_P, w_CM)` fusing prompt prior, cross-modal transport, and in-domain mean, while marginalizing epistemic transport variance `(Sigma_m + sigma^2_trans I_d)^{-1}` to eliminate precision-nullspace collapse and tune nat-scale via unweighted OOF NLL.
   - **Regime 2 — Labeled Calibration (`cal_bias`):** Fits convex class-balanced per-class logit offsets via L-BFGS (`0.01 * ||b||^2`) on `100%` labeled `calibration_latest_views`.
   - **Regime 3 — Offline Diagnostic (`joint_oracle`):** Offline joint multi-stage training reported on separated rows.

## 2. Evaluation Scope & Metrics

- **Closed-Set MNAR Metrics:** `Paired Macro-F1` (`8-cls` & `7-cls`), `Text Macro-F1`, `Image Macro-F1`, `3-View Mean Macro-F1`, `Early Recall`, `Late Recall`, and `ECE (P)`.
- **Open-Vocabulary & Retrieval Restoration:** Zero-shot `Unseen-Prompt OOD Macro-F1` (`Raw -> Re-Anchored`) and cross-modal retrieval `I->T R@1` (`Raw -> Re-Anchored`).
- **Component Ablation (Fixed Bank + `tsvm`):** `none_vmf`, `prompt_vmf`, `cm_raw_oas`, `cm_sfloor`, `cm_vmf`, `hybrid_vmf`, and `css_ct`.
- **Visual Diagnostic Figures (`figures/`, PNG 300 DPI + PDF):**
  - `fig1_same_bank_regimes_{backbone}` — Mean Paired Macro-F1 across all 14 mergers + 5 SOTA/oracle tracks under `raw`, `gr_clip`, `cal_bias`, and `zero_mnar` (`CSS-CT`).
  - `fig2_mnar_pathology_hubness_{backbone}` — Early vs Late Recall collapse and Sato Excess Gini vs ECE diagnostics.
  - `fig3_encoder_preservation_pareto_{backbone}` — Raw vs `O(d)` Re-Anchored `I->T R@1` and `Unseen-Prompt OOD F1`.
  - `fig4_ablation_bootstrap_{backbone}` — Pillar ablation ladder and Paired Bootstrap 95% CIs.
