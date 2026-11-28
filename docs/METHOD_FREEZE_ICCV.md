# Method Freeze & Unified Same-Bank ICCV Evaluation Protocol (`CSS-CT` v3)

> **Status:** OFFICIAL POST-DEV FREEZE (`CSS-CT` v3 Canonical Specification — see `docs/FRAMEWORK_SPECIFICATION.md`). Hyperparameters, `Sim(d)` cone-aperture Procrustes, spectral-floored marginal epistemic precision, unbiased LOO sensor weighting, and all-class tangent-projected `Sigma_OAS` stat-OOF temperature scaling are permanently frozen after `dev` validation across all multi-altitude disaster benchmarks. Confirmatory `test` evaluation runs **once** via `--confirm-freeze` on the identical trained `train + cal` expert bank, writing separate `dev` (`ICCV_PROTOCOL_RESULTS.json`) and confirmatory `test` (`ICCV_PROTOCOL_TEST_RESULTS.json`) artifacts plus `ICCV_ECOSYSTEM_MASTER_SUMMARY.md`.

## 1. Four-Layer Scientific Separation & Information Regimes

Every comparison in this benchmark isolates **exactly one layer** at a time while holding the remaining three layers strictly identical:

1. **Contract Layer:** Fixed split manifests, zero UID/image/text overlap across `train` / `cal` / `dev` / `test`, and exact modality masks across `P1–P4`.
2. **Source Layer:** One canonical 4-stage `std_ce` LoRA Q/V (`r=8, alpha=16`) expert bank trained per `(dataset, backbone, protocol, seed)`. All 14 merging operators consume the **exact same 4 expert deltas**. Separate trajectory tracks (`seq_ft`, `seq_lmnar` [counterfactual `L_MNAR` loss], `seq_replay256`, `mg_clip`, `pal_rls`, `joint_oracle`) follow the exact same split contract.
3. **Merge Layer:** Parameter-space consolidation via 14 verbatim-parity author operators:
   - `mean`, `exposure`, `ta_sum`, `ties20`, `magmax` (s=0.5), `magmax_s1` (s=1.0), `knots` (ICLR'25), `tsvm` (CVPR'25), `iso_c` (ICML'25), `iso_cts` (ICML'25), `wudi` (ICML'25), `opcm` (ICLR'25, alpha=0.5), `dop` (NeurIPS'25, dual-SVD MGDA), `dc_lora` (CVPR'26, r=8).
4. **Readout Layer (Strictly Partitioned by Data/Label Privileges):**
   - **Regime 0 — Weight-Only / Baseline SOTA (`raw`):** Cosine prompt logits at native model scale (and native output for `mg_clip` [ICCV'25 Highlight] and `pal_rls` [PAL-RLS Analytic Head, 2025]).
   - **Regime 1A — Zero-Label Post-Hoc Gap/Mean Controls (`gr_clip`, `unlab_center`, `ncm`, `oas_lda`):** `gr_clip` (NeurIPS'25), `unlab_center`, plus `0 D_cal`-label generative baselines `ncm` (`Omega = I_d`) and `oas_lda` (`Sigma_OAS^-1` without cross-modal transport).
   - **Regime 1B — `CSS-CT` (`zero_mnar`, Ours — `0` `D_cal` labels, `0` stored exemplars, `0` Stage-4 probe reliance):**
     1. *Pillar 1 (`O(|Y|d + d^2)` Canonical Sufficiency):* Aligns stage adapters isometrically (`O(d)`) onto `theta_0` on `S^{d-1}`, accumulating exact Chan-Golub-LeVeque centroids, pooled scatter, and `O(d^2)` paired cross-moments `(N_TI, s_T, s_I, C_TI)` directly in `theta_0`.
     2. *Pillar 2 (Single Isometric `O(d)` Frame Re-Anchoring):* Solves uncentered orthogonal Procrustes `Q*_{m->0} in O(d)` from unlabeled anchors `P U U_cal` (`0` labels, `0` `fit4_recs`), mapping test features directly into canonical `B_0` with isometric norm preservation (avoiding hubness inflation) and partial cross-modal `I->T R@1` recovery.
     3. *Pillar 3 (Conjugate Empirical-Bayes, Epistemic Precision & All-`|Y|` Stat-OOF `tau_m`):* Computes unbiased cone-adjusted LOCO sensor weights `(w_P, w_CM)`, applies trace-preserving epistemic shrinkage `alpha_m` on partial towers plus unconditional spherical Mahalanobis norm equalization, and calibrates `tau_m` against all `|Y|` centroids purely from canonical bank sufficient statistics.
   - **Regime 2 — Labeled Calibration (`cal_bias`):** Fits convex class-balanced per-class logit offsets via L-BFGS (`0.01 * ||b||^2`) on `100%` labeled `calibration_latest_views`.
   - **Regime 3 — Offline Diagnostic (`joint_oracle`):** Offline joint multi-stage training reported on separated rows.

## 2. Evaluation Scope & Metrics

- **Closed-Set MNAR Metrics:** `Paired Macro-F1` (`8-cls` & `7-cls`), `Text Macro-F1`, `Image Macro-F1`, `3-View Mean Macro-F1`, `Early Recall`, `Late Recall`, `ECE (P)`, and `bECE (P)`.
- **Prompt-Paraphrase Transfer & Retrieval Recovery:** Zero-shot `Prompt-Transfer F1` (`Raw -> Re-Anchored`) and cross-modal retrieval `I->T R@1` (`Raw -> Re-Anchored`).
- **Component Ablation (Fixed Bank + `tsvm`):** `none_vmf`, `prompt_vmf`, `cm_raw_oas`, `cm_sfloor`, `cm_vmf`, `hybrid_vmf`, `css_no_reanchor`, and `css_ct`.
- **Visual Diagnostic Figures (`figures/`, PNG 300 DPI + PDF):**
  - `fig1_same_bank_regimes_{backbone}` — Mean Paired Macro-F1 across all 14 mergers + 7 SOTA/oracle tracks under `raw`, `gr_clip`, `ncm`, `cal_bias`, and `zero_mnar` (`CSS-CT`).
  - `fig2_mnar_pathology_hubness_{backbone}` — Early vs Late Recall collapse and Sato Excess Gini vs ECE diagnostics.
  - `fig3_encoder_preservation_pareto_{backbone}` — Raw vs `O(d)` Re-Anchored `I->T R@1` and `Prompt-Paraphrase F1`.
  - `fig4_ablation_bootstrap_{backbone}` — Pillar ablation ladder and Paired Bootstrap 95% CIs.
