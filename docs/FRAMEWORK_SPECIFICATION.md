# Canonical `CSS-CT` Framework Specification (v2)

> **Scope:** Formal mathematical and system contract for **Zero-Support Missing-Not-At-Random (MNAR)** continual multimodal Vision-Language Model (VLM) adaptation and weight consolidation (**Canonical Spherical Sufficiency & Conjugate Transport — `CSS-CT`**).

---

## 1. Problem Setting & Privilege Contracts

We consider a sequential multimodal stream over `T` chronological stages `t = 1, ..., T`. Each sample `(x_T, x_I, M, y) in D_t` carries a label `y in Y_seen(t)` and an observed modality mask `M subseteq {T, I}` (`"text"`, `"image"`, or paired `("text", "image")`).

- **Zero-Support Structural MNAR (`P_train(m in M | y = c) = 0`):** There exist classes `c notin S_m` whose modality `m` is structurally absent across **all** training stages `D_1 .. D_T` (for example, `Y_early` observed Image-only in `P4`, Text-only in `P3`, or disconnected across early stages in `P1`/`P2`).
- **Co-Observation Graph `G_obs`:** Edges connect classes co-occurring within any observed mask `M`; disconnected components induce unconstrained inter-component logit gauges under naive discriminative heads.
- **Strict Resource & Label Privileges (`zero_mnar`):**
  1. **Zero Stored Exemplars (`0` replay buffer):** Past training inputs are discarded immediately after stage sufficient-statistics accumulation.
  2. **Zero Calibration Labels (`0` `D_cal` labels):** Unlabeled anchor evaluation streams `U_cal` (`calibration_latest_views`) and class prompt prototypes `P` (`orig_prompts`) may be inspected as *features only*; labels in `D_cal` are strictly forbidden.
  3. **Model-Agnostic Plug-and-Play Consolidation:** Any weight merger `Delta W_merged = M(Delta W_1..T)` or sequential adapter trajectory can be evaluated directly under `CSS-CT` without retraining.

---

## 2. Pillar 1 — Exemplar-Free Canonical `theta_0` Bank (`O(|Y|d + d^2)`)

All historical evidence lives permanently inside the frozen pretrained canonical frame `theta_0` on the unit hypersphere `S^{d-1}` (`StreamingClassBank`).

At the end of stage `t`, for each modality `m in {T, I}` with stage adapter features `U_m^{(t)}` and frozen base features `U_0^{(t)}`:

1. Compute the single **isometric uncentered orthogonal map** `Q_{m,t->0} in O(d)` (`fit_canonical_reanchor(U_0^{(t)}, U_m^{(t)})`) and rotate normalized embeddings onto `theta_0`: `u_tilde = u_m^{(t)} @ Q_{m,t->0}^T`. Because `Q in O(d)` preserves `L2` norms (`||u_tilde||_2 = 1.0`), no mean-shift off the unit sphere occurs.
2. Accumulate exact running class centroids `(N_{m,c}, mu_{m,c}^{(0)})` for `c in S_m`, total sample counts, pooled within-class scatter `S_m^{(0)} in R^{d x d}` (degrees of freedom `dof_m`), and paired cross-moments from paired `theta_0` base embeddings:
   `B_0 = { (N_{m,c}, mu_{m,c}^{(0)})_{c in S_m}, S_m^{(0)}, (N_TI, s_T^{(0)}, s_I^{(0)}, C_TI^{(0)}) }`
3. **Exact Constant Memory Footprint:** `memory_floats() <= O(|Y|d + d^2)` independent of dataset cardinality `N`. No per-sample vectors or Stage-4 training records are retained (`0` exemplars).

---

## 3. Pillar 2 — Single Isometric `O(d)` Frame Re-Anchoring

Given any consolidated or continually adapted encoder `theta_merged` (`14` weight mergers, `seq_ft`, `seq_lmnar`, `mg_clip`, etc.), we bring the encoder back into `theta_0` **once** rather than repeatedly transporting the bank:

1. **Unlabeled Canonical Anchors (`P U U_cal`, zero labels):** Extract unadapted `theta_0` and adapted `theta_merged` embeddings on unlabeled calibration views `U_cal` (plus class prompts `P` for text).
2. **Closed-Form Orthogonal Procrustes:**
   `Q*_{m->0} = argmax_{Q in O(d)} Tr(Q^T (lambda_id I_d + U_0^T U_merged)) = U_svd V_svd^T`
3. **Direct Canonical Lookup (`canonical_stats`):** During inference, test features are rotated once by `u_canonical = l2_normalize(u_merged @ Q*_{m->0}^T)` and scored directly against canonical `B_0` in `theta_0` — eliminating redundant Procrustes steps.
4. **Zero-Hubness & Projection Weight Folding:** Because `Q*_{m->0} in O(d)` is an exact isometry on `S^{d-1}`, it avoids Sato et al. (2026) linear-centering hubness collapse (`Excess Gini`), folds cleanly into CLIP/SigLIP projection layers with zero extra runtime parameters (`verify_projection_weight_fold`), and restores Open-Vocabulary `Unseen-Prompt OOD` and cross-modal `I->T R@1`.

---

## 4. Pillar 3 — Conjugate Empirical-Bayes Transport, Marginal Precision & Sufficient-Statistic OOF `tau_m`

Inside `theta_0`, for each tower `m in {T, I}` (`SelfNormalizedBayesTransportHead`):

1. **Two Independent Cross-Modal Sensors:**
   - **Prompt Sensor (`prompt`):** Orthogonal Procrustes alignment from frozen prompt prototypes `P_{obs}` onto observed centroids `mu_{m, obs}^{(0)}`.
   - **Cross-Modal Moment Sensor (`cross`):** Exact covariance Procrustes computed solely from `O(d^2)` sufficient moments `(N_TI, s_T^{(0)}, s_I^{(0)}, C_TI^{(0)})` transporting observed centroids from the complementary tower `m'` to `m`.
2. **Cone-Adjusted Leave-One-Class-Out Reliability Weights:** Removing the shared global cone cosine baseline `r_base`, leave-one-class-out accuracies yield effective reliabilities `r_P^eff, r_CM^eff` and conjugate weights `w_s = r_s^eff / (1 - (r_s^eff)^2)`, synthesizing missing class centroids `c notin S_m` at average observed norm `nu_bar_m`, plus empirical-Bayes shrinkage on low-count observed classes (`10 / (n_c + 10)`).
3. **Marginal Epistemic Precision (`Sigma_m + sigma^2_trans I_d`) & Mahalanobis Equalization:** When `S_m != Y_seen`, OAS covariance inversion without epistemic transport variance `sigma^2_trans` explodes along unobserved nullspace directions. `CSS-CT` marginalizes `Omega_m = (Sigma_hat_m + max(sigma^2_trans, lambda_eff) I_d)^{-1}` and equalizes quadratic Mahalanobis norms `||mu_c||_{Omega_m}` between observed and transported classes so missing classes suffer zero norm penalty.
4. **Stage-Agnostic Parametric OOF Temperature Calibration (`_fit_stat_oof_tau`):** Calibrates scalar `tau_m > 0` in exact nat-scale via L-BFGS NLL over 4-fold parametric out-of-fold draws purely from `(N_{m,c}, mu_{m,c}^{(0)}, S_m^{(0)} / dof_m)` — completely eliminating Stage-4 sample dependence and fixing `tau_m = 0.05` collapse whenever the final stage is unimodal (`P1`, `P2`).
5. **Multimodal Fusion:** `ell_m(x; c) = tau_m * (u_canonical^T Omega_m mu_hat_{m,c}^{(0)} - 0.5 mu_hat_{m,c}^{(0)T} Omega_m mu_hat_{m,c}^{(0)})`, summed across available test views `ell_paired = ell_T + ell_I`.

---

## 5. Counterfactual `L_MNAR` Track & Four-Layer Causal Benchmark

While `CSS-CT` operates purely plug-and-play without changing stage training objectives, `mnarm` also provides **training-time `L_MNAR`** (`seq_lmnar`, [l_mnar.py](file:///src/mnarm/methods/l_mnar.py)) restricting tower gradients to each modality's supported class set `S_m` with propensity log-offset `log pi_{m,c}`. Including `seq_lmnar` alongside `seq_ft` in `mnarm.runners.iccv_same_bank` directly tests a critical causal counterfactual: **removing negative-collider gradients during training alone cannot predict classes with `P_train(M | y) = 0` unless paired with `CSS-CT`'s cross-modal density bridge.**

Every benchmark verdict separates **Contract -> Source Bank -> Merge Operator (`14` verbatim methods) -> Readout Regime (`raw`, `gr_clip`, `unlab_center`, `zero_mnar` [`CSS-CT`], `cal_bias`)** under strict `allow_test: false` development hygiene.
