"""First-principles MNAR continual learning, merging, re-anchoring, and SOTA baselines."""

from mnarm.methods.bayes_transport import SelfNormalizedBayesTransportHead
from mnarm.methods.l_mnar import (
    analyze_co_observation_graph,
    compute_multimodal_lmnar_loss,
    compute_standard_ce_loss,
    tower_support_propensity_loss,
)
from mnarm.methods.logit_controls import (
    apply_bias,
    fit_cal_bias,
    fit_gr_clip_gap,
    fit_unlabeled_center,
    gr_clip_logits,
)
from mnarm.methods.merging import ALL_MERGE_METHODS, merge_task_deltas
from mnarm.methods.reanchor import apply_reanchor, fit_canonical_reanchor, verify_projection_weight_fold
from mnarm.methods.spherical_stats import _identity_anchored_procrustes, _procrustes_from_cross_moments
from mnarm.methods.streaming_bank import StreamingClassBank
from mnarm.methods.vlm_cl_sota import (
    IntraModalCosineClassifier,
    PALRLSClassifier,
    mg_clip_eval_logits,
    train_mg_clip_stage,
)

__all__ = [
    "ALL_MERGE_METHODS",
    "IntraModalCosineClassifier",
    "PALRLSClassifier",
    "SelfNormalizedBayesTransportHead",
    "StreamingClassBank",
    "analyze_co_observation_graph",
    "apply_bias",
    "apply_reanchor",
    "compute_multimodal_lmnar_loss",
    "compute_standard_ce_loss",
    "fit_cal_bias",
    "fit_canonical_reanchor",
    "fit_gr_clip_gap",
    "fit_unlabeled_center",
    "gr_clip_logits",
    "merge_task_deltas",
    "mg_clip_eval_logits",
    "tower_support_propensity_loss",
    "train_mg_clip_stage",
    "verify_projection_weight_fold",
]
