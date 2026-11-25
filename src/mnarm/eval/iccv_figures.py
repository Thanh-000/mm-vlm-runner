"""Publication-ready visual diagnostics for the ICCV Same-Bank benchmark (PNG 300 DPI + PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnarm.methods import ALL_MERGE_METHODS

SEQ_ARMS = ("seq_ft", "seq_lmnar", "mg_clip", "pal_rls", "seq_replay256", "joint_oracle")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
        }
    )


def _save_pair(fig: plt.Figure, stem: Path) -> list[Path]:
    png_p, pdf_p = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png_p, dpi=300, facecolor="white")
    fig.savefig(pdf_p, facecolor="white")
    plt.close(fig)
    return [png_p, pdf_p]


def _anchor_proto(per_proto: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    key = "P4" if "P4" in per_proto else next(iter(per_proto))
    return key, per_proto[key]


def _plot_regimes(backbone: str, per_proto: dict[str, dict[str, Any]], fig_dir: Path) -> list[Path]:
    protos = list(per_proto.keys())
    order = [a for a in (list(ALL_MERGE_METHODS) + list(SEQ_ARMS)) if any(a in per_proto[p]["arms"] for p in protos)]
    y_pos = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(9.5, max(5.6, 0.36 * len(order) + 1.5)), layout="constrained")
    styles = [
        ("raw", "Raw (Weight-Only / SOTA)", "o", "#555555", 52),
        ("gr_clip", "+GR-CLIP (NeurIPS'25, 0-label)", "s", "#E69F00", 52),
        ("cal_bias", "+cal_bias (100% D_cal)", "D", "#56B4E9", 54),
        ("zero_mnar", "+CSS-CT (Ours, 0 D_cal)", "^", "#0072B2", 78),
    ]
    for reg, lbl, mk, col, sz in styles:
        xs, ys = [], []
        for idx, arm in enumerate(order):
            vals = [per_proto[p]["arms"][arm]["regimes"][reg]["paired"]["macro_f1"] * 100.0 for p in protos if arm in per_proto[p]["arms"] and reg in per_proto[p]["arms"][arm]["regimes"]]
            if vals:
                xs.append(sum(vals) / len(vals))
                ys.append(idx)
        if xs:
            ax.scatter(xs, ys, marker=mk, s=sz, color=col, edgecolors="white", linewidths=0.6, label=lbl, zorder=4 if reg == "zero_mnar" else 3)

    for idx, arm in enumerate(order):
        r_raw = [per_proto[p]["arms"][arm]["regimes"]["raw"]["paired"]["macro_f1"] * 100.0 for p in protos if arm in per_proto[p]["arms"] and "raw" in per_proto[p]["arms"][arm]["regimes"]]
        r_our = [per_proto[p]["arms"][arm]["regimes"]["zero_mnar"]["paired"]["macro_f1"] * 100.0 for p in protos if arm in per_proto[p]["arms"] and "zero_mnar" in per_proto[p]["arms"][arm]["regimes"]]
        if r_raw and r_our:
            x0, x1 = sum(r_raw) / len(r_raw), sum(r_our) / len(r_our)
            ax.plot([x0, x1], [idx, idx], color="#0072B2" if x1 >= x0 else "#D55E00", linewidth=2.0, alpha=0.75, zorder=2)

    n_merg = sum(1 for a in order if a in ALL_MERGE_METHODS)
    if 0 < n_merg < len(order):
        ax.axhline(n_merg - 0.5, color="#888888", linestyle=":", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Paired Macro-F1 (%) across " + ", ".join(protos))
    ax.set_title(f"{backbone}: Same-Bank Regimes Across 14 Mergers & VLM-CL Baselines")
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    return _save_pair(fig, fig_dir / f"fig1_same_bank_regimes_{backbone}")


def _plot_mnar_hubness(backbone: str, anchor_name: str, anchor: dict[str, Any], fig_dir: Path) -> list[Path]:
    arms = anchor["arms"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), layout="constrained")
    specs = [
        ("tsvm", "raw", "TSV-M Raw", "o", "#D55E00"),
        ("tsvm", "gr_clip", "TSV-M +GR", "s", "#E69F00"),
        ("dc_lora", "raw", "DC-LoRA Raw", "P", "#CC79A7"),
        ("dop", "raw", "DOP Raw", "X", "#777777"),
        ("wudi", "raw", "WUDI Raw", "v", "#888888"),
        ("mg_clip", "raw", "MG-CLIP", "d", "#D55E00"),
        ("pal_rls", "raw", "PAL-RLS", "p", "#56B4E9"),
        ("seq_ft", "raw", "Seq-FT Raw", "h", "#444444"),
        ("seq_lmnar", "raw", "Seq-LMNAR Raw", "v", "#009E73"),
        ("tsvm", "zero_mnar", "TSV-M +CSS-CT", "^", "#0072B2"),
        ("dc_lora", "zero_mnar", "DC-LoRA +CSS-CT", "*", "#009E73"),
    ]
    for arm, reg, lbl, mk, col in specs:
        pr = arms.get(arm, {}).get("regimes", {}).get(reg, {}).get("paired")
        if not pr:
            continue
        er = pr.get("early_recall")
        lr = pr.get("late_recall")
        if er is not None and lr is not None:
            x, y = lr * 100.0, er * 100.0
            ax1.scatter([x], [y], marker=mk, s=95 if reg == "zero_mnar" else 65, color=col, edgecolors="white", linewidths=0.6, label=lbl, zorder=4)
            ax1.annotate(lbl, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
        gini = pr.get("excess_gini")
        ece = pr.get("ece")
        if gini is not None and ece is not None:
            ax2.scatter([gini], [ece], marker=mk, s=95 if reg == "zero_mnar" else 65, color=col, edgecolors="white", linewidths=0.6, zorder=4)
            ax2.annotate(lbl, (gini, ece), xytext=(4, 3), textcoords="offset points", fontsize=7.5)

    ax1.plot([0, 100], [0, 100], color="#888888", linestyle="--", linewidth=1.0, label="Balanced parity (y=x)")
    ax1.set_xlim(-3, 105)
    ax1.set_ylim(-3, 105)
    ax1.set_xlabel("Late-Class Recall (%)")
    ax1.set_ylabel("Zero-Support Early-Class Recall (%)")
    ax1.set_title(f"{backbone} ({anchor_name}): MNAR Early-Class Collapse vs Recovery")
    ax1.legend(loc="lower right", fontsize=7.5, ncol=2, frameon=True)

    ax2.axvline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Excess Prediction Gini (Sato Hubness Concentration)")
    ax2.set_ylabel("Expected Calibration Error (ECE)")
    ax2.set_title(f"{backbone} ({anchor_name}): Hubness Reduction & Calibration")
    return _save_pair(fig, fig_dir / f"fig2_mnar_pathology_hubness_{backbone}")


def _plot_encoder_preservation(backbone: str, anchor_name: str, anchor: dict[str, Any], fig_dir: Path) -> list[Path]:
    arms = anchor["arms"]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), layout="constrained")
    enc_arms = [a for a in ("tsvm", "knots", "iso_cts", "wudi", "dop", "dc_lora", "seq_ft", "seq_lmnar", "mg_clip") if a in arms]
    raw_lbl, re_lbl = "Raw Adapted Encoder", "After O(d) Isometric Re-Anchor"
    for idx, arm in enumerate(enc_arms):
        ep = arms[arm]["encoder_preservation"]
        x0 = ep["retrieval"]["i2t_r1"] * 100.0
        y0 = ep["unseen_prompt_paired_f1"] * 100.0
        x1 = ep.get("retrieval_reanchored", ep["retrieval"])["i2t_r1"] * 100.0
        y1 = ep.get("unseen_prompt_reanchored_f1", ep["unseen_prompt_paired_f1"]) * 100.0
        ax.scatter([x0], [y0], marker="o", s=65, color="#D55E00", edgecolors="white", zorder=3, label=raw_lbl if idx == 0 else None)
        ax.scatter([x1], [y1], marker="*", s=135, color="#009E73", edgecolors="white", zorder=4, label=re_lbl if idx == 0 else None)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "#009E73", "lw": 1.6, "alpha": 0.85})
        ax.annotate(arm, (x1, y1), xytext=(5, 4), textcoords="offset points", fontsize=8.0)
    ax.set_xlabel("Zero-Shot Cross-Modal Retrieval I->T R@1 (%)")
    ax.set_ylabel("Unseen-Prompt OOD Paired Macro-F1 (%)")
    ax.set_title(f"{backbone} ({anchor_name}): Isometric O(d) Frame Re-Anchoring Preservation")
    ax.legend(loc="best", frameon=True)
    return _save_pair(fig, fig_dir / f"fig3_encoder_preservation_pareto_{backbone}")


def _plot_ablation_bootstrap(backbone: str, anchor_name: str, anchor: dict[str, Any], fig_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), layout="constrained")
    ab_map = anchor.get("ablation_tsvm", {})
    names = list(ab_map.keys())
    if names:
        xs = list(range(len(names)))
        w = 0.25
        for off, vw, lbl, col in ((-w, "paired", "Paired F1", "#0072B2"), (0.0, "text", "Text F1", "#009E73"), (w, "image", "Image F1", "#E69F00")):
            vals = [ab_map[n][vw]["macro_f1"] * 100.0 for n in names]
            ax1.bar([x + off for x in xs], vals, width=w, color=col, label=lbl)
        ax1.set_xticks(xs)
        ax1.set_xticklabels(names, rotation=22, ha="right", fontsize=8.5)
        ax1.set_ylabel("Macro-F1 (%)")
        ax1.set_title(f"{backbone} ({anchor_name}): CSS-CT Pillar Ablation (TSV-M Bank)")
        ax1.legend(loc="best", frameon=True, fontsize=8.5)

    boot = anchor.get("bootstrap", {})
    b_keys = list(boot.keys())
    if b_keys:
        ys = list(range(len(b_keys)))
        pts = [boot[k]["delta_macro_f1"] * 100.0 for k in b_keys]
        lo_err = [max(0.0, boot[k]["delta_macro_f1"] * 100.0 - boot[k]["ci95_low"] * 100.0) for k in b_keys]
        hi_err = [max(0.0, boot[k]["ci95_high"] * 100.0 - boot[k]["delta_macro_f1"] * 100.0) for k in b_keys]
        cols = ["#0072B2" if boot[k]["ci95_low"] > 0.0 else "#666666" for k in b_keys]
        for idx, (pt, le, he, col) in enumerate(zip(pts, lo_err, hi_err, cols)):
            ax2.errorbar([pt], [idx], xerr=[[le], [he]], fmt="o", color=col, ecolor=col, elinewidth=2.0, capsize=3.5)
        ax2.axvline(0.0, color="#D55E00", linestyle="--", linewidth=1.2)
        ax2.set_yticks(ys)
        ax2.set_yticklabels([k.replace("tsvm:zero_mnar_vs_", "vs ") for k in b_keys], fontsize=8.5)
        ax2.invert_yaxis()
        ax2.set_xlabel("Paired Macro-F1 Gain (pts, 95% Bootstrap CI)")
        ax2.set_title(f"{backbone} ({anchor_name}): Paired Bootstrap vs SOTA")
    return _save_pair(fig, fig_dir / f"fig4_ablation_bootstrap_{backbone}")


def generate_iccv_suite_figures(config: dict[str, Any], per_proto: dict[str, dict[str, Any]], fig_dir: Path) -> list[Path]:
    """Generate 4 multi-format (PNG 300 DPI + PDF) ICCV diagnostic figures from Same-Bank results."""
    if not per_proto:
        return []
    _configure_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    backbone = str(config["backbone"])
    anchor_name, anchor = _anchor_proto(per_proto)
    paths: list[Path] = []
    paths.extend(_plot_regimes(backbone, per_proto, fig_dir))
    paths.extend(_plot_mnar_hubness(backbone, anchor_name, anchor, fig_dir))
    paths.extend(_plot_encoder_preservation(backbone, anchor_name, anchor, fig_dir))
    paths.extend(_plot_ablation_bootstrap(backbone, anchor_name, anchor, fig_dir))
    return paths
