"""Markdown, JSON, and booktabs LaTeX reporting for unified CSS-CT + 2025-2026 SOTA Same-Bank suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mnarm.core import write_json_atomic
from mnarm.methods import ALL_MERGE_METHODS

SEQ_ORACLE_ARMS = ("seq_ft", "seq_replay256", "mg_clip", "pal_rls", "joint_oracle")
ARM_REGIME_TAG = {
    **{m: "Weight-Only -> Merge" for m in ALL_MERGE_METHODS},
    "seq_ft": "Sequential (0 Replay)",
    "seq_replay256": "Sequential (Buffer<=256)",
    "mg_clip": "VLM-CL SOTA (ICCV'25)",
    "pal_rls": "Analytic MMCIL (2025)",
    "joint_oracle": "Offline Joint Oracle",
}


def _pf1(arec: dict[str, Any], regime: str) -> str:
    reg = arec["regimes"].get(regime)
    return f"{reg['paired']['macro_f1']*100:.2f}" if reg else "--"


def render_protocol_report(results: dict[str, Any], out_path: Path) -> None:
    """Write single-protocol Markdown report comparing 11 mergers + VLM-CL/Analytic SOTA + CSS-CT."""
    cfg = results["config"]
    lines = [
        f"# ICCV Same-Bank Report — {cfg['backbone']} {cfg['protocol']} (dev-Only, Strict Layer Isolation)\n",
        "- **Regimes:** `raw` · `gr_clip` (NeurIPS'25, 0-label) · `unlab_center` (0-label) · `zero_mnar` (**CSS-CT Ours: 0 D_cal labels, 0 exemplars**) · `cal_bias` (100% D_cal labels)\n",
        "| Arm | Track / Privilege | Raw (P) | +GR-CLIP (P) | +unlab (P) | +cal (P) | **+CSS-CT (P)** | +CSS-CT 3-View | Early Rec (Raw->CSS) | ECE (CSS) | Unseen OOD (Raw->Re) | I->T R@1 (Raw->Re) | thT/thI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|",
    ]
    for arm in list(ALL_MERGE_METHODS) + list(SEQ_ORACLE_ARMS):
        arec = results["arms"].get(arm)
        if not arec:
            continue
        raw_r = arec["regimes"]["raw"]
        zn_r = arec["regimes"].get("zero_mnar") or raw_r
        enc = arec["encoder_preservation"]
        th_t = arec["cone_drift"]["text"]["theta_deg"]
        th_i = arec["cone_drift"]["image"]["theta_deg"]
        er_str = f"{raw_r['paired']['early_recall']*100:.1f}->{zn_r['paired']['early_recall']*100:.1f}"
        ood_re = enc.get("unseen_prompt_reanchored_f1", enc["unseen_prompt_paired_f1"])
        r1_raw = enc["retrieval"]["i2t_r1"] * 100.0
        r1_re = enc.get("retrieval_reanchored", enc["retrieval"])["i2t_r1"] * 100.0
        lines.append(
            f"| `{arm}` | {ARM_REGIME_TAG[arm]} | {_pf1(arec, 'raw')} | {_pf1(arec, 'gr_clip')} | {_pf1(arec, 'unlab_center')} | "
            f"{_pf1(arec, 'cal_bias')} | **{_pf1(arec, 'zero_mnar')}** | {zn_r['summary']['view3_mean_f1']*100:.2f} | "
            f"{er_str} | {zn_r['paired']['ece']:.3f} | {enc['unseen_prompt_paired_f1']*100:.2f}->{ood_re*100:.2f} | "
            f"{r1_raw:.2f}->{r1_re:.2f} | {th_t:.1f}/{th_i:.1f} |"
        )

    lines.append("\n## Component Ablation (`tsvm` Fixed Bank, 0 D_cal Labels)\n")
    lines.append("| Ablation Variant | Paired F1 | Text F1 | Image F1 | 3-View Mean | ECE (P) | Early / Late Rec |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for ab_name, ab_rec in results.get("ablation_tsvm", {}).items():
        lines.append(
            f"| `{ab_name}` | **{ab_rec['paired']['macro_f1']*100:.2f}** | {ab_rec['text']['macro_f1']*100:.2f} | "
            f"{ab_rec['image']['macro_f1']*100:.2f} | {ab_rec['summary']['view3_mean_f1']*100:.2f} | "
            f"{ab_rec['paired']['ece']:.4f} | {ab_rec['paired']['early_recall']*100:.1f} / {ab_rec['paired']['late_recall']*100:.1f} |"
        )

    lines.append("\n## Paired Bootstrap Comparisons (1,000 resamples)\n")
    for bkey, brec in results.get("bootstrap", {}).items():
        lines.append(
            f"- `{bkey}`: delta = `{brec['delta_macro_f1']*100:+.2f}` pts, "
            f"95% CI = `[{brec['ci95_low']*100:+.2f}, {brec['ci95_high']*100:+.2f}]`, p = `{brec['p_value']:.4f}`"
        )
    text = "\n".join(lines) + "\n"
    if "$" in text:
        raise AssertionError("Forbidden dollar character detected in ICCV_PROTOCOL_REPORT.md")
    out_path.write_text(text, encoding="utf8")


def write_same_bank_suite_outputs(config: dict[str, Any], per_proto: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """Export unified P1-P4 summary Markdown, booktabs LaTeX table, and machine-readable JSON."""
    backbone = str(config["backbone"])
    protos = [p for p in ("P1", "P2", "P3", "P4") if p in per_proto] + [p for p in per_proto if p not in ("P1", "P2", "P3", "P4")]
    hdr_cols = " | ".join(f"{p} (Raw / +GR / +cal / **CSS-CT**)" for p in protos)
    sep_cols = " | ".join("---:" for _ in protos)
    lines = [
        f"# ICCV Same-Bank Summary — {backbone} (CrisisMMD `dev`, Same Source Bank)\n",
        f"- **Protocols:** `{', '.join(protos)}` | **Zero `D_cal` labels & O(|Y|d + d^2) moments** in `CSS-CT` (`zero_mnar`)\n",
        f"| Arm | {hdr_cols} | Mean (Raw / +GR / +cal / **CSS-CT**) |",
        f"|---|{sep_cols}|---:|",
    ]
    tex_rows: list[str] = []
    compact_protocols: dict[str, Any] = {
        p: {arm: {reg: float(arec["regimes"][reg]["paired"]["macro_f1"]) for reg in arec["regimes"]} for arm, arec in per_proto[p]["arms"].items()}
        for p in protos
    }

    for arm in list(ALL_MERGE_METHODS) + list(SEQ_ORACLE_ARMS):
        cells, raw_vals, cal_vals, our_vals, tex_cells = [], [], [], [], []
        for p in protos:
            arec = per_proto[p]["arms"].get(arm)
            if not arec:
                cells.append("--"); tex_cells.extend(["--", "--", "--"]); continue
            r0 = arec["regimes"]["raw"]["paired"]["macro_f1"] * 100.0
            rg = arec["regimes"]["gr_clip"]["paired"]["macro_f1"] * 100.0 if "gr_clip" in arec["regimes"] else float("nan")
            r2 = arec["regimes"]["cal_bias"]["paired"]["macro_f1"] * 100.0 if "cal_bias" in arec["regimes"] else float("nan")
            r1 = arec["regimes"]["zero_mnar"]["paired"]["macro_f1"] * 100.0 if "zero_mnar" in arec["regimes"] else float("nan")
            raw_vals.append(r0)
            if r2 == r2:
                cal_vals.append(r2)
            if r1 == r1:
                our_vals.append(r1)
            rg_s, r2_s, r1_s = (f"{rg:.2f}" if rg == rg else "--"), (f"{r2:.2f}" if r2 == r2 else "--"), (f"**{r1:.2f}**" if r1 == r1 else "--")
            cells.append(f"{r0:.2f} / {rg_s} / {r2_s} / {r1_s}")
            tex_cells.extend([f"{r0:.2f}", r2_s, f"\\textbf{{{r1:.2f}}}" if r1 == r1 else "--"])
        m_raw = f"{sum(raw_vals)/len(raw_vals):.2f}" if raw_vals else "--"
        m_cal = f"{sum(cal_vals)/len(cal_vals):.2f}" if cal_vals else "--"
        m_our = f"**{sum(our_vals)/len(our_vals):.2f}**" if our_vals else "--"
        lines.append(f"| `{arm}` | {' | '.join(cells)} | {m_raw} / {m_cal} / {m_our} |")
        tex_label = arm.replace("_", "\\_")
        m_our_tex = f"\\textbf{{{sum(our_vals)/len(our_vals):.2f}}}" if our_vals else "--"
        tex_rows.append(f"\\texttt{{{tex_label}}} & {' & '.join(tex_cells)} & {m_raw} & {m_cal} & {m_our_tex} \\\\")
        if arm == ALL_MERGE_METHODS[-1]:
            tex_rows.append("\\midrule")

    md_text = "\n".join(lines) + "\n"
    if "$" in md_text:
        raise AssertionError("Forbidden dollar character detected in ICCV_SAME_BANK summary markdown")
    (out_dir / f"ICCV_SAME_BANK_{backbone}_SUMMARY.md").write_text(md_text, encoding="utf8")

    tex_cols = "l" + "rrr" * len(protos) + "rrr"
    p_hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{{p}}}" for p in protos) + " & \\multicolumn{3}{c}{Mean}"
    sub_hdr = " & ".join("Raw & +cal & CSS-CT" for _ in range(len(protos) + 1))
    tex_body = "\n".join([
        "% Auto-generated booktabs table (Paired Macro-F1 % across identical expert bank)",
        f"\\begin{{tabular}}{{{tex_cols}}}",
        "\\toprule",
        f"Method & {p_hdr} \\\\",
        f" & {sub_hdr} \\\\",
        "\\midrule",
        *tex_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]) + "\n"
    (out_dir / f"ICCV_SAME_BANK_{backbone}_TABLE.tex").write_text(tex_body, encoding="utf8")

    payload = {"schema": "mnarm-iccv-same-bank-v2", "backbone": backbone, "allow_test": False, "protocols": compact_protocols}
    write_json_atomic(out_dir / f"ICCV_SAME_BANK_{backbone}_RESULTS.json", payload)
    return payload
