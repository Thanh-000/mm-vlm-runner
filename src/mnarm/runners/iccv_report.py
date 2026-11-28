"""Markdown, JSON, and booktabs LaTeX reporting for unified CSS-CT + 2025-2026 SOTA Same-Bank suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mnarm.core import write_json_atomic
from mnarm.eval import generate_iccv_suite_figures
from mnarm.methods import ALL_MERGE_METHODS

SEQ_ORACLE_ARMS = ("seq_ft", "seq_magmax", "seq_lmnar", "seq_replay256", "mg_clip", "pal_rls", "joint_oracle")
ARM_REGIME_TAG = {
    **{m: "Weight-Only -> Merge" for m in ALL_MERGE_METHODS},
    "magmax": "MaxAbs-Ind (s=0.5) -> Merge",
    "seq_ft": "Sequential (0 Replay)",
    "seq_magmax": "Continual Merging SOTA (ECCV'24)",
    "seq_lmnar": "Sequential L_MNAR (0 Replay)",
    "seq_replay256": "Sequential (Buffer<=256)",
    "mg_clip": "VLM-CL SOTA (ICCV'25)",
    "pal_rls": "PAL-RLS (Analytic Head, 2025)",
    "joint_oracle": "Offline Joint Oracle",
}


def _pf1(arec: dict[str, Any], regime: str) -> str:
    reg = arec["regimes"].get(regime)
    return f"{reg['paired']['macro_f1']*100:.2f}" if reg else "--"


def render_protocol_report(results: dict[str, Any], out_path: Path) -> None:
    """Write single-protocol Markdown report comparing 14 mergers + VLM-CL/Analytic SOTA + CSS-CT."""
    cfg = results["config"]
    ds_name = str(cfg.get("dataset", "crisismmd"))
    split_name = str(results.get("split", "dev"))
    split_tag = "Confirmatory Test, Frozen CSS-CT v3" if split_name == "test" else "dev-Only, Strict Layer Isolation"
    lines = [
        f"# ICCV Same-Bank Report — {ds_name} {cfg['backbone']} {cfg['protocol']} ({split_tag})\n",
        "- **Regimes:** `raw` · `gr_clip` (NeurIPS'25, 0-label) · `unlab_center` (0-label) · `ncm` / `oas_lda` (0 D_cal fair head) · `zero_mnar` (**CSS-CT Ours: 0 D_cal labels, 0 exemplars**) · `cal_bias` (100% D_cal labels)\n",
        "| Arm | Track / Privilege | Raw (P) | +GR-CLIP (P) | +unlab (P) | +NCM (P) | +OAS-LDA (P) | +cal (P) | **+CSS-CT (P)** | +CSS-CT 3-View | Early Rec (Raw->CSS) | ECE (CSS) | Prompt-Transfer F1 (Raw->Re) | I->T R@1 (Raw->Re) | thT/thI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|",
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
            f"{_pf1(arec, 'ncm')} | {_pf1(arec, 'oas_lda')} | {_pf1(arec, 'cal_bias')} | **{_pf1(arec, 'zero_mnar')}** | "
            f"{zn_r['summary']['view3_mean_f1']*100:.2f} | {er_str} | {zn_r['paired']['ece']:.3f} | {enc['unseen_prompt_paired_f1']*100:.2f}->{ood_re*100:.2f} | "
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
        p_h = brec.get("p_value_holm", brec["p_value"])
        lines.append(
            f"- `{bkey}`: delta = `{brec['delta_macro_f1']*100:+.2f}` pts, "
            f"95% CI = `[{brec['ci95_low']*100:+.2f}, {brec['ci95_high']*100:+.2f}]`, p_raw = `{brec['p_value']:.4f}`, p_holm = `{p_h:.4f}`"
        )
    text = "\n".join(lines) + "\n"
    if "$" in text:
        raise AssertionError("Forbidden dollar character detected in ICCV_PROTOCOL_REPORT.md")
    out_path.write_text(text, encoding="utf8")


def write_same_bank_suite_outputs(
    config: dict[str, Any], per_proto: dict[str, dict[str, Any]], out_dir: Path, split_name: str = "dev",
) -> dict[str, Any]:
    """Export unified P1-P4 summary Markdown, booktabs LaTeX table, and machine-readable JSON."""
    backbone, ds_name = str(config["backbone"]), str(config.get("dataset", "crisismmd"))
    is_test = str(split_name) == "test"
    split_hdr = "`test` Confirmatory, Same Source Bank" if is_test else "`dev`, Same Source Bank"
    protos = [p for p in ("P1", "P2", "P3", "P4") if p in per_proto] + [p for p in per_proto if p not in ("P1", "P2", "P3", "P4")]
    hdr_cols = " | ".join(f"{p} (Raw / +NCM / +cal / **CSS-CT**)" for p in protos)
    sep_cols = " | ".join("---:" for _ in protos)
    lines = [
        f"# ICCV Same-Bank Summary — {ds_name} {backbone} ({split_hdr})\n",
        f"- **Protocols:** `{', '.join(protos)}` | **Zero `D_cal` labels & O(|Y|d + d^2) moments** in `CSS-CT` (`zero_mnar`)\n",
        f"| Arm | {hdr_cols} | Mean (Raw / +NCM / +cal / **CSS-CT**) |",
        f"|---|{sep_cols}|---:|",
    ]
    tex_rows: list[str] = []
    compact_protocols: dict[str, Any] = {
        p: {arm: {reg: float(arec["regimes"][reg]["paired"]["macro_f1"]) for reg in arec["regimes"]} for arm, arec in per_proto[p]["arms"].items()}
        for p in protos
    }

    for arm in list(ALL_MERGE_METHODS) + list(SEQ_ORACLE_ARMS):
        cells, raw_vals, ncm_vals, cal_vals, our_vals, tex_cells = [], [], [], [], [], []
        for p in protos:
            arec = per_proto[p]["arms"].get(arm)
            if not arec:
                cells.append("--"); tex_cells.extend(["--", "--", "--", "--"]); continue
            r0 = arec["regimes"]["raw"]["paired"]["macro_f1"] * 100.0
            rn = arec["regimes"]["ncm"]["paired"]["macro_f1"] * 100.0 if "ncm" in arec["regimes"] else float("nan")
            r2 = arec["regimes"]["cal_bias"]["paired"]["macro_f1"] * 100.0 if "cal_bias" in arec["regimes"] else float("nan")
            r1 = arec["regimes"]["zero_mnar"]["paired"]["macro_f1"] * 100.0 if "zero_mnar" in arec["regimes"] else float("nan")
            raw_vals.append(r0)
            if rn == rn:
                ncm_vals.append(rn)
            if r2 == r2:
                cal_vals.append(r2)
            if r1 == r1:
                our_vals.append(r1)
            rn_s, r2_s, r1_s = (f"{rn:.2f}" if rn == rn else "--"), (f"{r2:.2f}" if r2 == r2 else "--"), (f"**{r1:.2f}**" if r1 == r1 else "--")
            cells.append(f"{r0:.2f} / {rn_s} / {r2_s} / {r1_s}")
            tex_cells.extend([f"{r0:.2f}", rn_s, r2_s, f"\\textbf{{{r1:.2f}}}" if r1 == r1 else "--"])
        m_raw = f"{sum(raw_vals)/len(raw_vals):.2f}" if raw_vals else "--"
        m_ncm = f"{sum(ncm_vals)/len(ncm_vals):.2f}" if ncm_vals else "--"
        m_cal = f"{sum(cal_vals)/len(cal_vals):.2f}" if cal_vals else "--"
        m_our = f"**{sum(our_vals)/len(our_vals):.2f}**" if our_vals else "--"
        lines.append(f"| `{arm}` | {' | '.join(cells)} | {m_raw} / {m_ncm} / {m_cal} / {m_our} |")
        tex_label = arm.replace("_", "\\_")
        m_our_tex = f"\\textbf{{{sum(our_vals)/len(our_vals):.2f}}}" if our_vals else "--"
        tex_rows.append(f"\\texttt{{{tex_label}}} & {' & '.join(tex_cells)} & {m_raw} & {m_ncm} & {m_cal} & {m_our_tex} \\\\")
        if arm == ALL_MERGE_METHODS[-1]:
            tex_rows.append("\\midrule")

    md_text = "\n".join(lines) + "\n"
    if "$" in md_text:
        raise AssertionError("Forbidden dollar character detected in ICCV_SAME_BANK summary markdown")
    suf = "_TEST" if is_test else ""
    (out_dir / f"ICCV_SAME_BANK_{backbone}{suf}_SUMMARY.md").write_text(md_text, encoding="utf8")

    tex_cols = "l" + "rrrr" * len(protos) + "rrrr"
    p_hdr = " & ".join(f"\\multicolumn{{4}}{{c}}{{{p}}}" for p in protos) + " & \\multicolumn{4}{c}{Mean}"
    sub_hdr = " & ".join("Raw & +NCM & +cal & CSS-CT" for _ in range(len(protos) + 1))
    tex_body = "\n".join([
        f"% Auto-generated booktabs table ({split_name} Paired Macro-F1 % across identical expert bank)",
        f"\\begin{{tabular}}{{{tex_cols}}}",
        "\\toprule",
        f"Method & {p_hdr} \\\\",
        f" & {sub_hdr} \\\\",
        "\\midrule",
        *tex_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]) + "\n"
    (out_dir / f"ICCV_SAME_BANK_{backbone}{suf}_TABLE.tex").write_text(tex_body, encoding="utf8")

    fig_paths = generate_iccv_suite_figures(config, per_proto, out_dir / ("figures_test" if is_test else "figures"))
    payload = {
        "schema": "mnarm-iccv-same-bank-v2",
        "backbone": backbone,
        "split": str(split_name),
        "allow_test": bool(is_test),
        "protocols": compact_protocols,
        "figures": [p.name for p in fig_paths],
    }
    write_json_atomic(out_dir / f"ICCV_SAME_BANK_{backbone}{suf}_RESULTS.json", payload)
    return payload


def write_ecosystem_master_summary(out_root: Path) -> Path:
    """Aggregate 4-dataset x 2-backbone x 4-protocol dev & test results into ICCV master tables."""
    import json
    master_data: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    for sp, fname in (("dev", "ICCV_PROTOCOL_RESULTS.json"), ("test", "ICCV_PROTOCOL_TEST_RESULTS.json")):
        for ds in ("crisismmd", "dmd", "incidents_vqa", "disaster_m3"):
            for bb in ("clip_b16", "siglip_b16"):
                bdir = "clip_same_bank" if bb == "clip_b16" else "siglip_same_bank"
                sdir = (out_root / bdir) if (ds == "crisismmd" and (out_root / bdir).is_dir()) else (out_root / ds / bdir)
                p_jsons = [(p, sdir / f"{bb}_{p.lower()}" / fname) for p in ("P1", "P2", "P3", "P4") if (sdir / f"{bb}_{p.lower()}" / fname).is_file()]
                if not p_jsons:
                    continue
                vals = {k: [] for k in ("raw", "gr_clip", "unlab_center", "ncm", "oas_lda", "cal_bias", "zero_mnar")}
                v3 = {k: [] for k in ("ncm", "cal_bias", "zero_mnar")}
                wins = {"raw": 0, "gr_clip": 0, "ncm": 0, "cal_bias": 0, "tot": 0}
                refs = {k: [] for k in ("tsvm_css", "seq_mm_ncm", "lmnar_css", "mg_clip", "pal_rls", "joint_ncm")}
                for _, jp in p_jsons:
                    res = json.loads(jp.read_text(encoding="utf8"))
                    arms = res.get("arms", {})
                    for m in ALL_MERGE_METHODS:
                        rg = arms[m]["regimes"]
                        zp = rg["zero_mnar"]["paired"]["macro_f1"]
                        wins["tot"] += 1
                        for wk in ("raw", "gr_clip", "ncm", "cal_bias"):
                            if wk in rg and zp > rg[wk]["paired"]["macro_f1"]:
                                wins[wk] += 1
                        for k in vals:
                            if k in rg:
                                vals[k].append(rg[k]["paired"]["macro_f1"] * 100.0)
                        for k in v3:
                            if k in rg:
                                v3[k].append(rg[k]["summary"]["view3_mean_f1"] * 100.0)
                    refs["tsvm_css"].append(arms["tsvm"]["regimes"]["zero_mnar"]["paired"]["macro_f1"] * 100.0)
                    if "seq_magmax" in arms:
                        mm_rg = arms["seq_magmax"]["regimes"]
                        refs["seq_mm_ncm"].append((mm_rg.get("ncm") or mm_rg["raw"])["paired"]["macro_f1"] * 100.0)
                    refs["lmnar_css"].append(arms["seq_lmnar"]["regimes"]["zero_mnar"]["paired"]["macro_f1"] * 100.0)
                    if "mg_clip" in arms:
                        refs["mg_clip"].append(arms["mg_clip"]["regimes"]["raw"]["paired"]["macro_f1"] * 100.0)
                    if "pal_rls" in arms:
                        refs["pal_rls"].append(arms["pal_rls"]["regimes"]["raw"]["paired"]["macro_f1"] * 100.0)
                    jn_rg = arms["joint_oracle"]["regimes"]
                    refs["joint_ncm"].append((jn_rg.get("ncm") or jn_rg["raw"])["paired"]["macro_f1"] * 100.0)
                avg = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
                master_data[sp].append({
                    "dataset": ds, "backbone": bb, "n_protocols": len(p_jsons),
                    "raw": avg(vals["raw"]), "gr_clip": avg(vals["gr_clip"]), "unlab_center": avg(vals["unlab_center"]),
                    "ncm": avg(vals["ncm"]), "oas_lda": avg(vals["oas_lda"]),
                    "cal_bias": avg(vals["cal_bias"]), "css_ct": avg(vals["zero_mnar"]),
                    "v3_cal": avg(v3["cal_bias"]), "v3_css": avg(v3["zero_mnar"]),
                    "wins": wins, **{k: avg(v) for k, v in refs.items()},
                })

    md_lines = ["# ICCV 4-Benchmark Multi-Altitude Ecosystem Master Summary (`CSS-CT` v2.2-fair)\n"]
    tex_lines = ["% Auto-generated 4-benchmark ICCV Master Booktabs Table", "\\begin{tabular}{llrrrrrrrrrrrr}", "\\toprule", "Split / Dataset & Backbone & Raw & +GR & +NCM & +OAS & +cal & **CSS-CT** & TSV+CSS & SeqMM+NCM & MG-CLIP & PAL-RLS & Joint+NCM \\\\", "\\midrule"]
    for sp in ("test", "dev"):
        rows = master_data[sp]
        if not rows:
            continue
        md_lines.append(f"## Split: `{sp}` ({'Confirmatory Frozen Test' if sp == 'test' else 'Validation Dev'})\n")
        md_lines.append("| Dataset | Backbone | Raw | +GR-CLIP | +NCM (0 D_cal) | +OAS-LDA (0 D_cal) | +cal (100% D_cal) | **CSS-CT (0 D_cal)** | 3-View (+cal -> CSS) | Wins (vs Raw / NCM / cal) | TSV+CSS | SeqMM+NCM | LMNAR+CSS | MG-CLIP | PAL-RLS | Joint+NCM |")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            w = r["wins"]
            md_lines.append(
                f"| `{r['dataset']}` | `{r['backbone']}` | {r['raw']:.2f} | {r['gr_clip']:.2f} | {r['ncm']:.2f} | {r['oas_lda']:.2f} | {r['cal_bias']:.2f} | **{r['css_ct']:.2f}** | "
                f"{r['v3_cal']:.2f} -> **{r['v3_css']:.2f}** | {w['raw']}/{w['tot']} · {w['ncm']}/{w['tot']} · {w['cal_bias']}/{w['tot']} | "
                f"{r['tsvm_css']:.2f} | {r['seq_mm_ncm']:.2f} | {r['lmnar_css']:.2f} | {r['mg_clip']:.2f} | {r['pal_rls']:.2f} | {r['joint_ncm']:.2f} |"
            )
            ds_tex, bb_tex = r["dataset"].replace("_", "\\_"), r["backbone"].replace("_", "\\_")
            tex_lines.append(
                f"{sp}:\\texttt{{{ds_tex}}} & \\texttt{{{bb_tex}}} & {r['raw']:.2f} & {r['gr_clip']:.2f} & {r['ncm']:.2f} & {r['oas_lda']:.2f} & {r['cal_bias']:.2f} & \\textbf{{{r['css_ct']:.2f}}} & {r['tsvm_css']:.2f} & {r['seq_mm_ncm']:.2f} & {r['mg_clip']:.2f} & {r['pal_rls']:.2f} & {r['joint_ncm']:.2f} \\\\"
            )
        md_lines.append("")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    md_text = "\n".join(md_lines) + "\n"
    if "$" in md_text:
        raise AssertionError("Forbidden dollar character detected in ICCV_ECOSYSTEM_MASTER_SUMMARY.md")
    md_path = out_root / "ICCV_ECOSYSTEM_MASTER_SUMMARY.md"
    md_path.write_text(md_text, encoding="utf8")
    (out_root / "ICCV_ECOSYSTEM_MASTER_TABLE.tex").write_text("\n".join(tex_lines), encoding="utf8")
    write_json_atomic(out_root / "ICCV_ECOSYSTEM_MASTER_RESULTS.json", master_data)
    return md_path
