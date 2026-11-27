"""Aggregation, LaTeX tables, and severity-sweep plots for the 3-piece ICCV supplementary suite."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnarm.core import write_json_atomic
from mnarm.eval.iccv_figures import _configure_style, _save_pair

SUPP_ARMS = ("opcm", "knots", "tsvm", "dc_lora", "iso_c", "magmax", "seq_ft", "seq_magmax", "seq_lmnar", "seq_replay256", "mg_clip", "pal_rls", "joint_oracle")
SWEEP_TRACKS = (
    ("opcm", "zero_mnar", "OPCM + CSS-CT (Ours)", "^", "#0072B2", 2.4),
    ("tsvm", "zero_mnar", "TSV-M + CSS-CT (Ours)", "s", "#009E73", 2.2),
    ("tsvm", "ncm", "TSV-M + NCM (0-label)", "o", "#E69F00", 1.9),
    ("tsvm", "cal_bias", "TSV-M + cal (100% D_cal)", "D", "#56B4E9", 1.7),
    ("tsvm", "raw", "TSV-M (Raw)", "x", "#777777", 1.6),
    ("seq_magmax", "ncm", "Seq-MagMax + NCM", "v", "#CC79A7", 1.7),
    ("seq_replay256", "raw", "Seq-Replay256 (Raw)", "P", "#333333", 1.6),
    ("mg_clip", "raw", "MG-CLIP (ICCV'25)", "d", "#D55E00", 1.8),
    ("pal_rls", "raw", "PAL-RLS (2025)", "p", "#8C564B", 1.8),
)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / float(len(vals))
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / float(len(vals) - 1)
    return m, math.sqrt(max(0.0, var))


def _fmt_ms(m: float, s: float, tex: bool = False) -> str:
    if m != m:
        return "--"
    sep = " \\ensuremath{\\pm} " if tex else " ± "
    return f"{m:.2f}{sep}{s:.2f}"


def _collect_proto_jsons(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not root.is_dir():
        return {}
    selected: dict[Path, Path] = {}
    for fn in ("ICCV_PROTOCOL_RESULTS.json", "ICCV_PROTOCOL_TEST_RESULTS.json"):
        for p in sorted(root.rglob(fn)):
            selected[p.parent] = p
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in selected.values():
        d = json.loads(p.read_text(encoding="utf8"))
        c = d["config"]
        out[(str(c.get("dataset", "crisismmd")), str(c["backbone"]), str(c["protocol"]))] = d
    return out


def _cell_val(rec: dict[str, Any], arm: str, reg: str, metric: str = "macro_f1") -> float | None:
    rg = rec.get("arms", {}).get(arm, {}).get("regimes", {}).get(reg)
    if not rg:
        return None
    val = rg["summary"]["view3_mean_f1"] if metric == "view3" else rg["paired"].get(metric)
    return float(val) * 100.0 if val is not None else None


def _summarize_piece1(ms_root: Path, out_dir: Path) -> dict[str, Any]:
    seed_dirs = sorted(p for p in ms_root.glob("*/seed_*") if p.is_dir()) if ms_root.is_dir() else []
    by_seed: dict[int, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for sd in seed_dirs:
        m = re.match(r"seed_(\d+)$", sd.name)
        if m:
            by_seed[int(m.group(1))].update(_collect_proto_jsons(sd))
    if not by_seed:
        return {}
    seeds = sorted(by_seed)
    lines = [
        f"# Piece 1: Multi-Seed & Class-Order Permutation ({len(seeds)} seeds: {', '.join(map(str, seeds))})\n",
        "| Arm | All P1-P4 Raw | All P1-P4 +NCM | All P1-P4 +cal | **All P1-P4 CSS-CT** | P3/P4 Raw | P3/P4 +NCM | **P3/P4 CSS-CT** |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex_rows: list[str] = []
    summary_json: dict[str, Any] = {"seeds": seeds, "arms": {}}
    for arm in SUPP_ARMS:
        def _agg(protos: tuple[str, ...], reg: str) -> tuple[float, float]:
            per_s = []
            for s in seeds:
                vs = [_cell_val(r, arm, reg) for (_, _, pr), r in by_seed[s].items() if pr in protos]
                vs_c = [v for v in vs if v is not None]
                if vs_c:
                    per_s.append(sum(vs_c) / float(len(vs_c)))
            return _mean_std(per_s)
        stats = {
            k: _agg(prs, rg)
            for k, prs, rg in (
                ("all_raw", ("P1", "P2", "P3", "P4"), "raw"),
                ("all_ncm", ("P1", "P2", "P3", "P4"), "ncm"),
                ("all_cal", ("P1", "P2", "P3", "P4"), "cal_bias"),
                ("all_css", ("P1", "P2", "P3", "P4"), "zero_mnar"),
                ("p34_raw", ("P3", "P4"), "raw"),
                ("p34_ncm", ("P3", "P4"), "ncm"),
                ("p34_css", ("P3", "P4"), "zero_mnar"),
            )
        }
        summary_json["arms"][arm] = {k: {"mean": round(v[0], 3), "std": round(v[1], 3)} for k, v in stats.items() if v[0] == v[0]}
        lines.append(
            f"| `{arm}` | {_fmt_ms(*stats['all_raw'])} | {_fmt_ms(*stats['all_ncm'])} | {_fmt_ms(*stats['all_cal'])} | "
            f"**{_fmt_ms(*stats['all_css'])}** | {_fmt_ms(*stats['p34_raw'])} | {_fmt_ms(*stats['p34_ncm'])} | **{_fmt_ms(*stats['p34_css'])}** |"
        )
        arm_tex = arm.replace("_", "\\_")
        tex_rows.append(
            f"\\texttt{{{arm_tex}}} & {_fmt_ms(*stats['all_raw'], True)} & {_fmt_ms(*stats['all_ncm'], True)} & "
            f"\\textbf{{{_fmt_ms(*stats['all_css'], True)}}} & {_fmt_ms(*stats['p34_ncm'], True)} & \\textbf{{{_fmt_ms(*stats['p34_css'], True)}}} \\\\"
        )
    md = "\n".join(lines) + "\n"
    assert "$" not in md
    (out_dir / "ICCV_SUPP_PIECE1_MULTISEED.md").write_text(md, encoding="utf8")
    tex = "\n".join(["\\begin{tabular}{lrrrrr}", "\\toprule", "Method & Raw & +NCM & CSS-CT & P3/P4 +NCM & P3/P4 CSS-CT \\\\", "\\midrule", *tex_rows, "\\bottomrule", "\\end{tabular}"]) + "\n"
    (out_dir / "ICCV_SUPP_PIECE1_MULTISEED.tex").write_text(tex, encoding="utf8")
    return summary_json


def _summarize_piece2(supp_root: Path, out_dir: Path) -> dict[str, Any]:
    l14_map = _collect_proto_jsons(supp_root / "scaling_l14")
    b16_map = {k: v for k, v in _collect_proto_jsons(supp_root / "multiseed").items() if k[1] == "clip_b16"}
    # Keep only seed_11 for B/16 scaling match when available
    s11_paths = list((supp_root / "multiseed").glob("*/seed_11")) if (supp_root / "multiseed").is_dir() else []
    if s11_paths:
        b16_map = {}
        for sp in s11_paths:
            b16_map.update({k: v for k, v in _collect_proto_jsons(sp).items() if k[1] == "clip_b16"})
    ood_map = _collect_proto_jsons(supp_root / "ood_general")
    if not l14_map and not ood_map:
        return {}

    lines = ["# Piece 2: Backbone Scaling (`CLIP ViT-L/14`) & Out-of-Domain (`ScienceQA-MM8`)\n"]
    res_json: dict[str, Any] = {"scaling_l14": {}, "ood_scienceqa": {}}
    if l14_map:
        lines.append("## 2A. CLIP-B/16 vs CLIP-L/14 (Paired Macro-F1 % across matched datasets)\n")
        lines.append("| Arm | B/16 Raw / +NCM / **CSS-CT** | L/14 Raw / +NCM / **CSS-CT** | L/14 P3/P4 (+NCM -> **CSS-CT**) |")
        lines.append("|---|---|---|---|")
        matched_ds = sorted({k[0] for k in l14_map})
        for arm in SUPP_ARMS:
            def _m(src: dict[tuple[str, str, str], dict[str, Any]], prs: tuple[str, ...], rg: str) -> float:
                vs = [_cell_val(r, arm, rg) for (ds, _, pr), r in src.items() if ds in matched_ds and pr in prs]
                vc = [v for v in vs if v is not None]
                return sum(vc) / float(len(vc)) if vc else float("nan")
            b_r, b_n, b_c = _m(b16_map, ("P1", "P2", "P3", "P4"), "raw"), _m(b16_map, ("P1", "P2", "P3", "P4"), "ncm"), _m(b16_map, ("P1", "P2", "P3", "P4"), "zero_mnar")
            l_r, l_n, l_c = _m(l14_map, ("P1", "P2", "P3", "P4"), "raw"), _m(l14_map, ("P1", "P2", "P3", "P4"), "ncm"), _m(l14_map, ("P1", "P2", "P3", "P4"), "zero_mnar")
            l_p34_n, l_p34_c = _m(l14_map, ("P3", "P4"), "ncm"), _m(l14_map, ("P3", "P4"), "zero_mnar")
            res_json["scaling_l14"][arm] = {
                k: (round(v, 4) if v == v else None)
                for k, v in (("l14_raw", l_r), ("l14_ncm", l_n), ("l14_css", l_c), ("l14_p34_css", l_p34_c))
            }
            fmt = lambda x: f"{x:.2f}" if x == x else "--"
            lines.append(f"| `{arm}` | {fmt(b_r)} / {fmt(b_n)} / **{fmt(b_c)}** | {fmt(l_r)} / {fmt(l_n)} / **{fmt(l_c)}** | {fmt(l_p34_n)} -> **{fmt(l_p34_c)}** |")
    if ood_map:
        lines.append("\n## 2B. Out-of-Domain Generalization on ScienceQA-MM8 (CLIP-B/16 + SigLIP-B/16)\n")
        lines.append("| Arm | All P1-P4 Raw | +GR-CLIP | +NCM | +cal | **CSS-CT** | P3/P4 (+NCM -> **CSS-CT**) | 3-View CSS | P3/P4 ErRec (Raw -> CSS) |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---|")
        for arm in SUPP_ARMS:
            def _mo(prs: tuple[str, ...], rg: str, mt: str = "macro_f1") -> float:
                vs = [_cell_val(r, arm, rg, mt) for (_, _, pr), r in ood_map.items() if pr in prs]
                vc = [v for v in vs if v is not None]
                return sum(vc) / float(len(vc)) if vc else float("nan")
            r_all, gr_all, n_all, c_all, z_all = _mo(("P1", "P2", "P3", "P4"), "raw"), _mo(("P1", "P2", "P3", "P4"), "gr_clip"), _mo(("P1", "P2", "P3", "P4"), "ncm"), _mo(("P1", "P2", "P3", "P4"), "cal_bias"), _mo(("P1", "P2", "P3", "P4"), "zero_mnar")
            n34, z34 = _mo(("P3", "P4"), "ncm"), _mo(("P3", "P4"), "zero_mnar")
            v3 = _mo(("P1", "P2", "P3", "P4"), "zero_mnar" if arm != "pal_rls" else "raw", "view3")
            er0, er1 = _mo(("P3", "P4"), "raw", "early_recall"), _mo(("P3", "P4"), "zero_mnar", "early_recall")
            res_json["ood_scienceqa"][arm] = {
                k: (round(v, 4) if v == v else None)
                for k, v in (("raw", r_all), ("ncm", n_all), ("css", z_all), ("p34_css", z34))
            }
            fmt = lambda x: f"{x:.2f}" if x == x else "--"
            lines.append(f"| `{arm}` | {fmt(r_all)} | {fmt(gr_all)} | {fmt(n_all)} | {fmt(c_all)} | **{fmt(z_all)}** | {fmt(n34)} -> **{fmt(z34)}** | {fmt(v3)} | {fmt(er0)} -> **{fmt(er1)}** |")
    md = "\n".join(lines) + "\n"
    assert "$" not in md
    (out_dir / "ICCV_SUPP_PIECE2_SCALING_OOD.md").write_text(md, encoding="utf8")
    tex_p2 = []
    if res_json["scaling_l14"]:
        tex_p2.extend(["% Piece 2A: CLIP ViT-L/14 Scaling", "\\begin{tabular}{lrrrr}", "\\toprule", "Method & L/14 Raw & L/14 +NCM & L/14 CSS-CT & L/14 P3/P4 CSS-CT \\\\", "\\midrule"])
        for arm, rv in res_json["scaling_l14"].items():
            f = lambda x: f"{x:.2f}" if (x is not None and x == x) else "--"
            arm_tex = arm.replace("_", "\\_")
            tex_p2.append(f"\\texttt{{{arm_tex}}} & {f(rv['l14_raw'])} & {f(rv['l14_ncm'])} & \\textbf{{{f(rv['l14_css'])}}} & \\textbf{{{f(rv['l14_p34_css'])}}} \\\\")
        tex_p2.extend(["\\bottomrule", "\\end{tabular}", ""])
    if res_json["ood_scienceqa"]:
        tex_p2.extend(["% Piece 2B: ScienceQA-MM8 Out-of-Domain", "\\begin{tabular}{lrrrr}", "\\toprule", "Method & Raw & +NCM & CSS-CT & P3/P4 CSS-CT \\\\", "\\midrule"])
        for arm, rv in res_json["ood_scienceqa"].items():
            f = lambda x: f"{x:.2f}" if (x is not None and x == x) else "--"
            arm_tex = arm.replace("_", "\\_")
            tex_p2.append(f"\\texttt{{{arm_tex}}} & {f(rv['raw'])} & {f(rv['ncm'])} & \\textbf{{{f(rv['css'])}}} & \\textbf{{{f(rv['p34_css'])}}} \\\\")
        tex_p2.extend(["\\bottomrule", "\\end{tabular}", ""])
    if tex_p2:
        (out_dir / "ICCV_SUPP_PIECE2_SCALING_OOD.tex").write_text("\n".join(tex_p2), encoding="utf8")
    return res_json


def _summarize_piece3(sweep_root: Path, out_dir: Path) -> dict[str, Any]:
    if not sweep_root.is_dir():
        return {}
    by_rate: dict[int, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for rd in sorted(sweep_root.glob("*/rho_*")):
        m = re.match(r"rho_(\d+)$", rd.name)
        if m:
            by_rate[int(m.group(1))].update(_collect_proto_jsons(rd))
    if not by_rate:
        return {}
    rates = sorted(by_rate)
    lines = [
        "# Piece 3: MNAR Missingness Severity Sweep (P3/P4 Mean over Backbones)\n",
        "| Method / Regime | " + " | ".join(f"{r}%" for r in rates) + " |",
        "|---|" + "|".join("---:" for _ in rates) + "|",
    ]
    series: dict[str, dict[str, list[float]]] = {}
    for arm, reg, lbl, mk, col, lw in SWEEP_TRACKS:
        f1s, ers = [], []
        for r in rates:
            recs = [rec for (_, _, pr), rec in by_rate[r].items() if pr in ("P3", "P4")]
            vf = [v for v in (_cell_val(rc, arm, reg, "macro_f1") for rc in recs) if v is not None]
            ve = [v for v in (_cell_val(rc, arm, reg, "early_recall") for rc in recs) if v is not None]
            f1s.append(sum(vf) / float(len(vf)) if vf else float("nan"))
            ers.append(sum(ve) / float(len(ve)) if ve else float("nan"))
        series[lbl] = {"f1": f1s, "early_recall": ers}
        cells = [f"{v:.2f}" if v == v else "--" for v in f1s]
        lines.append(f"| {lbl} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    assert "$" not in md
    (out_dir / "ICCV_SUPP_PIECE3_MNAR_SWEEP.md").write_text(md, encoding="utf8")
    col_spec = "l" + "r" * len(rates)
    pct_sym = "\\%"
    hdr_tex = " & ".join(f"{r}{pct_sym}" for r in rates)
    rows_tex = []
    for _, _, lbl, _, _, _ in SWEEP_TRACKS:
        vals = " & ".join(f"{v:.2f}" if v == v else "--" for v in series[lbl]["f1"])
        lbl_tex = lbl.replace("%", "\\%").replace("_", "\\_")
        rows_tex.append(f"{lbl_tex} & {vals} \\\\")
    tex3 = "\n".join([f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule", f"Method & {hdr_tex} \\\\", "\\midrule", *rows_tex, "\\bottomrule", "\\end{tabular}"]) + "\n"
    (out_dir / "ICCV_SUPP_PIECE3_MNAR_SWEEP.tex").write_text(tex3, encoding="utf8")

    _configure_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    for arm, reg, lbl, mk, col, lw in SWEEP_TRACKS:
        ys_f1, ys_er = series[lbl]["f1"], series[lbl]["early_recall"]
        ax1.plot(rates, ys_f1, marker=mk, color=col, linewidth=lw, markersize=6, label=lbl)
        ax2.plot(rates, ys_er, marker=mk, color=col, linewidth=lw, markersize=6, label=lbl)
    ax1.set_title("Paired Macro-F1 (%) vs. MNAR Missingness Rate")
    ax2.set_title("Early-Class Recall (%) vs. MNAR Missingness Rate")
    for ax, ylab in ((ax1, "Paired Macro-F1 (%)"), (ax2, "Early-Class Recall (%)")):
        ax.set_xlabel("Early-Class Missing-Modality Rate (%)")
        ax.set_ylabel(ylab)
        ax.set_xticks(rates)
    ax1.legend(loc="lower left", frameon=True, fontsize=7.8)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _save_pair(fig, fig_dir / "fig_mnar_severity_sweep")
    json_series = {lbl: {k: [round(v, 4) if v == v else None for v in vs] for k, vs in d.items()} for lbl, d in series.items()}
    return {"rates_pct": rates, "series": json_series}


def write_supplementary_suite(supp_root: Path) -> Path:
    """Aggregate all 3 supplementary pieces into Markdown, LaTeX, JSON, and sweep figures."""
    supp_root.mkdir(parents=True, exist_ok=True)
    p1 = _summarize_piece1(supp_root / "multiseed", supp_root)
    p2 = _summarize_piece2(supp_root, supp_root)
    p3 = _summarize_piece3(supp_root / "mnar_sweep", supp_root)
    payload = {"schema": "mnarm-iccv-supplementary-v1", "piece1_multiseed": p1, "piece2_scaling_ood": p2, "piece3_mnar_sweep": p3}
    write_json_atomic(supp_root / "ICCV_SUPPLEMENTARY_MASTER_RESULTS.json", payload)
    sections = ["# ICCV 3-Piece Supplementary Suite Summary\n"]
    for fn in ("ICCV_SUPP_PIECE1_MULTISEED.md", "ICCV_SUPP_PIECE2_SCALING_OOD.md", "ICCV_SUPP_PIECE3_MNAR_SWEEP.md"):
        fp = supp_root / fn
        if fp.is_file():
            sections.append(fp.read_text(encoding="utf8"))
    master_md = "\n".join(sections)
    assert "$" not in master_md
    out_md = supp_root / "ICCV_SUPPLEMENTARY_MASTER_SUMMARY.md"
    out_md.write_text(master_md, encoding="utf8")
    return out_md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    out_md = write_supplementary_suite(args.root)
    print(f"[SUPPLEMENTARY SUMMARY WRITTEN] {out_md}", flush=True)


if __name__ == "__main__":
    main()
