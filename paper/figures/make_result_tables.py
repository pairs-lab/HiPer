#!/usr/bin/env python3
"""
Write every results table of the paper, and the numbers quoted in the text,
from the saved summaries (results/<run>/<split>/summary.json). No result
number in the manuscript is typed by hand.

Outputs (paper/figures/):
  tab_persistence.tex  E2: THOR test + ObChange, all persistence models
  tab_ablation.tex     E4: ablations of the full model (THOR test)
  tab_assoc.tex        E3: association success (THOR test + ObChange)
  numbers.tex          \\newcommand macros for numbers quoted in the text

  python paper/figures/make_result_tables.py --run main
"""

import argparse
import json
import math
import os
from pathlib import Path

# same layout as hiper.evaluation.paths (this script needs no package import)
RESULTS = Path(os.environ.get("HIPER_DATA", "~/lifelong_ws/data")).expanduser() \
    / "hiper_thor" / "results"
OUT = Path(__file__).resolve().parent

MODELS = [("ours", "HiPer (ours)"),
          ("pf_learned", r"PF, learned rates"),
          ("pf", r"PF($\lambda$)~\cite{rosen2016persistence}"),
          ("pf_class", r"PF, hand-set prior"),
          ("ours_hand_prior", r"HiPer, hand-set prior"),
          ("pocd", r"POCD-style~\cite{bogenberger2026glasses}"),
          ("elite", r"ELite scalar~\cite{gil2025elite}"),
          ("draft_v1", "Fuse-then-decay"),
          ("always", "Never forget")]
ABL = [("ours", "HiPer (full)"),
       ("abl_reset_rate_on_relocation", "rate reset at relocation"),
       ("abl_no_class_prior", "pooled rate, no class rates"),
       ("abl_hand_prior", "hand-set prior (CLIP labels)"),
       ("diag_hand_prior_gt_labels", "hand-set prior (true labels)"),
       ("abl_no_pose_inflation", "w/o pose inflation"),
       ("abl_clip_appearance", "CLIP instead of DINOv2")]
ASSOC = [("e3_none", "Geometry only"), ("e3_clip", "CLIP + geom."),
         ("e3_clip+dino", "CLIP + DINOv2 + geom."),
         ("ours", "DINOv2 + geom. (HiPer)"),
         ("abl_no_pose_inflation", "\\quad w/o pose inflation")]
LOWER = {"nll", "brier", "ece", "premature_prune_rate", "stale_rate",
         "novel_false_match", "mae"}


def load(run, split):
    p = RESULTS / run / split / "summary.json"
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:.{nd}f}"


def col(summ, models, metric):
    return {m: (summ[m][metric][0] if m in summ and metric in summ[m]
                else float("nan")) for m in models}


def best_of(vals, metric):
    v = {m: x for m, x in vals.items() if x == x}
    if not v:
        return None
    return (min if metric in LOWER else max)(v, key=v.get)


def sig(paired, m, metric):
    """Dagger if the paired bootstrap CI of (m - ours) excludes 0."""
    if paired is None or m not in paired or metric not in paired[m]:
        return ""
    d = paired[m][metric]
    return r"$^\dagger$" if d["ci_lo"] > 0 or d["ci_hi"] < 0 else ""


def table_persistence(test, obc=None):
    """THOR test: all methods, then the ablations of the full model."""
    mets = [("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE"),
            ("auroc", "AUROC"), ("premature_prune_rate", "Prem."),
            ("stale_rate", "Stale")]
    p = test["persistence"]
    paired = test.get("paired_vs_ours")
    ids = [m for m, _ in MODELS]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(mets) + "@{}}", r"\toprule",
             "Method & " + " & ".join(
                 lab + (r"$\downarrow$" if m in LOWER else r"$\uparrow$")
                 for m, lab in mets) + r"\\", r"\midrule"]
    best = {m: best_of(col(p, ids, m), m) for m, _ in mets}
    for mid, name in MODELS:
        cells = []
        for m, _ in mets:
            v = p[mid][m][0]
            c = fmt(v)
            if mid == best[m]:
                c = r"\textbf{" + c + "}"
            if mid != "ours":
                c += sig(paired, mid, m)
            cells.append(c)
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    for mid, name in ABL[1:5]:
        cells = [fmt(p[mid][m][0]) + sig(paired, mid, m) for m, _ in mets]
        lines.append(f"\\quad {name} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_obchange(obc, loro):
    """Real robot: sim-tuned (bootstrap over 5 rooms) vs leave-one-room-out
    (no significance marks: an exact sign-flip test over 5 rooms cannot
    reach p < 0.05)."""
    ids = [m for m, _ in MODELS]
    z, l_ = obc["persistence"], loro["persistence"]
    zp = obc.get("paired_vs_ours")
    mets = [("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE"),
            ("auroc", "AUC")]
    lines = [r"\begin{tabular}{@{}lcccc|cccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c|}{Tuned in simulation} & "
             r"\multicolumn{4}{c}{Leave-one-room-out}\\",
             "Method & " + " & ".join(
                 lab + (r"$\downarrow$" if m in LOWER else r"$\uparrow$")
                 for m, lab in mets + mets) + r"\\", r"\midrule"]
    bz = {m: best_of(col(z, ids, m), m) for m, _ in mets}
    bl = {m: best_of(col(l_, ids, m), m) for m, _ in mets}
    for mid, name in MODELS:
        cells = []
        for m, _ in mets:
            c = fmt(z[mid][m][0])
            c = r"\textbf{" + c + "}" if mid == bz[m] else c
            if mid != "ours":
                c += sig(zp, mid, m)
            cells.append(c)
        for m, _ in mets:
            c = fmt(l_[mid][m][0])
            cells.append(r"\textbf{" + c + "}" if mid == bl[m] else c)
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_ablation(test):
    p = test["persistence"]
    a = test["assoc_ablations_persistence"]
    asc = test["association"]
    rows = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r"Variant & NLL$\downarrow$ & ECE$\downarrow$ & Prem.$\downarrow$ "
            r"& Stale$\downarrow$ & Assoc.$\uparrow$\\", r"\midrule"]
    for mid, name in ABL:
        src = a if mid in a and mid != "ours" else p
        cfg = mid if mid in asc else "ours"
        cells = [fmt(src[mid][k][0]) for k in
                 ("nll", "ece", "premature_prune_rate", "stale_rate")]
        cells.append(fmt(asc[cfg]["success"][0]))
        rows.append(f"{name} & " + " & ".join(cells) + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def table_assoc(test, obc):
    mets = [("success", "All"), ("success_stationary", "Stat."),
            ("success_relocated", "Reloc."), ("success_reappeared", "Reapp."),
            ("novel_false_match", r"Novel FM$\downarrow$")]
    ap = test["assoc_ablations_persistence"]
    rows = [r"\begin{tabular}{@{}l" + "c" * len(mets) + "c|c@{}}",
            r"\toprule",
            r" & \multicolumn{%d}{c|}{THOR-Lifelong test} & ObCh.\\"
            % (len(mets) + 1),
            "Appearance & " + " & ".join(l for _, l in mets)
            + r" & NLL$\downarrow$ & All\\", r"\midrule"]
    for cid, name in ASSOC:
        if cid not in test["association"]:
            continue
        t = test["association"][cid]
        cells = [fmt(t[m][0]) for m, _ in mets]
        cells.append(fmt(ap[cid]["nll"][0]) if cid in ap else "--")
        o = obc["association"].get(cid) if obc else None
        cells.append(fmt(o["success"][0]) if o else "--")
        rows.append(f"{name} & " + " & ".join(cells) + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def numbers(test, obc, loro=None):
    out = []

    def put(name, x, nd=3):
        out.append(f"\\newcommand{{\\{name}}}{{{fmt(x, nd)}}}")
    for tag, s in (("Thor", test), ("Obc", obc), ("Loro", loro)):
        if s is None:
            continue
        p = s["persistence"]
        for mid, short in (("ours", "Ours"), ("pf_learned", "Pfl"),
                           ("pf", "Pf"), ("pf_class", "Pfc"),
                           ("ours_hand_prior", "Hand"), ("pocd", "Pocd"),
                           ("elite", "Elite"), ("draft_v1", "Draft"),
                           ("abl_no_rate_learning", "NoLearn"),
                           ("abl_no_class_prior", "NoPrior"),
                           ("abl_reset_rate_on_relocation", "Reset"),
                           ("diag_hand_prior_gt_labels", "HandGt"),
                           ("diag_pf_hand_gt_labels", "PfcGt"),
                           ("abl_clip_appearance", "ClipApp"),
                           ("abl_hand_prior", "HandAbl")):
            if mid not in p:
                continue
            for met, mshort in (("nll", "Nll"), ("ece", "Ece"),
                                ("brier", "Brier"), ("auroc", "Auroc"),
                                ("ap_absent", "Apa"),
                                ("premature_prune_rate", "Prem"),
                                ("stale_rate", "Stale")):
                put(f"r{tag}{short}{mshort}", p[mid][met][0])
        put(f"r{tag}Scenes", p["ours"]["nll"][4], 0)
        put(f"r{tag}Labeled", p["ours"]["n_labeled"][0] *
            p["ours"]["nll"][4], 0)
        t = s.get("timing", {})
        put(f"r{tag}MapMs", 1000.0 * t.get("mapping_s_per_session",
                                           float("nan")), 0)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--test-split", default="test",
                    help="dev only: e.g. 'val' on a smoke run")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    test, obc = load(args.run, args.test_split), load(args.run, "obchange")
    if test is None:
        raise SystemExit("no test summary yet")
    (out / "tab_persistence.tex").write_text(table_persistence(test))
    loro = load(args.run, "obchange_loro")
    if obc is not None and loro is not None:
        (out / "tab_obchange.tex").write_text(table_obchange(obc, loro))
    (out / "tab_ablation.tex").write_text(table_ablation(test))
    (out / "tab_assoc.tex").write_text(table_assoc(test, obc))
    (out / "numbers.tex").write_text(numbers(test, obc, loro))
    print("wrote tables from run", args.run)


if __name__ == "__main__":
    main()
