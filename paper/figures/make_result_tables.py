#!/usr/bin/env python3
"""
Write every results table of the paper, and the numbers quoted in the text,
from the saved summaries (results/<run>/<split>/summary.json). No result
number in the manuscript is typed by hand.

Outputs (paper/figures/):
  tab_persistence.tex  E2: THOR test + ObChange, all persistence models
  tab_ablation.tex     E4: ablations of the full model (THOR test)
  tab_assoc.tex        E3: association success (THOR test + ObChange)
  tab_rscan.tex        3RScan test (real, evaluated once), raw + calibrated
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

# Ours and the external baselines; ranked (bold best, underlined second
# best) in every table
MAIN = [("ours", "HiPer (ours)"),
        ("pf", r"PF($\lambda$)~\cite{rosen2016persistence}"),
        ("pf_reloc", r"PF($\lambda$), relocation-aware"),
        ("pocd", r"POCD-style~\cite{bogenberger2026glasses}"),
        ("elite", r"ELite scalar~\cite{gil2025elite}"),
        ("draft_v1", "Fuse-then-decay"),
        ("always", "Never forget")]
# re-tuned ablations of HiPer (rows of tab_ablation.tex, names only here)
RETUNED = [("pf_nbr", "PF, retrieved rates"),
           ("ours_class", "HiPer, CLIP class rates"),
           ("pf_learned", "PF, CLIP class rates")]
MODELS = MAIN + RETUNED
ABL = [("ours", "HiPer (full)"),
       ("abl_reset_rate_on_relocation", "rate reset at relocation"),
       ("abl_no_class_prior", "pooled rate, no retrieval"),
       ("abl_dino_only_retrieval", "retrieval by DINOv2 only"),
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


def ranks(vals, metric):
    """{row: 'best' | 'second'} by printed value (ties at the printed
    precision share a rank); rows without a value get no rank."""
    shown = {m: fmt(x) for m, x in vals.items() if x == x}
    levels = sorted(set(shown.values()), key=float,
                    reverse=metric not in LOWER)
    tag = dict(zip(levels[:2], ("best", "second")))
    return {m: tag.get(v) for m, v in shown.items()}


def deco(cell, rank):
    """Bold the best, underline the second best."""
    if rank == "best":
        return r"\textbf{" + cell + "}"
    if rank == "second":
        return r"\underline{" + cell + "}"
    return cell


def sig(paired, m, metric):
    """Dagger if the paired bootstrap CI of (m - ours) excludes 0."""
    if paired is None or m not in paired or metric not in paired[m]:
        return ""
    d = paired[m][metric]
    return r"$^\dagger$" if d["ci_lo"] > 0 or d["ci_hi"] < 0 else ""


def table_persistence(test, obc=None):
    """Held-out THOR: ours and the baselines (ranked), then the re-tuned
    ablations and the ablations at HiPer's parameters; raw and calibrated
    (maps fitted on val)."""
    mets = [("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE"),
            ("auroc", "AUROC")]
    cmets = [("nll", "NLL"), ("ece", "ECE")]
    dmets = [("premature_prune_rate", "Prem."), ("stale_rate", "Stale")]
    p = test["persistence"]
    paired = test.get("paired_vs_ours")
    main = [m for m, _ in MAIN if m in p]
    hdr = (lambda m, lab: lab + (r"$\downarrow$" if m in LOWER  # noqa: E731
                                 else r"$\uparrow$"))
    ncol = 1 + len(mets) + len(cmets) + len(dmets)
    lines = [r"\begin{tabular}{@{}l" + "c" * len(mets) + "|" +
             "c" * len(cmets) + "|" + "c" * len(dmets) + "@{}}", r"\toprule",
             r" & \multicolumn{%d}{c|}{Prediction} & "
             r"\multicolumn{%d}{c|}{Calibrated} & "
             r"\multicolumn{%d}{c}{Map maintenance}\\"
             % (len(mets), len(cmets), len(dmets)),
             "Method & " + " & ".join(hdr(m, l) for m, l in
                                      mets + cmets + dmets) + r"\\",
             r"\midrule"]
    rk = {m: ranks(col(p, main, m), m) for m, _ in mets + dmets}
    crk = {m: ranks({x: (p[x + "_cal"][m][0] if x + "_cal" in p
                         else float("nan")) for x in main}, m)
           for m, _ in cmets}

    def row(mid, name, ranked, indent=""):
        cells = []
        for m, _ in mets:
            c = fmt(p[mid][m][0])
            c = deco(c, rk[m].get(mid)) if ranked else c
            cells.append(c + ("" if mid == "ours" else sig(paired, mid, m)))
        cm = mid + "_cal"
        for m, _ in cmets:
            if cm not in p:
                cells.append("--")
                continue
            c = fmt(p[cm][m][0])
            c = deco(c, crk[m].get(mid)) if ranked else c
            cells.append(c + ("" if mid == "ours" else sig(paired, cm, m)))
        for m, _ in dmets:
            c = fmt(p[mid][m][0])
            c = deco(c, rk[m].get(mid)) if ranked else c
            cells.append(c + ("" if mid == "ours" else sig(paired, mid, m)))
        return f"{indent}{name} & " + " & ".join(cells) + r"\\"

    for mid, name in MAIN:
        if mid in p:
            lines.append(row(mid, name, True))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_earlier(ta, tb):
    """Splits looked at during development: original test (20 scenes) and
    test2 (20 scenes); NLL and AUROC, daggers from each split's paired
    bootstrap; ours and the baselines ranked, then the re-tuned ablations."""
    names = dict(MODELS)
    lines = [r"\begin{tabular}{@{}lcc|cc@{}}", r"\toprule",
             r" & \multicolumn{2}{c|}{Test (20)} & "
             r"\multicolumn{2}{c}{Test-B (20)}\\",
             r"Method & NLL$\downarrow$ & AUROC$\uparrow$ & NLL$\downarrow$ "
             r"& AUROC$\uparrow$\\", r"\midrule"]
    main = [m for m in FRESH if m in ta["persistence"]
            and m in tb["persistence"]]
    rk = {(key, met): ranks(col(t["persistence"], main, met), met)
          for key, t in (("a", ta), ("b", tb)) for met in ("nll", "auroc")}

    def row(mid, ranked, indent=""):
        cells = []
        for key, t in (("a", ta), ("b", tb)):
            for met in ("nll", "auroc"):
                c = fmt(t["persistence"][mid][met][0])
                c = deco(c, rk[(key, met)].get(mid)) if ranked else c
                if mid != "ours":
                    c += sig(t.get("paired_vs_ours"), mid, met)
                cells.append(c)
        return f"{indent}{names[mid]} & " + " & ".join(cells) + r"\\"

    lines += [row(m, True) for m in main]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


FRESH = ["ours", "pf", "pf_reloc", "elite", "pocd"]   # development splits


def table_obchange(obc, loro):
    """Real robot: parameters from simulation vs leave-one-room-out
    (cross-fitted), raw and calibrated; ours and the baselines ranked, then
    the re-tuned ablations. No significance marks: with five rooms an exact
    sign-flip test cannot reach p < 0.05."""
    z, l_ = obc["persistence"], loro["persistence"]
    mets = [("nll", "NLL"), ("ece", "ECE"), ("auroc", "AUC")]
    cmets = [("nll", "NLL"), ("ece", "ECE")]
    hdr = (lambda m, lab: lab + (r"$\downarrow$" if m in LOWER  # noqa: E731
                                 else r"$\uparrow$"))
    lines = [r"\begin{tabular}{@{}lccc|ccc|cc@{}}", r"\toprule",
             r" & \multicolumn{3}{c|}{Tuned in simulation} & "
             r"\multicolumn{3}{c|}{Leave-one-room-out} & "
             r"\multicolumn{2}{c}{LORO, calibrated}\\",
             "Method & " + " & ".join(hdr(m, l) for m, l in
                                      mets + mets + cmets) + r"\\",
             r"\midrule"]
    main = [m for m, _ in MAIN if m in z and m in l_]
    rz = {m: ranks(col(z, main, m), m) for m, _ in mets}
    rl = {m: ranks(col(l_, main, m), m) for m, _ in mets}
    rc = {m: ranks({x: (l_[x + "_cal"][m][0] if x + "_cal" in l_
                        else float("nan")) for x in main}, m)
          for m, _ in cmets}

    def row(mid, name, ranked, indent=""):
        d = (lambda c, r: deco(c, r) if ranked else c)  # noqa: E731
        cells = [d(fmt(z[mid][m][0]), rz[m].get(mid)) for m, _ in mets]
        cells += [d(fmt(l_[mid][m][0]), rl[m].get(mid)) for m, _ in mets]
        cm = mid + "_cal"
        cells += [d(fmt(l_[cm][m][0]), rc[m].get(mid)) if cm in l_ else "--"
                  for m, _ in cmets]
        return f"{indent}{name} & " + " & ".join(cells) + r"\\"

    lines += [row(m, n, True) for m, n in MAIN if m in main]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_rscan(rs):
    """3RScan test environments (real data, evaluated once): ours and the
    baselines (ranked), then the re-tuned ablations and the ablations at
    HiPer's parameters; raw and calibrated. Daggers: paired bootstrap over
    environments (calibrated: to calibrated HiPer)."""
    mets = [("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE"),
            ("auroc", "AUC")]
    cmets = [("nll", "NLL"), ("ece", "ECE")]
    p = rs["persistence"]
    paired = rs.get("paired_vs_ours")
    main = [m for m, _ in MAIN if m in p]
    hdr = (lambda m, lab: lab + (r"$\downarrow$" if m in LOWER  # noqa: E731
                                 else r"$\uparrow$"))
    ncol = 1 + len(mets) + len(cmets)
    lines = [r"\begin{tabular}{@{}l" + "c" * len(mets) + "|" +
             "c" * len(cmets) + "@{}}", r"\toprule",
             r" & \multicolumn{%d}{c|}{Prediction} & "
             r"\multicolumn{%d}{c}{Calibrated}\\" % (len(mets), len(cmets)),
             "Method & " + " & ".join(hdr(m, l) for m, l in mets + cmets)
             + r"\\", r"\midrule"]
    rk = {m: ranks(col(p, main, m), m) for m, _ in mets}
    crk = {m: ranks({x: (p[x + "_cal"][m][0] if x + "_cal" in p
                         else float("nan")) for x in main}, m)
           for m, _ in cmets}

    def row(mid, name, ranked, indent=""):
        cells = [(deco(fmt(p[mid][m][0]), rk[m].get(mid)) if ranked
                  else fmt(p[mid][m][0]))
                 + ("" if mid == "ours" else sig(paired, mid, m))
                 for m, _ in mets]
        cm = mid + "_cal"
        cells += [((deco(fmt(p[cm][m][0]), crk[m].get(mid)) if ranked
                    else fmt(p[cm][m][0]))
                   + ("" if mid == "ours" else sig(paired, cm, m)))
                  if cm in p else "--" for m, _ in cmets]
        return f"{indent}{name} & " + " & ".join(cells) + r"\\"

    lines += [row(m, n, True) for m, n in MAIN if m in p]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_assoc(test, obc):
    """Association variants (ranked: bold best, underlined second best),
    then the pose-inflation ablation of HiPer's association."""
    mets = [("success", "All"), ("success_stationary", "Stat."),
            ("success_relocated", "Reloc."), ("success_reappeared", "Reapp."),
            ("novel_false_match", r"Novel FM$\downarrow$")]
    ap = test["assoc_ablations_persistence"]
    rows = [r"\begin{tabular}{@{}l" + "c" * len(mets) + "c|c@{}}",
            r"\toprule",
            r" & \multicolumn{%d}{c|}{THOR-Lifelong held-out} & ObCh.\\"
            % (len(mets) + 1),
            "Appearance & " + " & ".join(l for _, l in mets)
            + r" & NLL$\downarrow$ & All\\", r"\midrule"]
    ids = [c for c, _ in ASSOC if c in test["association"]]
    main = [c for c in ids if c != "abl_no_pose_inflation"]
    vals = {}
    for c in ids:
        t = test["association"][c]
        o = obc["association"].get(c) if obc else None
        vals[c] = ([t[m][0] for m, _ in mets]
                   + [ap[c]["nll"][0] if c in ap else float("nan")]
                   + [o["success"][0] if o else float("nan")])
    keys = [m for m, _ in mets] + ["nll", "success"]
    rk = [ranks({c: vals[c][j] for c in main}, k) for j, k in enumerate(keys)]
    for c, name in ASSOC:
        if c not in vals:
            continue
        cells = [deco(fmt(v), rk[j].get(c)) if c in main else fmt(v)
                 for j, v in enumerate(vals[c])]
        rows.append(f"{name} & " + " & ".join(cells) + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def numbers(test, obc, loro=None, fresh=None, old=None, rscan=None):
    out = []

    def put(name, x, nd=3):
        out.append(f"\\newcommand{{\\{name}}}{{{fmt(x, nd)}}}")
    for tag, s in (("Thor", test), ("Obc", obc), ("Loro", loro),
                   ("Fresh", fresh), ("Old", old), ("Rscan", rscan)):
        if s is None:
            continue
        p = s["persistence"]
        for mid, short in (("ours", "Ours"), ("pf_learned", "Pfl"),
                           ("pf_nbr", "Pfn"), ("ours_class", "Cls"),
                           ("ours_env", "Env"), ("pf_env", "Pfe"),
                           ("pf_reloc", "Pfr"),
                           ("abl_dino_only_retrieval", "Dino"),
                           ("abl_clip_class_keys", "ClsAbl"),
                           ("pf", "Pf"), ("pf_class", "Pfc"),
                           ("ours_hand_prior", "Hand"), ("pocd", "Pocd"),
                           ("elite", "Elite"), ("draft_v1", "Draft"),
                           ("abl_no_rate_learning", "NoLearn"),
                           ("abl_no_class_prior", "NoPrior"),
                           ("abl_reset_rate_on_relocation", "Reset"),
                           ("diag_hand_prior_gt_labels", "HandGt"),
                           ("diag_pf_hand_gt_labels", "PfcGt"),
                           ("abl_clip_appearance", "ClipApp"),
                           ("abl_hand_prior", "HandAbl"),
                           ("always", "Never")):
            for cid, csh in ((mid, short), (mid + "_cal", short + "Cal")):
                if cid not in p:
                    continue
                for met, mshort in (("nll", "Nll"), ("ece", "Ece"),
                                    ("brier", "Brier"), ("auroc", "Auroc"),
                                    ("ap_absent", "Apa"),
                                    ("premature_prune_rate", "Prem"),
                                    ("stale_rate", "Stale")):
                    put(f"r{tag}{csh}{mshort}", p[cid][met][0])
        put(f"r{tag}Scenes", p["ours"]["nll"][4], 0)
        put(f"r{tag}Labeled", p["ours"]["n_labeled"][0] *
            p["ours"]["nll"][4], 0)
        t = s.get("timing", {})
        put(f"r{tag}MapMs", 1000.0 * t.get("mapping_s_per_session",
                                           float("nan")), 0)
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5",
                    help="run of the held-out summary and ObChange (sim)")
    ap.add_argument("--test-split", default="heldout",
                    help="pooled held-out sets (scripts/pool_heldout.py)")
    ap.add_argument("--loro-run", default="v6",
                    help="run of the leave-one-room-out summary")
    ap.add_argument("--earlier-run", default="v3",
                    help="run of the development test splits")
    ap.add_argument("--rscan-run", default="rscan_v1",
                    help="run of the 3RScan test summary (rscan_test)")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    test, obc = load(args.run, args.test_split), load(args.run, "obchange")
    if test is None:
        raise SystemExit("no test summary yet")
    (out / "tab_persistence.tex").write_text(table_persistence(test))
    loro = load(args.loro_run, "obchange_loro")
    if obc is not None and loro is not None:
        (out / "tab_obchange.tex").write_text(table_obchange(obc, loro))
    (out / "tab_assoc.tex").write_text(table_assoc(test, obc))
    # earlier splits, each looked at during development
    old, fresh = (load(args.earlier_run, "test"),
                  load(args.earlier_run, "test2"))
    if old is not None and fresh is not None:
        (out / "tab_earlier.tex").write_text(table_earlier(old, fresh))
    rscan = load(args.rscan_run, "rscan_test")
    if rscan is not None:
        (out / "tab_rscan.tex").write_text(table_rscan(rscan))
    (out / "numbers.tex").write_text(numbers(test, obc, loro, fresh, old,
                                             rscan))
    print("wrote tables: held-out", args.run, args.test_split, "| loro",
          args.loro_run, "| earlier", args.earlier_run)


if __name__ == "__main__":
    main()
