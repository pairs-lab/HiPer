#!/usr/bin/env python3
"""
Additional analyses of the saved prediction rows (no new runs):

  data/nll_by_session.csv   per-scene mean NLL at each session index k
                            (prediction for session k+1), mean over scenes
                            with 95% bootstrap CI, per model
  data/per_sequence.csv     per-sequence NLL / AUROC of every model
  data/pred_hist.csv        share of predictions per 0.1 bin, for objects
                            that leave / stay, HiPer and PF(lambda)
  tab_groups.tex            NLL / AUROC by object group (fixed, moveable,
                            pickupable), THOR test
  extra_numbers.tex         macros quoted in the text

  python paper/figures/make_extra_results.py --run v2 --split test
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import conftest  # noqa: E402,F401

from hiper.evaluation import persistence_metrics as pm  # noqa: E402
from hiper.evaluation.paths import RESULTS_ROOT  # noqa: E402

OUT = Path(__file__).resolve().parent
MODELS = [("ours", "HiPer"), ("pf_nbr", "PF, retrieved rates"),
          ("ours_class", "HiPer, CLIP class rates"),
          ("pf_learned", "PF, CLIP class rates"), ("pf", r"PF($\lambda$)"),
          ("pf_reloc", r"PF($\lambda$), reloc.-aware"),
          ("elite", "ELite scalar"), ("pocd", "POCD-style")]
GROUPS = [("pickupable", "pickupable"), ("moveable", "moveable"),
          ("fixed", "fixed")]
# tab_groups: ours and the baselines (ablations: tab_ablation.tex)
GROUP_MAIN = ["ours", "pf", "pf_reloc", "elite", "pocd"]


def load(run, split):
    rows = [json.loads(l) for l in open(RESULTS_ROOT / run / split /
                                          "pred.jsonl")]
    keep = {m for m, _ in MODELS}
    return [r for r in rows if r["model"] in keep
            and r["pred_presence"] is not None
            and r["label_next"] is not None]


def scene_mean_ci(vals_by_scene, n_boot=2000, seed=0):
    v = np.array([x for x in vals_by_scene if np.isfinite(x)])
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n_boot, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), \
        float(np.percentile(b, 97.5))


def nll_by_session(rows):
    acc = defaultdict(lambda: defaultdict(list))   # (model, k) -> scene -> nll
    for r in rows:
        acc[(r["model"], r["k"])][r["scene"]].append(
            (r["pred_presence"], r["label_next"]))
    out = defaultdict(dict)
    for (m, k), by_scene in acc.items():
        per = [pm.nll(np.array([p for p, _ in v]), np.array([y for _, y in v]))
               for v in by_scene.values()]
        out[m][k] = scene_mean_ci(per)
    return out


def by_group(rows):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["model"], r["gt_group"])][r["scene"]].append(
            (r["pred_presence"], r["label_next"]))
    res = {}
    for (m, g), by_scene in acc.items():
        nll, au = [], []
        for v in by_scene.values():
            p = np.array([a for a, _ in v])
            y = np.array([b for _, b in v])
            nll.append(pm.nll(p, y))
            au.append(pm.auroc(p, y))
        au = [x for x in au if np.isfinite(x)]
        res[(m, g)] = (float(np.mean(nll)),
                       float(np.mean(au)) if au else float("nan"),
                       sum(len(v) for v in by_scene.values()))
    return res


def per_sequence(rows):
    """{model: {scene: {nll, auroc, ...}}} and HiPer's win counts."""
    per = pm.per_scene_metrics(rows)
    wins = {}
    for m, _ in MODELS:
        if m == "ours":
            continue
        sc = [s for s in per["ours"] if s in per[m]]
        wins[m] = (sum(per[m][s]["nll"] > per["ours"][s]["nll"] for s in sc),
                   sum(per[m][s]["auroc"] < per["ours"][s]["auroc"]
                       for s in sc), len(sc))
    return per, wins


def pred_hist(rows, models=("ours", "pf", "pf_reloc"), nbins=10):
    """Share of predictions per bin for objects that leave (label 0) and
    stay (label 1), and the confident errors (p > 0.9 for an object that
    leaves, p < 0.1 for one that stays)."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    hist, stats = {}, {}
    for m in models:
        p = np.array([r["pred_presence"] for r in rows if r["model"] == m])
        y = np.array([r["label_next"] for r in rows if r["model"] == m])
        for lab, name in ((0, "leave"), (1, "stay")):
            h, _ = np.histogram(p[y == lab], bins=edges)
            hist[f"{m}_{name}"] = h / max(1, h.sum())
        stats[m] = {"leave_low": float(np.mean(p[y == 0] < 0.5)),
                    "stay_high": float(np.mean(p[y == 1] > 0.5)),
                    "leave_median": float(np.median(p[y == 0])),
                    "conf_err": int(np.sum((y == 0) & (p > 0.9))
                                    + np.sum((y == 1) & (p < 0.1))),
                    "n": int(len(p))}
    return edges, hist, stats


# Ablation study (tab_ablation.tex): HiPer, then one component removed per
# row; (key, label, source, re-tuned). Sources: "p" persistence models,
# "a" association ablations (HiPer's belief replayed on other evidence).
ABLATIONS = [
    (r"Prior (Sec.~\ref{sec:prior})", [
        ("ours_class", "w/o retrieval: CLIP class rates", "p", True),
        ("abl_hand_prior", "w/o retrieval: hand-set table", "p", False),
        ("abl_dino_only_retrieval", "w/o CLIP, geometry in the key",
         "p", False),
        ("abl_no_class_prior", "w/o prior: one pooled rate", "p", False)]),
    (r"Per-object belief (Sec.~\ref{sec:filter})", [
        ("pf_nbr", "w/o per-object learning", "p", True),
        ("pf_learned", "w/o per-object learning, retrieval", "p",
         True)]),
    (r"Relocation (Sec.~\ref{sec:filter})", [
        ("abl_reset_rate_on_relocation", "w/o rate kept at relocation",
         "p", False)]),
    (r"Association (Sec.~\ref{sec:assoc})", [
        ("e3_none", "w/o appearance (geometry only)", "a", False),
        ("abl_no_pose_inflation", "w/o pose inflation", "a", False)]),
]
ABL_METS = [("nll", r"NLL$\downarrow$", False), ("brier",
            r"Brier$\downarrow$", False), ("auroc", r"AUC$\uparrow$", True)]


def ablation_rows(run, splits):
    """HiPer's labelled prediction rows and the association-ablation rows of
    the given result splits (scenes named '<split>:<scene>' when pooled)."""
    rows = []
    for sp in splits:
        tag = f"{sp}:" if len(splits) > 1 else ""
        for name in ("pred.jsonl", "pred_assoc_ablations.jsonl"):
            with open(RESULTS_ROOT / run / sp / name) as f:
                for line in f:
                    r = json.loads(line)
                    if name == "pred.jsonl" and r["model"] != "ours":
                        continue
                    if r["pred_presence"] is None or r["label_next"] is None:
                        continue
                    r["scene"] = tag + r["scene"]
                    rows.append(r)
    return rows


def ablation_table(datasets):
    """tab_ablation.tex: one column group per dataset (NLL, Brier, AUROC),
    daggers = paired bootstrap over sequences vs HiPer; bold best and
    underlined second best over all rows. ``datasets``: [(title, summary,
    raw rows of HiPer + association ablations)]."""
    per, paired = [], []
    for _, summ, rows in datasets:
        pp = dict(summ["paired_vs_ours"])
        ps = pm.per_scene_metrics(rows)
        for key in ps:
            if key == "ours":
                continue
            pp[key] = {met: pm.paired_bootstrap(
                {s: v[met] for s, v in ps["ours"].items()},
                {s: v[met] for s, v in ps[key].items()})
                for met, _, _ in ABL_METS}
        per.append({**summ["persistence"],
                    **summ["assoc_ablations_persistence"]})
        paired.append(pp)
    keys = ["ours"] + [k for _, grp in ABLATIONS for k, _, _, _ in grp]
    rank = []
    for d in range(len(datasets)):
        for met, _, higher in ABL_METS:
            shown = {k: f"{per[d][k][met][0]:.3f}" for k in keys
                     if k in per[d]}
            lv = sorted(set(shown.values()), key=float, reverse=higher)
            tag = dict(zip(lv[:2], ("best", "second")))
            rank.append({k: tag.get(v) for k, v in shown.items()})
    ncol = 1 + len(datasets) * len(ABL_METS)

    def cells(k):
        out, j = [], 0
        for d in range(len(datasets)):
            for met, _, _ in ABL_METS:
                if k not in per[d]:
                    out.append("--")
                    j += 1
                    continue
                c = f"{per[d][k][met][0]:.3f}"
                r = rank[j].get(k)
                c = (rf"\textbf{{{c}}}" if r == "best" else
                     rf"\underline{{{c}}}" if r == "second" else c)
                q = paired[d].get(k, {}).get(met)
                if k != "ours" and q and (q["ci_lo"] > 0 or q["ci_hi"] < 0):
                    c += r"$^\dagger$"
                out.append(c)
                j += 1
        return " & ".join(out)
    lines = [r"\begin{tabular}{@{}l" + "|".join(
        "c" * len(ABL_METS) for _ in datasets) + r"@{}}", r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{{len(ABL_METS)}}}{{c{'|' if d < len(datasets) - 1 else ''}}}{{{t}}}"
            for d, (t, _, _) in enumerate(datasets)) + r"\\",
        "Variant & " + " & ".join(h for _ in datasets
                                   for _, h, _ in ABL_METS) + r"\\",
        r"\midrule", rf"\hiper (full) & {cells('ours')}\\"]
    for title, grp in ABLATIONS:
        lines += [r"\midrule", rf"\multicolumn{{{ncol}}}{{@{{}}l}}"
                  rf"{{\emph{{{title}}}}}\\"]
        for k, label, _, retuned in grp:
            mark = r"$^\ddagger$" if retuned else ""
            lines.append(rf"\hspace{{0.5em}}{label}{mark} & {cells(k)}\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def assoc_macros(run, split, rscan_run):
    """Association numbers quoted in the ablation text."""
    def pct(x):
        return f"{100 * x:.0f}"
    mac = {}
    t = json.load(open(RESULTS_ROOT / run / split / "summary.json"))
    a, ap = t["association"], t["assoc_ablations_persistence"]
    mac.update({"xAsThorAll": pct(a["ours"]["success"][0]),
                "xAsThorReloc": pct(a["ours"]["success_relocated"][0]),
                "xAsThorNovel": pct(a["ours"]["novel_false_match"][0]),
                "xAsThorGeoReloc": pct(a["e3_none"]["success_relocated"][0]),
                "xAsThorGeoNll": ap["e3_none"]["nll"][0],
                "xAsThorNoPoseNll": ap["abl_no_pose_inflation"]["nll"][0]})
    rp = RESULTS_ROOT / rscan_run / "rscan_test" / "summary.json"
    if rp.exists():
        r = json.load(open(rp))
        ra, rap = r["association"], r["assoc_ablations_persistence"]
        mac.update({"xAsRsAll": pct(ra["ours"]["success"][0]),
                    "xAsRsReloc": pct(ra["ours"]["success_relocated"][0]),
                    "xAsRsNovel": pct(ra["ours"]["novel_false_match"][0]),
                    "xAsRsGeoReloc": pct(ra["e3_none"]["success_relocated"][0]),
                    "xAsRsGeoNll": rap["e3_none"]["nll"][0]})
    return mac


def rscan_history(run="rscan_v1"):
    """POST HOC on the 3RScan test environments (after their single
    evaluation): share of labelled predictions made without any earlier
    session of the object (k = 0), and HiPer vs. the same model without
    per-object learning on the environments with >= 3 rescans."""
    pred = RESULTS_ROOT / run / "rscan_test" / "pred.jsonl"
    if not pred.exists():
        return {}
    rows = [r for r in map(json.loads, open(pred))
            if r["model"] in ("ours", "abl_no_rate_learning")
            and r["pred_presence"] is not None
            and r["label_next"] is not None]
    ours = [r for r in rows if r["model"] == "ours"]
    sessions = defaultdict(set)
    for r in ours:
        sessions[r["scene"]].add(r["k"])
    long_ = {s for s, ks in sessions.items() if max(ks) + 1 >= 3}
    per = pm.per_scene_metrics([r for r in rows if r["scene"] in long_])
    d = pm.paired_bootstrap({s: v["nll"] for s, v in per["ours"].items()},
                            {s: v["nll"] for s, v in
                             per["abl_no_rate_learning"].items()})
    return {"xRsKzeroPct": f"{100 * np.mean([r['k'] == 0 for r in ours]):.0f}",
            "xRsLongEnvs": len(long_),
            "xRsLongPct": f"{100 * np.mean([r['scene'] in long_ for r in ours]):.0f}",
            # paired_bootstrap(a, b): mean of a - b = HiPer - no learning
            "xRsLongDiff": d["diff"], "xRsLongLo": d["ci_lo"],
            "xRsLongHi": d["ci_hi"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v5")
    ap.add_argument("--loro-run", default="v6")
    ap.add_argument("--analysis-run", default="v3",
                    help="val summary, training-size and diagnostic runs "
                         "(same THOR parameters as the final run)")
    ap.add_argument("--split", default="heldout")
    args = ap.parse_args()
    rows = load(args.run, args.split)
    (OUT / "data").mkdir(exist_ok=True)
    nbs = nll_by_session(rows)
    ks = sorted({k for m in nbs for k in nbs[m]})
    with open(OUT / "data/nll_by_session.csv", "w") as f:
        f.write("k," + ",".join(f"{m},{m}_lo,{m}_hi" for m, _ in MODELS)
                + "\n")
        for k in ks:
            f.write(f"{k}," + ",".join(
                ",".join(f"{x:.4f}" for x in nbs[m].get(k, (np.nan,) * 3))
                for m, _ in MODELS) + "\n")
    grp = by_group(rows)
    # AUROC is undefined for a group with one label class (fixed objects
    # never change): such a group gets an NLL column only
    has_auc = {g: any(np.isfinite(grp[(m, g)][1]) for m, _ in MODELS
                      if (m, g) in grp) for g, _ in GROUPS}
    names = dict(MODELS)
    main = [m for m in GROUP_MAIN if all((m, g) in grp for g, _ in GROUPS)]

    def ranks(vals, higher):
        """{model: 'best' | 'second'} by printed value (ties share)."""
        shown = {m: f"{v:.3f}" for m, v in vals.items() if np.isfinite(v)}
        levels = sorted(set(shown.values()), key=float, reverse=higher)
        tag = dict(zip(levels[:2], ("best", "second")))
        return {m: tag.get(v) for m, v in shown.items()}

    def deco(c, r):
        return (rf"\textbf{{{c}}}" if r == "best" else
                rf"\underline{{{c}}}" if r == "second" else c)
    rk = {}
    for g, _ in GROUPS:
        rk[(g, 0)] = ranks({m: grp[(m, g)][0] for m in main}, False)
        if has_auc[g]:
            rk[(g, 1)] = ranks({m: grp[(m, g)][1] for m in main}, True)
    ncols = sum(2 if has_auc[g] else 1 for g, _ in GROUPS)
    lines = [r"\begin{tabular}{@{}l" + "c" * ncols + r"@{}}",
             r"\toprule",
             " & " + " & ".join(
                 (rf"\multicolumn{{2}}{{c}}{{{lab} ($n$={grp[('ours', g)][2]})}}"
                  if has_auc[g] else
                  rf"{lab} ($n$={grp[('ours', g)][2]})")
                 for g, lab in GROUPS) + r"\\",
             "Method & " + " & ".join(
                 r"NLL$\downarrow$ & AUC$\uparrow$" if has_auc[g] else
                 r"NLL$\downarrow$" for g, _ in GROUPS) + r"\\",
             r"\midrule"]

    def row(m, ranked, indent=""):
        cells = []
        for g, _ in GROUPS:
            nll, au, _ = grp[(m, g)]
            c = f"{nll:.3f}"
            cells.append(deco(c, rk[(g, 0)].get(m)) if ranked else c)
            if has_auc[g]:
                c = f"{au:.3f}"
                cells.append(deco(c, rk[(g, 1)].get(m)) if ranked else c)
        return f"{indent}{names[m]} & " + " & ".join(cells) + r"\\"
    lines += [row(m, True) for m in main]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_groups.tex").write_text("\n".join(lines) + "\n")
    # numbers quoted in the text
    o, b = nbs["ours"], nbs["pf"]
    first, last = ks[0], ks[-1]
    mac = {
        "xNllFirstOurs": o[first][0], "xNllFirstPf": b[first][0],
        "xNllLastOurs": o[last][0], "xNllLastPf": b[last][0],
        "xGapFirst": b[first][0] - o[first][0],
        "xGapLast": b[last][0] - o[last][0],
    }
    gaps = {k: b[k][0] - o[k][0] for k in ks}
    kmax, kmin = max(gaps, key=gaps.get), min(gaps, key=gaps.get)
    mac.update({"xGapMax": gaps[kmax], "xGapMaxK": int(kmax),
                "xGapMin": gaps[kmin], "xGapMinK": int(kmin),
                "xNllMaxGapOurs": o[kmax][0], "xNllMaxGapPf": b[kmax][0]})
    # per-sequence wins of HiPer (figures 1 and 5)
    per, wins = per_sequence(rows)
    with open(OUT / "data/per_sequence.csv", "w") as f:
        f.write("seq," + ",".join(f"{m}_nll,{m}_auroc" for m, _ in MODELS)
                + "\n")
        for s in sorted(per["ours"]):
            f.write(f"{s}," + ",".join(
                f"{per[m][s]['nll']:.4f},{per[m][s]['auroc']:.4f}"
                for m, _ in MODELS) + "\n")
    mac["xSeqN"] = len(per["ours"])
    for m, tag in (("pf", "Pf"), ("pf_reloc", "Pfr"), ("pf_nbr", "Pfn"),
                   ("elite", "Elite"), ("pocd", "Pocd")):
        mac[f"xWinNll{tag}"], mac[f"xWinAuroc{tag}"] = (int(wins[m][0]),
                                                        int(wins[m][1]))
    # predictions for objects that leave / stay (figure 6b)
    edges, hist, st = pred_hist(rows)
    with open(OUT / "data/pred_hist.csv", "w") as f:
        f.write("lo,hi,mid," + ",".join(hist) + "\n")
        for i in range(len(edges) - 1):
            f.write(f"{edges[i]:.2f},{edges[i + 1]:.2f},"
                    f"{(edges[i] + edges[i + 1]) / 2:.2f}," + ",".join(
                        f"{hist[c][i]:.4f}" for c in hist) + "\n")
    for m, tag in (("ours", "Ours"), ("pf", "Pf"), ("pf_reloc", "Pfr")):
        mac[f"xLeaveLow{tag}"] = f"{100 * st[m]['leave_low']:.0f}"
        mac[f"xStayHigh{tag}"] = f"{100 * st[m]['stay_high']:.1f}"
        mac[f"xLeaveMed{tag}"] = st[m]["leave_median"]
        mac[f"xConfErr{tag}"] = f"{st[m]['conf_err']:,}"
    mac["xPredN"] = f"{st['ours']['n']:,}"
    # qualitative figure (scene by its fixed rule): mean predicted presence
    # after the last shown session for objects that stay / leave
    q = json.load(open(OUT / "qualitative_prob.json"))
    qrows = load(q.get("run", args.run), q["split"])
    kq = max(q["sessions"])
    for m, tag in (("ours", "Ours"), ("pf", "Pf"), ("pf_reloc", "Pfr"),
                   ("elite", "Elite"), ("pocd", "Pocd")):
        rs = [r for r in qrows if r["scene"] == q["scene"]
              and r["model"] == m and r["k"] == kq]
        for lab, name in ((1, "Stay"), (0, "Leave")):
            v = [r["pred_presence"] for r in rs if r["label_next"] == lab]
            mac[f"xQual{name}{tag}"] = float(np.mean(v))
    # leave-one-room-out: exact sign-flip test and rooms won, PF vs HiPer
    lo = json.load(open(RESULTS_ROOT / args.loro_run / "obchange_loro" /
                        "summary.json"))
    mac["xLoroPfSignP"] = lo["signflip_vs_ours"]["pf"]["nll"]["p_value"]
    lrows = [json.loads(l) for l in open(RESULTS_ROOT / args.loro_run /
                                         "obchange_loro" / "pred.jsonl")]
    per = pm.per_scene_metrics([r for r in lrows
                                if r["model"] in ("ours", "pf")])
    mac["xLoroPfRoomsBetter"] = int(sum(
        per["pf"][s_]["nll"] < per["ours"][s_]["nll"] for s_ in per["ours"]))
    # oracle with each object's true simulator rate (val, diagnostic;
    # scripts/explore_oracle.py) vs HiPer on val
    orc = RESULTS_ROOT / "explore_cache" / "oracle_val.json"
    vs = RESULTS_ROOT / args.analysis_run / "val" / "summary.json"
    if orc.exists() and vs.exists():
        o_ = json.load(open(orc))
        best = min((v for k, v in o_.items() if k.startswith("hiper_")),
                   key=lambda v: v["nll"])
        pfo = o_["pf_true_rate"]
        v = json.load(open(vs))["persistence"]["ours"]
        mac.update({"xOracleNll": best["nll"], "xOracleAuroc": best["auroc"],
                    "xOracleBrier": best["brier"],
                    "xOraclePfNll": pfo["nll"], "xValOursNll": v["nll"][0],
                    "xValOursAuroc": v["auroc"][0],
                    "xValOursBrier": v["brier"][0]})
    # DIAGNOSTIC (not a method choice): leave-one-room-out with the
    # DINOv2-only retrieval key under the same grids
    ld = RESULTS_ROOT / args.analysis_run / "obchange_loro_dino" / \
        "summary.json"
    if ld.exists():
        q_ = json.load(open(ld))["persistence"]["ours"]
        mac.update({"xLoroDinoNll": q_["nll"][0],
                    "xLoroDinoAuroc": q_["auroc"][0]})
    ts = RESULTS_ROOT / args.analysis_run / "trainsize_val_train+train2.json"
    if not ts.exists():
        ts = RESULTS_ROOT / args.analysis_run / "trainsize_val.json"
    if ts.exists():
        t = json.load(open(ts))
        mac.update({"xTsFewScenes": t[0]["train_scenes"],
                    "xTsAllScenes": t[-1]["train_scenes"],
                    "xTsFewOurs": t[0]["ours"], "xTsAllOurs": t[-1]["ours"],
                    "xTsFewCls": t[0]["ours_class"],
                    "xTsAllCls": t[-1]["ours_class"]})
        mid = next((r for r in t if r["train_scenes"] == 20), None)
        if mid is not None:
            mac.update({"xTsMidScenes": 20, "xTsMidOurs": mid["ours"],
                        "xTsMidCls": mid["ours_class"]})
        with open(OUT / "data/trainsize.csv", "w") as f:
            f.write("scenes,ours,ours_class\n")
            for r in t:
                f.write(f"{r['train_scenes']},{r['ours']:.4f},"
                        f"{r['ours_class']:.4f}\n")
    mac.update(rscan_history())
    mac.update(assoc_macros(args.run, args.split, "rscan_v1"))
    held = json.load(open(RESULTS_ROOT / args.run / args.split /
                          "summary.json"))
    datasets = [("THOR-Lifelong held-out", held,
                 ablation_rows(args.run, held.get("pooled_splits",
                                                  [args.split])))]
    rs = RESULTS_ROOT / "rscan_v1" / "rscan_test" / "summary.json"
    if rs.exists():
        datasets.append(("3RScan test", json.load(open(rs)),
                         ablation_rows("rscan_v1", ["rscan_test"])))
    (OUT / "tab_ablation.tex").write_text(ablation_table(datasets) + "\n")
    (OUT / "extra_numbers.tex").write_text("".join(
        f"\\newcommand{{\\{k}}}"
        f"{{{v if isinstance(v, (int, str)) else f'{v:.3f}'}}}\n"
        for k, v in mac.items()))
    print("\n".join(lines))
    for m, _ in MODELS:
        print(m, {k: round(nbs[m][k][0], 3) for k in ks if k in nbs[m]})


if __name__ == "__main__":
    main()
