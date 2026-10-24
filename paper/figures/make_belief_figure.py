#!/usr/bin/env python3
"""
Data for fig_hpf_belief.tex: two mugs with the same class prior (lifetime
2 days), observed in daily sessions with the released
HierarchicalPersistenceFilter. Mug A stays put; mug B is moved every second
day (a relocation). PF(lambda_c) (no rate learning) predicts the same value
for both. Parameters: the val-tuned ones from --selected if given, else the
filter defaults. Output: figures/data/hpf_belief.csv
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import conftest  # noqa: E402,F401

from hiper.ephemerality.persistence_v2 import (  # noqa: E402
    HIT, HierarchicalPersistenceFilter)

DAY = 86400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default=None,
                    help="results/<run>/selected.json (tuned HPF params)")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    params = {}
    if args.selected:   # selected.json (stage 1) or final.json
        d = json.load(open(args.selected))
        params = dict(d["families"]["ours"]["params"] if "families" in d
                      else d["hpf"]["params"])
        # filter parameters only (the retrieval keys K, tau, k_nbr set the
        # prior, which this figure fixes to a 2-day lifetime)
        params = {k: v for k, v in params.items()
                  if k in ("p_miss", "p_false", "kappa0", "variance_gate",
                           "tau_pi", "tau_v")}
    table = {"Mug": 1.0 / (2.0 * DAY)}
    hpf = HierarchicalPersistenceFilter(lambda_table=table, **params)
    pf = HierarchicalPersistenceFilter(lambda_table=table, learn_rate=False,
                                       **{k: v for k, v in params.items()
                                          if k in ("p_miss", "p_false")})
    for m in (hpf, pf):
        m.create("A", "Mug", 0.0)
        m.create("B", "Mug", 0.0)
    rows = ["day,a_pred,a_mean,a_sd,a_var,b_pred,b_mean,b_sd,b_var,pf_pred,"
            "tau_v"]
    for d in range(1, args.days + 1):
        t = d * DAY
        for m in (hpf, pf):
            m.observe("A", t, HIT)
            if d % 2 == 0:
                m.relocate("B", t)
            else:
                m.observe("B", t, HIT)
        nxt = t + DAY
        a_v, b_v = hpf.rate_var("A"), hpf.rate_var("B")
        rows.append(",".join(f"{x:.6g}" for x in (
            d, hpf.presence("A", nxt), hpf.rate_mean("A"), a_v ** 0.5, a_v,
            hpf.presence("B", nxt), hpf.rate_mean("B"), b_v ** 0.5, b_v,
            pf.presence("A", nxt), hpf.tau_v)))
    out = ROOT / "paper/figures/data/hpf_belief.csv"
    out.write_text("\n".join(rows) + "\n")
    print("wrote", out, "params", params or "defaults")


if __name__ == "__main__":
    main()
