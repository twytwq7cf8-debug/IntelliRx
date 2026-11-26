"""
Paired significance test: baseline vs improved on the SAME test patients.

Uses the per-patient results already saved by --eval:
    <baseline_root>/output/baseline_<n>/test_detailed_results.json
    <improved_root>/output/improved_<n>/test_detailed_results.json

For each metric and training size it reports:
  - mean baseline, mean improved, difference
  - 95% paired bootstrap confidence interval of the difference
  - p-value of a paired permutation (sign-flip) test
  - for P@1 and Hit@5 (0/1 per patient): exact McNemar test with win/loss counts

Run from the improved_model folder:
    python paired_test.py
    python paired_test.py --baseline_root .. --improved_root . --sizes 200,400,800
Writes paired_test_results.csv.
"""
import argparse
import csv
import json
import os
from math import comb

import numpy as np

METRICS = [("p_at_1", "P@1"), ("mrr", "MRR"), ("h_at_5", "Hit@5"), ("f1_at_k", "F1@5"),
           ("jaccard", "Jaccard"), ("DDI_rate", "DDI rate"), ("group_rate", "Group rate"), ("CMS", "CMS")]
BINARY = {"p_at_1", "h_at_5"}
LOWER_IS_BETTER = {"DDI_rate", "group_rate", "CMS"}


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [r["question"] for r in rows], [r["qa_metrics"] for r in rows]


def mcnemar_exact(b_only, i_only):
    """Two-sided exact McNemar (binomial on discordant pairs)."""
    n = b_only + i_only
    if n == 0:
        return 1.0
    k = min(b_only, i_only)
    p = 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_root", default="..")
    ap.add_argument("--improved_root", default=".")
    ap.add_argument("--baseline_tag", default="baseline")
    ap.add_argument("--improved_tag", default="improved")
    ap.add_argument("--sizes", default="200,400,800")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1203)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out_rows = []
    for n in [int(s) for s in args.sizes.split(",")]:
        pb = os.path.join(args.baseline_root, "output", f"{args.baseline_tag}_{n}", "test_detailed_results.json")
        pi = os.path.join(args.improved_root, "output", f"{args.improved_tag}_{n}", "test_detailed_results.json")
        if not (os.path.exists(pb) and os.path.exists(pi)):
            print(f"[{n}] missing file(s):\n  {pb}\n  {pi}\n")
            continue
        qb, mb = load(pb)
        qi, mi = load(pi)
        if qb != qi:
            print(f"[{n}] WARNING: test patients are not in the same order / not the same set -> skipped")
            continue
        N = len(qb)
        print(f"=== train size {n}  (paired over {N} test patients)")
        print(f"{'metric':>10} {'base':>7} {'impr':>7} {'diff':>7}   {'95% CI of diff':>17}  {'p (perm)':>9}  {'McNemar':>22}")
        for key, name in METRICS:
            b = np.array([m.get(key, 0.0) for m in mb], dtype=float)
            i = np.array([m.get(key, 0.0) for m in mi], dtype=float)
            d = i - b
            obs = d.mean()
            # paired bootstrap CI
            idx = rng.integers(0, N, size=(args.n_boot, N))
            boots = d[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            # sign-flip permutation test
            signs = rng.choice([-1.0, 1.0], size=(args.n_boot, N))
            perm = (signs * d).mean(axis=1)
            p_perm = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (args.n_boot + 1)
            mc = ""
            p_mc = None
            if key in BINARY:
                i_only = int(np.sum((i == 1) & (b == 0)))
                b_only = int(np.sum((b == 1) & (i == 0)))
                p_mc = mcnemar_exact(b_only, i_only)
                mc = f"impr-only {i_only:3d} / base-only {b_only:3d}, p={p_mc:.4f}"
            flag = "*" if p_perm < 0.05 else " "
            print(f"{name:>10} {b.mean():7.4f} {i.mean():7.4f} {obs:+7.4f}   [{lo:+.4f}, {hi:+.4f}]  {p_perm:9.4f}{flag} {mc}")
            out_rows.append({"train_size": n, "metric": name, "baseline": round(b.mean(), 4),
                             "improved": round(i.mean(), 4), "diff": round(obs, 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4), "p_perm": round(p_perm, 4),
                             "p_mcnemar": None if p_mc is None else round(p_mc, 4),
                             "lower_is_better": key in LOWER_IS_BETTER})
        print()
    if out_rows:
        with open("paired_test_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print("* = p < 0.05.  DDI rate / Group rate / CMS: lower is safer (negative diff = safer).")
        print("Saved paired_test_results.csv")


if __name__ == "__main__":
    main()
