"""Ablation table + paired tests vs the baseline, from the saved test results.
Run from TraceDR-model\ablation:  python ablation_summary.py   (optional: --sizes 200,400,800)
Reads: ..\output\baseline_<n>  |  .\output\abl_*_<n>  |  ..\improved_model\output\improved_<n>"""
import argparse, csv, json, os
import numpy as np

VARIANTS = [  # (label, root, tag, listwise, drugs_first)
    ("TraceDR (baseline)", "..", "baseline", "", ""),
    ("+ listwise loss", ".", "abl_listwise", "x", ""),
    ("+ drugs-first graph", ".", "abl_drugsfirst", "", "x"),
    ("+ both (improved)", os.path.join("..", "improved_model"), "improved", "x", "x"),
]
METRICS = [("p_at_1", "P@1"), ("mrr", "MRR"), ("h_at_5", "Hit@5"), ("f1_at_k", "F1@5"),
           ("jaccard", "Jaccard"), ("DDI_rate", "DDI"), ("CMS", "CMS")]

def load(root, tag, n):
    p = os.path.join(root, "output", f"{tag}_{n}", "test_detailed_results.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p, encoding="utf-8"))
    return [r["question"] for r in rows], [r["qa_metrics"] for r in rows]

def perm_p(d, rng, n_iter=10000):
    obs = d.mean()
    perm = (rng.choice([-1.0, 1.0], size=(n_iter, len(d))) * d).mean(axis=1)
    return (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_iter + 1)

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="800")
args = ap.parse_args()
rng = np.random.default_rng(1203)
out = []
for n in [int(s) for s in args.sizes.split(",")]:
    base = load("..", "baseline", n)
    print(f"\n=== train size {n}   (* = significantly different from baseline, p<0.05, paired permutation test)")
    print(f"{'variant':22s} {'LW':>2} {'DF':>2} " + " ".join(f"{m:>8s}" for _, m in METRICS))
    for label, root, tag, lw, df in VARIANTS:
        res = load(root, tag, n)
        if res is None:
            print(f"{label:22s} {lw:>2} {df:>2}  (no results yet: {root}/output/{tag}_{n})")
            continue
        q, mets = res
        cells, row = [], {"train_size": n, "variant": label, "listwise": bool(lw), "drugs_first": bool(df)}
        for key, name in METRICS:
            v = np.array([m.get(key, 0.0) for m in mets], dtype=float)
            star = " "
            if base is not None and tag != "baseline" and q == base[0]:
                b = np.array([m.get(key, 0.0) for m in base[1]], dtype=float)
                p = perm_p(v - b, rng)
                star = "*" if p < 0.05 else " "
                row[f"{key}_p_vs_base"] = round(p, 4)
            cells.append(f"{v.mean():7.4f}{star}")
            row[key] = round(v.mean(), 4)
        print(f"{label:22s} {lw:>2} {df:>2} " + " ".join(cells))
        out.append(row)
if out:
    keys = sorted({k for r in out for k in r}, key=lambda k: (k not in ("train_size", "variant", "listwise", "drugs_first"), k))
    with open("ablation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    print("\nLW = listwise drug-only loss, DF = drugs-first graph. DDI/CMS: lower is safer.")
    print("Saved ablation_results.csv")
