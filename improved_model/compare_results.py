"""Side-by-side baseline vs improved.  Usage: python compare_results.py"""
import csv, os
cols = ["p_at_1", "mrr", "h_at_5", "f1_at_k", "jaccard", "DDI_rate", "group_rate", "CMS"]
def load(tag):
    path = f"{tag}_results.csv"
    if not os.path.exists(path):
        return {}
    return {int(r["train_size"]): r for r in csv.DictReader(open(path))}
base, imp = load("baseline"), load("improved")
print(f"{'size':>5} {'metric':>10} {'baseline':>9} {'improved':>9} {'change':>8}")
for n in sorted(set(base) | set(imp)):
    for c in cols:
        b = float(base[n][c]) if n in base and base[n][c] else None
        i = float(imp[n][c]) if n in imp and imp[n][c] else None
        d = f"{i - b:+.4f}" if b is not None and i is not None else ""
        fmt = lambda v: f"{v:9.4f}" if v is not None else f"{'-':>9}"
        print(f"{n:5d} {c:>10} {fmt(b)} {fmt(i)} {d:>8}")
    print()
print("(DDI_rate / group_rate / CMS: lower is safer)")
