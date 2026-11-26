"""Collect output/<tag>_<size>/test_metrics.json into one table (printed + <tag>_results.csv).
Usage:  python collect_results.py baseline      (default tag: baseline)"""
import csv
import glob
import json
import os
import re
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
cols = ["p_at_1", "mrr", "h_at_5", "precision_at_k", "recall_at_k", "f1_at_k", "jaccard",
        "answer_presence", "DDI_rate", "group_rate", "CMS", "num_questions"]
rows = []
for path in glob.glob(os.path.join("output", f"{tag}_*", "test_metrics.json")):
    m_size = re.search(rf"{re.escape(tag)}_(\d+)", path)
    if not m_size:
        continue
    m = json.load(open(path))["test_metrics"]
    rows.append({"train_size": int(m_size.group(1)), **{c: m.get(c) for c in cols}})
rows.sort(key=lambda r: r["train_size"])

if not rows:
    print(f"No results yet in output/{tag}_*/test_metrics.json")
else:
    out = f"{tag}_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_size"] + cols)
        w.writeheader()
        w.writerows(rows)
    print("train_size " + " ".join(f"{c:>9s}" for c in cols[:7]))
    for r in rows:
        print(f"{r['train_size']:10d} " + " ".join(f"{(r[c] or 0):9.4f}" for c in cols[:7]))
    print("Saved", out)
