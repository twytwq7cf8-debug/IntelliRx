"""
make_subset.py -- build a smaller, reproducible version of the DrugRec benchmark for TraceDR.

Two modes (auto-detected):

  A) From EXISTING intermediate pickles (already contain 'top_k_drugs'; NO Neo4j needed)
       python make_subset.py --from_intermediate intermediate_data/DrugRec0716 \
                             --out_bench DrugRec_small --frac 0.1
     -> samples each split separately (keeps the original train/dev/test separation)
     -> writes intermediate_data/DrugRec_small/{train,dev,test}.pkl

  B) From the raw DrugRec.json (streamed with ijson, so a multi-GB file is fine)
       python make_subset.py --from_json data/DrugRec.json --out_bench DrugRec_small --n 2000
     -> random sample of N patients, split 80/10/10
     -> if records already contain 'top_k_drugs' they go to intermediate_data/<out_bench>/
        (ready to train); otherwise to data/<out_bench>/ and you must run
        `python train_TraceDR.py --benchmark <out_bench> --retrieval` (needs Neo4j).

The sampled IDs are saved in <out_dir>/subset_ids.json so baseline and improved models
are always compared on the exact same patients.
"""
import argparse
import json
import os
import pickle
import random


def sample_dict(d, k, rng):
    keys = list(d.keys())
    k = min(k, len(keys))
    chosen = sorted(rng.sample(keys, k), key=lambda x: (str(type(x)), x))
    return {key: d[key] for key in chosen}


def from_intermediate(args, rng):
    out_dir = os.path.join(args.intermediate_dir, args.out_bench)
    os.makedirs(out_dir, exist_ok=True)
    ids = {}
    for split in ["train", "dev", "test"]:
        src = os.path.join(args.from_intermediate, f"{split}.pkl")
        with open(src, "rb") as f:
            data = pickle.load(f)
        n_full = len(data)
        k = getattr(args, f"n_{split}") or max(1, int(round(n_full * args.frac)))
        sub = sample_dict(data, k, rng)
        with open(os.path.join(out_dir, f"{split}.pkl"), "wb") as f:
            pickle.dump(sub, f)
        ids[split] = list(sub.keys())
        print(f"{split:5s}: {n_full:6d} -> {len(sub):6d}")
    return out_dir, ids


def learning_curve(args, rng):
    """
    Nested training subsets (1000 in 2000 in 3000 ...) that all share ONE fixed dev set
    and ONE fixed test set, so results across sizes are directly comparable.
    Writes intermediate_data/<out_bench>_<size>/{train,dev,test}.pkl for every size.
    """
    sizes = sorted(int(s) for s in args.train_sizes.split(","))
    n_dev = args.n_dev or 500
    n_test = args.n_test or 500

    fixed = {}
    for split, k in [("dev", n_dev), ("test", n_test)]:
        with open(os.path.join(args.from_intermediate, f"{split}.pkl"), "rb") as f:
            data = pickle.load(f)
        fixed[split] = sample_dict(data, k, rng)
        print(f"{split:5s}: {len(data):6d} -> {len(fixed[split]):6d} (shared by all sizes)")
        del data

    with open(os.path.join(args.from_intermediate, "train.pkl"), "rb") as f:
        train = pickle.load(f)
    keys = list(train.keys())
    rng.shuffle(keys)                      # one permutation -> prefixes are nested
    if sizes[-1] > len(keys):
        print(f"WARNING: largest size {sizes[-1]} > available train {len(keys)}; capping.")

    all_ids = {"dev": list(fixed["dev"]), "test": list(fixed["test"])}
    for size in sizes:
        sub_keys = keys[:min(size, len(keys))]
        bench = f"{args.out_bench}_{size}"
        out_dir = os.path.join(args.intermediate_dir, bench)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
            pickle.dump({k: train[k] for k in sub_keys}, f)
        for split in ["dev", "test"]:
            with open(os.path.join(out_dir, f"{split}.pkl"), "wb") as f:
                pickle.dump(fixed[split], f)
        all_ids[f"train_{size}"] = sub_keys
        print(f"{bench}: train {len(sub_keys)} | dev {n_dev} | test {n_test}")

    out_dir = os.path.join(args.intermediate_dir, f"{args.out_bench}_ids")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, all_ids


def iter_json(path):
    """Yield (key, record) from a JSON list or dict without loading it all into RAM."""
    try:
        import ijson
    except ImportError:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        yield from (data.items() if isinstance(data, dict) else enumerate(data))
        return
    with open(path, "rb") as f:
        first = f.read(1)
        while first.isspace():
            first = f.read(1)
    with open(path, "rb") as f:
        if first == b"[":
            for i, rec in enumerate(ijson.items(f, "item", use_float=True)):
                yield i, rec
        else:
            for k, rec in ijson.kvitems(f, "", use_float=True):
                yield k, rec


def from_json(args, rng):
    print("Counting records ...")
    n_total = sum(1 for _ in iter_json(args.from_json))
    n = min(args.n, n_total)
    chosen = set(rng.sample(range(n_total), n))
    print(f"Total records: {n_total}; sampling {n}")

    picked = []
    for pos, (key, rec) in enumerate(iter_json(args.from_json)):
        if pos in chosen:
            picked.append((key, rec))
    rng.shuffle(picked)

    n_tr, n_dev = int(0.8 * n), int(0.1 * n)
    splits = {
        "train": dict(picked[:n_tr]),
        "dev": dict(picked[n_tr:n_tr + n_dev]),
        "test": dict(picked[n_tr + n_dev:]),
    }

    has_topk = "top_k_drugs" in picked[0][1]
    base = args.intermediate_dir if has_topk else args.data_dir
    out_dir = os.path.join(base, args.out_bench)
    os.makedirs(out_dir, exist_ok=True)
    for split, d in splits.items():
        with open(os.path.join(out_dir, f"{split}.pkl"), "wb") as f:
            pickle.dump(d, f)
        print(f"{split:5s}: {len(d)}")

    if has_topk:
        print("Records already contain 'top_k_drugs' -> ready to train (no --retrieval).")
    else:
        print("Records are raw patients -> run with --retrieval once (needs Neo4j running).")
        print("Top-level keys of first record:", list(picked[0][1].keys()))
    return out_dir, {s: list(d.keys()) for s, d in splits.items()}


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_intermediate", help="dir with existing train/dev/test.pkl (with top_k_drugs)")
    src.add_argument("--from_json", help="path to raw DrugRec.json")
    p.add_argument("--out_bench", default="DrugRec_small")
    p.add_argument("--frac", type=float, default=0.1, help="fraction per split (intermediate mode)")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_dev", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    p.add_argument("--n", type=int, default=2000, help="total patients (json mode)")
    p.add_argument("--train_sizes", default=None,
                   help="learning-curve mode, e.g. 1000,2000,3000,4000,5000 (needs --from_intermediate)")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--intermediate_dir", default="intermediate_data")
    p.add_argument("--seed", type=int, default=1203)
    args = p.parse_args()

    rng = random.Random(args.seed)
    if args.train_sizes:
        assert args.from_intermediate, "--train_sizes needs --from_intermediate"
        out_dir, ids = learning_curve(args, rng)
    else:
        out_dir, ids = from_intermediate(args, rng) if args.from_intermediate else from_json(args, rng)

    with open(os.path.join(out_dir, "subset_ids.json"), "w") as f:
        json.dump({"seed": args.seed, "ids": ids}, f, default=str)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
