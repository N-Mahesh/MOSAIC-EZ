"""Evaluate the binary valence target the paper recommends.

Sec. VI concludes that this archive supports a valence distinction and nothing
finer, but the protocol table reported only five categories and three zones. A
recommendation the paper never evaluates is not a result, so this adds the k=2
arm over the identical images, protocols, features, and head recipe.
"""
import argparse, json
from pathlib import Path
import numpy as np
from leakage_ablation import build_records, evaluate, leakage_groups

VALENCE = {"natural": 0, "joy": 0, "anger": 1, "fear": 1, "sadness": 1}

def cc(v, k):
    return (v - 1.0 / k) / (1.0 - 1.0 / k)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--archive-root", type=Path, required=True)
ap.add_argument("--embeddings", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
a = ap.parse_args()

records = build_records(a.archive_root.resolve())
arch = np.load(a.embeddings, allow_pickle=False)
index = {r: i for i, r in enumerate(arch["rels"].tolist())}
emb = arch["embeddings"]
groups = leakage_groups(records, 4)
trees = sorted({r["tree"] for r in records})
sub = lambda p: [r for r in records if p(r)]
protocols = {
    "vendor_balanced": (sub(lambda r: r["tree"] == trees[0] and r["split"] == "train"),
                        sub(lambda r: r["tree"] == trees[0] and r["split"] == "test")),
    "vendor_talaat":   (sub(lambda r: r["tree"] == trees[1] and r["split"] == "train"),
                        sub(lambda r: r["tree"] == trees[1] and r["split"] == "test")),
    "cross_tree":      (sub(lambda r: r["tree"] == trees[0]),
                        sub(lambda r: r["tree"] == trees[1] and r["split"] == "test")),
}
zoneable = [r for r in records if r["zone"] is not None]
for name, ga in (("naive_random", False), ("group_aware", True)):
    per = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        if ga:
            ug = sorted({groups[r["rel"]] for r in zoneable})
            o = rng.permutation(len(ug))
            tg = {ug[i] for i in o[: int(round(len(ug) * .15))]}
            per.append(([r for r in zoneable if groups[r["rel"]] not in tg],
                        [r for r in zoneable if groups[r["rel"]] in tg]))
        else:
            o = rng.permutation(len(zoneable)); c = int(round(len(zoneable) * .15))
            per.append(([zoneable[i] for i in o[c:]], [zoneable[i] for i in o[:c]]))
    protocols[name] = per

out = {}
for name, spec in protocols.items():
    if isinstance(spec, tuple):
        tr = [r for r in spec[0] if r["zone"] is not None]
        te = [r for r in spec[1] if r["zone"] is not None]
        for r in tr + te: r["_v"] = VALENCE[r["label"]]
        out[name] = evaluate(tr, te, emb, index, {0: 0, 1: 1}, "_v", a.seeds, groups)
    else:
        runs = []
        for seed, (tr, te) in zip(a.seeds, spec):
            for r in tr + te: r["_v"] = VALENCE[r["label"]]
            runs.append(evaluate(tr, te, emb, index, {0: 0, 1: 1}, "_v", [seed], groups))
        agg = dict(runs[0])
        for f in ("balanced_accuracy_mean", "accuracy_mean", "kappa_mean", "nn1_balanced_accuracy",
                  "test_leaked_share", "test_group_leaked_share", "n_test", "n_train"):
            agg[f] = float(np.mean([r[f] for r in runs]))
        agg["balanced_accuracy_std"] = float(np.std([r["balanced_accuracy_mean"] for r in runs], ddof=1))
        agg["n_test"] = int(round(agg["n_test"]))
        out[name] = agg
    print(f"{name:18s} ba={out[name]['balanced_accuracy_mean']:.4f} sd={out[name]['balanced_accuracy_std']:.4f} "
          f"1nn={out[name]['nn1_balanced_accuracy']:.4f} n={out[name]['n_test']}", flush=True)

vals = [e["balanced_accuracy_mean"] for e in out.values()]
ccr = max(cc(v, 2) for v in vals) - min(cc(v, 2) for v in vals)
res = {"seeds": a.seeds, "per_protocol": out, "raw_range": max(vals) - min(vals),
       "chance_corrected_range": ccr}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"\nbinary valence: raw range {max(vals)-min(vals):.4f}  cc-range {ccr:.4f}")
