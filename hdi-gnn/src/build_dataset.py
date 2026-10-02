"""Build a train-ready HDI CSV from the curated seed of real positives.

Reads data/hdi_seed.csv (real, cited herb-drug interactions, label=1) and adds
negatives by pairing constituents x drugs that are NOT known positives.

    python build_dataset.py        # 1:1 neg:pos -> data/hdi.csv
    python build_dataset.py 2      # 2:1 neg:pos

Then train on it: python train.py data/hdi.csv

ponytail: negatives are ASSUMED non-interacting (open-world absence, not a
confirmed negative). Fine for a starter benchmark; state the rule in the paper
and swap in confirmed negatives when you have them.
"""
import csv
import os
import random
import sys

SEED = "data/hdi_seed.csv"
OUT = "data/hdi.csv"


def load_seed(path):
    pos, c_smiles, d_smiles = set(), {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            pos.add((r["constituent"], r["drug"]))
            c_smiles[r["constituent"]] = r["smiles_a"]
            d_smiles[r["drug"]] = r["smiles_b"]
    return pos, c_smiles, d_smiles


def build(seed=SEED, out=OUT, ratio=1, rng_seed=0):
    pos, c_smiles, d_smiles = load_seed(seed)
    cand = [(c, d) for c in c_smiles for d in d_smiles if (c, d) not in pos]
    random.Random(rng_seed).shuffle(cand)
    negs = cand[: ratio * len(pos)]
    assert not (set(negs) & pos), "negative leaked into positives"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles_a", "smiles_b", "label"])
        for c, d in pos:
            w.writerow([c_smiles[c], d_smiles[d], 1])
        for c, d in negs:
            w.writerow([c_smiles[c], d_smiles[d], 0])
    print(f"wrote {out}: {len(pos)} pos + {len(negs)} neg")
    return len(pos), len(negs)


if __name__ == "__main__":
    ratio = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    build(ratio=ratio)
