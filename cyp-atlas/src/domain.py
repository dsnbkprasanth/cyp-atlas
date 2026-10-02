"""Reviewer task: are the 18k atlas predictions trustworthy, or extrapolation?

Applicability domain = max Tanimoto similarity of each atlas compound to the
CYP training set. Reports the distribution and the in-domain fraction, and adds
a 'max_train_tanimoto' + 'in_domain' column to the atlas.

    python domain.py     # reads data/cyp_labels.csv + data/cyp_atlas.csv

Writes results/atlas_domain.csv (summary) + fig5_domain.png, and
data/cyp_atlas_domain.csv (atlas + domain columns).

ponytail: in-domain cutoff 0.3 Tanimoto (ECFP4) — the common AD threshold; move it
if your reviewers prefer 0.4. Training set subsampled to 4000 for the NN scan (speed);
raise SAMPLE for the final run if you want the exact max.
"""
import csv
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

CUT = 0.3
SAMPLE = 4000  # training compounds sampled for the NN scan


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def load_fps(path, col):
    out = []
    for r in csv.DictReader(open(path)):
        f = fp(r[col])
        if f is not None:
            out.append((r, f))
    return out


def main():
    os.makedirs("results/figures", exist_ok=True)
    train = [f for _, f in load_fps("data/cyp_labels.csv", "smiles")]
    if len(train) > SAMPLE:
        train = random.Random(0).sample(train, SAMPLE)
    atlas = load_fps("data/cyp_atlas.csv", "smiles")

    sims = []
    for r, f in atlas:
        s = max(DataStructs.BulkTanimotoSimilarity(f, train))
        r["max_train_tanimoto"] = round(s, 3)
        r["in_domain"] = int(s >= CUT)
        sims.append(s)
    sims = np.array(sims)

    with open("data/cyp_atlas_domain.csv", "w", newline="") as fout:
        cols = list(atlas[0][0].keys())
        w = csv.DictWriter(fout, fieldnames=cols); w.writeheader()
        for r, _ in atlas:
            w.writerow(r)
    with open("results/atlas_domain.csv", "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["metric", "value"])
        w.writerow(["atlas_compounds", len(sims)])
        w.writerow(["in_domain_frac (>=%.2f)" % CUT, round(float((sims >= CUT).mean()), 3)])
        w.writerow(["median_max_tanimoto", round(float(np.median(sims)), 3)])

    plt.figure(); plt.hist(sims, bins=40)
    plt.axvline(CUT, color="r", ls="--", lw=1, label=f"AD cutoff {CUT}")
    plt.xlabel("max Tanimoto to training set"); plt.ylabel("phytochemicals")
    plt.title("Applicability domain of the atlas"); plt.legend()
    plt.savefig("results/figures/fig5_domain.png", dpi=200, bbox_inches="tight"); plt.close()

    assert len(sims) > 0 and 0 <= sims.max() <= 1
    print(f"in-domain (>= {CUT}): {(sims >= CUT).mean():.1%} of {len(sims)}; "
          f"median max-Tanimoto {np.median(sims):.3f}")


if __name__ == "__main__":
    main()
