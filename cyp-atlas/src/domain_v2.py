"""Applicability-domain filter + strict (reviewer-grade) atlas delivery.

Reviewer: 62% of 17,963 phytochemicals fall outside AD -> atlas is extrapolative.
Fix: deliver the filtered atlas as the primary artefact, keep the flagged full
atlas as supplementary. Also sweep AD cutoffs (0.3 / 0.35 / 0.40 / 0.50) so the
paper can defend its choice numerically rather than by convention.

    python domain_v2.py            # reads data/cyp_labels.csv + data/cyp_atlas.csv
                                   # writes data/cyp_atlas_indomain.csv (primary)
                                   # writes results/revision/ad_sweep.csv

ponytail: AD = max Tanimoto to training (ECFP4, 2048 bits). Standard, 1 line of math,
no deeper AD method needed until the referees ask for leverage/density/kNN-stdev.
"""
import csv, os, random
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

DEFAULT_CUT = 0.40
SAMPLE = 4000


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None


def load(path, col):
    out = []
    for r in csv.DictReader(open(path)):
        f = fp(r[col])
        if f is not None:
            out.append((r, f))
    return out


def main():
    os.makedirs("results/revision", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    train = [f for _, f in load("data/cyp_labels.csv", "smiles")]
    if len(train) > SAMPLE:
        train = random.Random(0).sample(train, SAMPLE)
    atlas = load("data/cyp_atlas.csv", "smiles")

    sims = []
    for r, f in atlas:
        s = max(DataStructs.BulkTanimotoSimilarity(f, train))
        r["max_train_tanimoto"] = round(s, 3)
        r["in_domain_040"] = int(s >= 0.40)
        sims.append(s)
    sims = np.array(sims)

    cols = list(atlas[0][0].keys())
    with open("data/cyp_atlas_domain.csv", "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=cols); w.writeheader()
        for r, _ in atlas:
            w.writerow(r)
    # Primary deliverable: in-domain compounds only
    with open("data/cyp_atlas_indomain.csv", "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=cols); w.writeheader()
        for (r, _), s in zip(atlas, sims):
            if s >= DEFAULT_CUT:
                w.writerow(r)

    # AD cutoff sweep (reviewer: don't just pick 0.3)
    with open("results/revision/ad_sweep.csv", "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["AD_Tanimoto_cutoff", "in_domain_n", "in_domain_frac",
                    "median_similarity", "mean_similarity"])
        for cut in (0.20, 0.30, 0.35, 0.40, 0.50):
            m = sims >= cut
            w.writerow([cut, int(m.sum()), round(float(m.mean()), 3),
                        round(float(np.median(sims)), 3),
                        round(float(np.mean(sims)), 3)])

    plt.figure(figsize=(6, 4)); plt.hist(sims, bins=50)
    for cut, c in [(0.30, "gray"), (0.40, "red"), (0.50, "black")]:
        plt.axvline(cut, color=c, ls="--", lw=1,
                    label=f"AD={cut} ({(sims >= cut).mean():.0%})")
    plt.xlabel("max Tanimoto to training set (ECFP4)")
    plt.ylabel("phytochemicals")
    plt.title(f"Applicability domain (N={len(sims)})")
    plt.legend()
    plt.savefig("results/figures/fig5_domain.png", dpi=200, bbox_inches="tight")
    plt.close()

    n_in = int((sims >= DEFAULT_CUT).sum())
    print(f"AD@{DEFAULT_CUT}: {n_in}/{len(sims)} ({n_in/len(sims):.1%}) in-domain "
          f"-> data/cyp_atlas_indomain.csv (primary)")
    print("full atlas with per-row flag kept at data/cyp_atlas_domain.csv (SI)")


if __name__ == "__main__":
    main()
