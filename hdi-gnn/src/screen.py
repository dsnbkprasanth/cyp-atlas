"""Screen IMPPAT phytochemicals x drug panel for PREDICTED herb-drug interactions.

    python screen.py ../imppat_phytocompounds.sdf [top_n]

Trains RandomForest (best baseline) on data/hdi.csv, scores every
(IMPPAT compound x seed-drug) pair, writes:
  results/screen_top_hits.csv          ranked predictions
  results/figures/fig6_screen.png      score distribution

ponytail: these are PREDICTIONS, not validated interactions. Treat top hits as
hypotheses; confirm against literature/PBPK/wet-lab before any claim.
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from featurize import morgan

RES, DATA, SEED = "results", "data/hdi.csv", "data/hdi_seed.csv"


def train_rf():
    rows = list(csv.DictReader(open(DATA)))
    X = np.array([np.concatenate([morgan(r["smiles_a"]), morgan(r["smiles_b"])]) for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    return RandomForestClassifier(n_estimators=400, random_state=0).fit(X, y)


def main(sdf, top_n=200):
    os.makedirs(f"{RES}/figures", exist_ok=True)
    rf = train_rf()
    drugs = {r["drug"]: r["smiles_b"] for r in csv.DictReader(open(SEED))}
    dfp = {k: morgan(v) for k, v in drugs.items()}

    names, comp = [], []
    for m in Chem.SDMolSupplier(sdf):
        if m is None:
            continue
        try:
            comp.append(morgan(Chem.MolToSmiles(m)))
        except Exception:
            continue
        names.append(m.GetProp("_Name") if m.HasProp("_Name") else "?")
    comp = np.array(comp)

    rows = []
    for dn, fb in dfp.items():
        X = np.hstack([comp, np.tile(fb, (len(comp), 1))])
        p = rf.predict_proba(X)[:, 1]
        rows.extend(zip(names, [dn] * len(names), np.round(p, 3)))
    rows.sort(key=lambda r: -r[2])

    with open(f"{RES}/screen_top_hits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phytochemical", "drug", "pred_prob_INTERACT"])
        w.writerows(rows[:top_n])

    plt.figure(); plt.hist([r[2] for r in rows], bins=40)
    plt.xlabel("predicted P(interact)"); plt.ylabel("pairs")
    plt.title(f"IMPPAT screen: {len(rows)} pairs")
    plt.savefig(f"{RES}/figures/fig6_screen.png", dpi=200, bbox_inches="tight"); plt.close()

    assert rows and 0.0 <= rows[0][2] <= 1.0
    print(f"scored {len(rows)} pairs; top {top_n} -> {RES}/screen_top_hits.csv")


if __name__ == "__main__":
    sdf = sys.argv[1] if len(sys.argv) > 1 else "../imppat_phytocompounds.sdf"
    main(sdf, int(sys.argv[2]) if len(sys.argv) > 2 else 200)
