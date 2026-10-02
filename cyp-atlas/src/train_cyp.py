"""Train one CYP-inhibition classifier per enzyme (multi-task by shared features).

    python train_cyp.py           # reads data/cyp_labels.csv, writes models/ + results/

Per-task 5-fold CV AUROC (each CYP has its own tested subset). Saves models
(pickle) for atlas.py + risk.py, metrics.csv, and a per-task AUROC bar figure.
"""
import csv
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from featurize import morgan_safe

LABELS = "data/cyp_labels.csv"
TASKS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "PGP"]


def _scaffold(smiles):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))
    except Exception:
        return smiles  # unparseable -> its own group


def load():
    rows = list(csv.DictReader(open(LABELS)))
    feats, scaf, keep = [], [], []
    for i, r in enumerate(rows):
        fp = morgan_safe(r["smiles"])
        if fp is not None:
            feats.append(fp); scaf.append(_scaffold(r["smiles"])); keep.append(i)
    X = np.array(feats)
    rows = [rows[i] for i in keep]
    hdr = rows[0].keys() if rows else []
    Y = {t: np.array([r[t] for r in rows]) for t in TASKS if t in hdr}  # only present columns
    return X, Y, np.array(scaf)


def _rf():
    return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)


def oof(X, y, groups):
    """Random 5-fold AUROC and scaffold 5-fold AUROC (out-of-fold pooled)."""
    mask = np.array([v in ("0", "1") for v in y])
    Xf, yf, gf = X[mask], y[mask].astype(int), groups[mask]
    if len(set(yf)) < 2 or len(yf) < 30:
        return None
    rnd = np.zeros(len(yf))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, yf):
        rnd[te] = _rf().fit(Xf[tr], yf[tr]).predict_proba(Xf[te])[:, 1]
    sca = np.zeros(len(yf))
    nsplit = min(5, len(set(gf)))
    for tr, te in GroupKFold(nsplit).split(Xf, yf, gf):
        sca[te] = _rf().fit(Xf[tr], yf[tr]).predict_proba(Xf[te])[:, 1]
    return roc_auc_score(yf, rnd), roc_auc_score(yf, sca), (Xf, yf)


def main():
    os.makedirs("models", exist_ok=True); os.makedirs("results/figures", exist_ok=True)
    X, Y, groups = load()
    models, metrics = {}, []
    for t in Y:
        res = oof(X, Y[t], groups)
        if res is None:
            print(f"{t}: too few labels, skipped"); continue
        rnd, sca, (Xf, yf) = res
        models[t] = _rf().fit(Xf, yf)
        metrics.append((t, len(yf), int(yf.sum()), round(rnd, 3), round(sca, 3)))
        print(f"{t}: n={len(yf)} pos={int(yf.sum())} random-AUROC={rnd:.3f} scaffold-AUROC={sca:.3f}")
    pickle.dump(models, open("models/cyp_models.pkl", "wb"))
    with open("results/metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["enzyme", "n", "positives", "random_AUROC", "scaffold_AUROC"]); w.writerows(metrics)
    if metrics:
        import numpy as _np
        names = [m[0] for m in metrics]; x = _np.arange(len(names))
        plt.figure()
        plt.bar(x - 0.2, [m[3] for m in metrics], 0.4, label="random split")
        plt.bar(x + 0.2, [m[4] for m in metrics], 0.4, label="scaffold split")
        plt.xticks(x, names, rotation=20); plt.ylabel("CV AUROC"); plt.ylim(0, 1)
        plt.axhline(0.5, color="k", ls="--", lw=.8); plt.legend()
        plt.title("Per-enzyme CYP inhibition model")
        plt.savefig("results/figures/fig1_cyp_auroc.png", dpi=200, bbox_inches="tight"); plt.close()
    assert models, "no models trained — check data/cyp_labels.csv"
    print("saved models/cyp_models.pkl")


if __name__ == "__main__":
    main()
