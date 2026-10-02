"""Reviewer task: justify the model. RF vs LogReg vs HistGB, per CYP,
random + scaffold split, AUROC + Brier (calibration).

    python benchmark.py     # reads data/cyp_labels.csv -> results/benchmark.csv + fig4

Reuses load()/_scaffold()/TASKS from train_cyp so features + splits match exactly.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from train_cyp import load, TASKS

MODELS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    "LogReg": lambda: LogisticRegression(max_iter=1000),
    "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
}


def oof_pred(make, X, y, splitter, groups=None):
    pred = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups):
        pred[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pred


def main():
    os.makedirs("results/figures", exist_ok=True)
    X, Y, groups = load()
    rows = []
    for t in Y:
        mask = np.array([v in ("0", "1") for v in Y[t]])
        Xf, yf, gf = X[mask], Y[t][mask].astype(int), groups[mask]
        if len(set(yf)) < 2 or len(yf) < 30:
            continue
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        gkf = GroupKFold(min(5, len(set(gf))))
        for name, make in MODELS.items():
            rp = oof_pred(make, Xf, yf, skf)
            sp = oof_pred(make, Xf, yf, gkf, gf)
            rows.append((t, name, round(roc_auc_score(yf, rp), 3),
                         round(roc_auc_score(yf, sp), 3), round(brier_score_loss(yf, rp), 3)))
            print(rows[-1])
    with open("results/benchmark.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["enzyme", "model", "random_AUROC", "scaffold_AUROC", "Brier"]); w.writerows(rows)

    # grouped bar: mean scaffold AUROC per model
    names = list(MODELS)
    means = [np.mean([r[3] for r in rows if r[1] == n]) for n in names]
    plt.figure(); plt.bar(names, means); plt.ylim(0, 1)
    plt.axhline(0.5, color="k", ls="--", lw=.8)
    plt.ylabel("mean scaffold-split AUROC"); plt.title("Model comparison (5 CYPs)")
    plt.savefig("results/figures/fig4_model_benchmark.png", dpi=200, bbox_inches="tight"); plt.close()
    assert rows
    print("wrote results/benchmark.csv, fig4")


if __name__ == "__main__":
    main()
