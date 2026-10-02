"""Post-run analysis: atlas prevalence + mechanism-split HDI-risk validation.

    python summarize.py

Reads data/cyp_atlas.csv + results/risk_validation.csv (already produced by
atlas.py / risk.py). Writes:
  results/atlas_prevalence.csv + fig2_atlas_prevalence.png   (predicted inhibitors per CYP)
  results/risk_summary.csv     + fig3_risk_box.png           (AUROC overall vs inhibitor-only)

Why the split: the atlas models INHIBITION. Some curated herbs act by INDUCTION
(St John's wort/hyperforin, garlic/allicin) — an inhibition model neither can nor
should flag them, so they belong in an induction analysis, not this one. We report
both numbers honestly.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

TASKS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
INDUCERS = {"hyperforin", "allicin"}   # ponytail: extend as the seed grows
CUT = 0.5


def prevalence():
    rows = list(csv.DictReader(open("data/cyp_atlas.csv")))
    n = len(rows)
    tasks = [c for c in rows[0] if c not in ("name", "smiles")]  # auto: CYPs (+PGP)
    counts = {t: sum(float(r[t]) >= CUT for r in rows) for t in tasks}
    with open("results/atlas_prevalence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["enzyme", "predicted_inhibitors", "total", "fraction"])
        for t in tasks:
            w.writerow([t, counts[t], n, round(counts[t] / n, 3)])
    plt.figure(); plt.bar(tasks, [counts[t] / n for t in tasks])
    plt.ylabel(f"fraction predicted inhibitor (p>={CUT})")
    plt.title(f"IMPPAT atlas: predicted liability ({n} phytochemicals)")
    plt.xticks(rotation=20)
    plt.savefig("results/figures/fig2_atlas_prevalence.png", dpi=200, bbox_inches="tight"); plt.close()
    print("prevalence:", counts, f"of {n}")


def _stats(rs, n_boot=2000, seed=0):
    import numpy as np
    y = np.array([int(r["label"]) for r in rs]); s = np.array([float(r["risk"]) for r in rs])
    if len(set(y)) < 2:
        return (float("nan"),) * 4 + (len(rs), int(y.sum()))
    au = roc_auc_score(y, s)
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); boot = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) > 1:
            boot.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # permutation p: how often shuffled labels beat observed AUROC
    perm = sum(roc_auc_score(rng.permutation(y), s) >= au for _ in range(n_boot))
    p = (perm + 1) / (n_boot + 1)
    return round(au, 3), round(lo, 3), round(hi, 3), round(p, 4), len(rs), int(y.sum())


def risk_split():
    rows = list(csv.DictReader(open("results/risk_validation.csv")))
    inhib = [r for r in rows if r["herb"] not in INDUCERS]
    all_s, inh_s = _stats(rows), _stats(inhib)
    with open("results/risk_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "AUROC", "CI95_lo", "CI95_hi", "perm_p", "pairs", "positives"])
        w.writerow(["all_curated_pairs", *all_s])
        w.writerow(["inhibition_mechanism_only", *inh_s])
    all_au, inh_au = (all_s[0], all_s[4]), (inh_s[0], inh_s[4])
    # risk distribution by label (inhibition-mechanism subset)
    pos = [float(r["risk"]) for r in inhib if r["label"] == "1"]
    neg = [float(r["risk"]) for r in inhib if r["label"] == "0"]
    plt.figure(); plt.boxplot([neg, pos])
    plt.xticks([1, 2], ["no interaction", "interaction"])
    plt.ylabel("mechanistic risk score"); plt.title("HDI risk by label (inhibition mechanism)")
    plt.savefig("results/figures/fig3_risk_box.png", dpi=200, bbox_inches="tight"); plt.close()
    print(f"risk AUROC  all={all_au[0]:.3f} (n={all_au[1]})  inhibitor-only={inh_au[0]:.3f} (n={inh_au[1]})")


if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    prevalence()
    risk_split()
    print("wrote results/atlas_prevalence.csv, risk_summary.csv, fig2, fig3")
