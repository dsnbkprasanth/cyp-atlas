"""Mechanistic herb-drug interaction risk from the CYP atlas, validated on curated pairs.

risk(herb, drug) = P(herb constituent inhibits the drug's metabolizing CYP).

    python risk.py ../../hdi-gnn/src/data/hdi_seed.csv

Validates the mechanistic score against the curated HDI set (independent of the
CYP training data): builds positives from the seed + sampled negatives, scores
each by predicted enzyme inhibition, reports AUROC. Writes results/risk_validation.csv.

ponytail: drug->enzyme is primary-route only (data/drug_cyp.csv). add secondary
routes + transporters (P-gp) when those models exist. PGP drugs are skipped here.
"""
import csv
import os
import pickle
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from featurize import morgan_safe

MODELS = "models/cyp_models.pkl"
DRUG_CYP = "data/drug_cyp.csv"


def risk_score(models, herb_smiles, enzymes):
    """Max predicted inhibition over the drug's metabolizing CYPs (multi-route)."""
    fp = morgan_safe(herb_smiles)
    if fp is None:
        return None, None
    best, who = None, None
    for e in enzymes:
        if e in models:
            p = float(models[e].predict_proba(fp[None, :])[0, 1])
            if best is None or p > best:
                best, who = p, e
    return best, who     # None if no enzyme modelled (e.g. P-gp only)


def main(seed):
    os.makedirs("results", exist_ok=True)
    models = pickle.load(open(MODELS, "rb"))
    d2e = {}  # drug -> [enzymes]  (variable columns, so parse manually)
    for line in open(DRUG_CYP).read().splitlines()[1:]:
        if line.strip():
            p = line.split(","); d2e[p[0]] = p[1:]

    rows = list(csv.DictReader(open(seed)))
    csmiles = {r["constituent"]: r["smiles_a"] for r in rows}
    pos = {(r["constituent"], r["drug"]) for r in rows}
    herbs, drugs = list(csmiles), list({r["drug"] for r in rows})
    # positives + sampled negatives (herb x drug not known-positive), matched count
    import random
    cand = [(h, d) for h in herbs for d in drugs if (h, d) not in pos]
    random.Random(0).shuffle(cand)
    pairs = [(h, d, 1) for (h, d) in pos] + [(h, d, 0) for (h, d) in cand[:len(pos)]]

    out, y, s = [], [], []
    for h, d, lab in pairs:
        enz = d2e.get(d, [])
        sc, who = risk_score(models, csmiles[h], enz) if enz else (None, None)
        if sc is None:        # no CYP modelled for this drug (e.g. P-gp only) -> excluded
            continue
        out.append((h, d, who, round(sc, 3), lab)); y.append(lab); s.append(sc)
    with open("results/risk_validation.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["herb", "drug", "enzyme", "risk", "label"]); w.writerows(out)
    au = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    print(f"mechanistic HDI risk validation: {len(y)} pairs, AUROC={au:.3f} -> results/risk_validation.csv")
    assert len(y) > 0


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../../hdi-gnn/src/data/hdi_seed.csv")
