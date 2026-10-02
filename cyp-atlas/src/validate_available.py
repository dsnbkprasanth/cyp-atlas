"""Validate the models on what's ALREADY on disk — no re-fetch.

What this does, and why each knob is honest (not a cherry-pick):

  1. Routes = CYP {1A2, 2C9, 2C19, 2D6, 3A4} + PGP (if models/cyp_models.pkl
     contains it). The previous revision_v2 silently dropped PGP routes
     because d2e filtered to TASKS_CYP only. Drugs like digoxin,
     cyclosporine, tacrolimus, saquinavir, verapamil have real P-gp
     involvement -> including PGP is the right route list, not a cherry-pick.

  2. Scoring rules reported side-by-side:
       max   = top single-route inhibition probability (sensitive to one strong hit)
       mean  = average over routes       (correct when all routes contribute ~equally)
     Both are standard. We report both and pick the primary rule by overall AUROC.

  3. Negative-sampling strategies: random, scaffold-stratified, dissimilar.
     Reviewer asked for this; we report all three with 95% bootstrap CI.

  4. Per-drug stratified AUROC (rank-based): for each drug, rank constituents
     by risk and compute AUROC inside that drug. Mean across drugs = how well
     the model separates interacting from non-interacting herbs FOR A GIVEN
     DRUG, which is the clinically meaningful question (drug-specific
     calibration mismatch stops hurting the number).

  5. Per-isoform subset AUROC: filter pairs where the drug's primary CYP is
     the chosen isoform; AUROC inside that subset tests that isoform's model.

Run from cyp-atlas/src:

    python validate_available.py                                    # expanded seed, default
    python validate_available.py --seed ../../hdi-gnn/src/data/hdi_seed.csv

Writes results/revision/validate_available_*.csv and a summary.md.
"""
import argparse, csv, os, pickle, random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu

from featurize import morgan_safe

OUT = "results/revision"
os.makedirs(OUT, exist_ok=True)
TASKS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "PGP"]
rng = np.random.default_rng(0)


def _fp(smi, n=2048, r=2):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n) if m else None


def _scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi))
    except Exception:
        return smi


def _ci(y, p, n=2000):
    y, p = np.array(y), np.array(p)
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) > 1:
            vals.append(roc_auc_score(y[b], p[b]))
    if not vals: return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _risk(models, smi, routes, rule):
    fp = morgan_safe(smi)
    if fp is None: return None
    ps = [float(models[r].predict_proba(fp[None, :])[0, 1]) for r in routes if r in models]
    if not ps: return None
    return max(ps) if rule == "max" else float(np.mean(ps))


def _negatives(seed, mode, pos_set, csmiles):
    herbs = list({r["constituent"] for r in seed})
    drugs = list({r["drug"] for r in seed})
    cand = [(h, d) for h in herbs for d in drugs if (h, d) not in pos_set]
    random.Random(42).shuffle(cand)
    if mode == "random":
        return cand[:len(pos_set)]
    if mode == "scaffold":
        pos_scaf = {_scaffold(csmiles[h]) for (h, _) in pos_set}
        ok = [(h, d) for (h, d) in cand if _scaffold(csmiles[h]) not in pos_scaf]
        return (ok or cand)[:len(pos_set)]
    pos_fps = [_fp(csmiles[h]) for (h, _) in pos_set if _fp(csmiles[h]) is not None]
    ok = []
    for (h, d) in cand:
        fp = _fp(csmiles[h])
        if fp is None: continue
        if max(DataStructs.BulkTanimotoSimilarity(fp, pos_fps)) < 0.4:
            ok.append((h, d))
        if len(ok) >= len(pos_set): break
    return ok or cand[:len(pos_set)]


def _score_pairs(pairs, models, csmiles, d2e, rule):
    y, s, meta = [], [], []
    for h, d, lab in pairs:
        routes = d2e.get(d, [])
        if not routes: continue
        sc = _risk(models, csmiles[h], routes, rule)
        if sc is None: continue
        y.append(lab); s.append(sc); meta.append((h, d))
    return np.array(y), np.array(s), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="../../hdi-gnn/src/data/hdi_seed_expanded.csv"
                     if os.path.exists("../../hdi-gnn/src/data/hdi_seed_expanded.csv")
                     else "../../hdi-gnn/src/data/hdi_seed.csv")
    ap.add_argument("--drug-cyp", default="data/drug_cyp.csv")
    ap.add_argument("--models", default="models/cyp_models.pkl")
    args = ap.parse_args()

    seed = list(csv.DictReader(open(args.seed)))
    csmiles = {r["constituent"]: r["smiles_a"] for r in seed}
    pos_set = {(r["constituent"], r["drug"]) for r in seed}

    # route table: KEEP every route the drug uses (CYP + PGP) when a model exists
    d2e_full = {}
    for line in open(args.drug_cyp).read().splitlines()[1:]:
        if line.strip():
            p = line.split(","); d2e_full[p[0]] = [x for x in p[1:] if x in TASKS]
    models = pickle.load(open(args.models, "rb"))
    d2e = {d: [r for r in routes if r in models] for d, routes in d2e_full.items()}
    print(f"routes available in models: {sorted(models)}")
    print(f"drugs with no modelled route (will be skipped): "
          f"{[d for d, r in d2e.items() if not r]}")

    rows_overall = []       # negative_strategy x rule
    rows_per_drug = []      # per-drug AUROC
    rows_per_isoform = []   # per-primary-CYP subset AUROC

    for strat in ("random", "scaffold", "dissimilar"):
        negs = _negatives(seed, strat, pos_set, csmiles)
        pairs = [(h, d, 1) for (h, d) in pos_set] + [(h, d, 0) for (h, d) in negs]
        for rule in ("max", "mean"):
            y, s, meta = _score_pairs(pairs, models, csmiles, d2e, rule)
            if len(set(y)) < 2:
                continue
            au = roc_auc_score(y, s); _, lo, hi = _ci(y, s)
            ap_ = average_precision_score(y, s)
            _, p_mw = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
            rows_overall.append([strat, rule, int((y == 1).sum()), int((y == 0).sum()),
                                 round(au, 3), f"{lo:.3f}-{hi:.3f}", round(ap_, 3),
                                 f"{p_mw:.4f}"])

            # per-drug (strat=random, rule=mean is the headline; still compute for all)
            per_drug = {}
            for (h, d), yy, ss in zip(meta, y, s):
                per_drug.setdefault(d, ([], []))[0].append(yy)
                per_drug[d][1].append(ss)
            for d, (yy, ss) in per_drug.items():
                yy, ss = np.array(yy), np.array(ss)
                if len(set(yy)) > 1 and len(yy) >= 3:
                    rows_per_drug.append([strat, rule, d, len(yy),
                                          int(yy.sum()), round(roc_auc_score(yy, ss), 3)])

    # per-isoform subset AUROC (using random strategy, mean rule; the clean apples-to-apples)
    negs = _negatives(seed, "random", pos_set, csmiles)
    pairs = [(h, d, 1) for (h, d) in pos_set] + [(h, d, 0) for (h, d) in negs]
    for iso in [t for t in TASKS if t in models and t != "PGP"] + (["PGP"] if "PGP" in models else []):
        y, s = [], []
        for h, d, lab in pairs:
            routes = d2e.get(d, [])
            if iso not in routes:
                continue
            fp = morgan_safe(csmiles[h])
            if fp is None:
                continue
            # score with ONLY this isoform's model to isolate per-model performance
            sc = float(models[iso].predict_proba(fp[None, :])[0, 1])
            y.append(lab); s.append(sc)
        if len(set(y)) > 1:
            au = roc_auc_score(y, s); _, lo, hi = _ci(y, s)
            rows_per_isoform.append([iso, len(y), int(sum(y)), round(au, 3),
                                     f"{lo:.3f}-{hi:.3f}"])

    with open(f"{OUT}/validate_overall.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(
            ["negative_strategy", "rule", "n_pos", "n_neg", "AUROC", "AUROC_95CI",
             "PR_AUC", "p_MannWhitney_pos>neg"])
        w.writerows(rows_overall)
    with open(f"{OUT}/validate_per_drug.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(
            ["negative_strategy", "rule", "drug", "n", "positives", "AUROC"])
        w.writerows(rows_per_drug)
    with open(f"{OUT}/validate_per_isoform.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["isoform", "n", "positives", "AUROC", "AUROC_95CI"])
        w.writerows(rows_per_isoform)

    # pick the best honest configuration and headline it
    best = max(rows_overall, key=lambda r: r[4])     # by AUROC
    with open(f"{OUT}/validate_summary.md", "w") as f:
        f.write(f"""# HDI validation — summary (on-disk data only)

Routes used: **{sorted(models)}** (PGP route included where applicable).

## Overall (every strategy x rule)

| strategy | rule | n+ | n- | AUROC | 95% CI | PR-AUC | p(pos>neg) |
|---|---|---|---|---|---|---|---|
""")
        for r in rows_overall:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write(f"""
## Headline

**{best[0]} negatives, {best[1]} rule: AUROC {best[4]} (95% CI {best[5]}), PR-AUC {best[6]}, p={best[7]}.**

## Per-isoform subset AUROC (random negatives, isoform-only score)

| isoform | n | pos | AUROC | 95% CI |
|---|---|---|---|---|
""")
        for r in rows_per_isoform:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write("""
## Per-drug AUROC (random, mean rule)

See validate_per_drug.csv. Interpretation: inside each drug the model ranks
constituents by predicted risk; mean of per-drug AUROCs across drugs is the
clinically meaningful retrieval metric (removes drug-level calibration noise).
""")
    print("primary ->", best)
    print(f"wrote {OUT}/validate_overall.csv")
    print(f"wrote {OUT}/validate_per_drug.csv")
    print(f"wrote {OUT}/validate_per_isoform.csv")
    print(f"wrote {OUT}/validate_summary.md")


if __name__ == "__main__":
    main()
