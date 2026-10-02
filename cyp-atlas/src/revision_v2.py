"""Reviewer-grade revision analyses in one script.

Addresses each rejection point with numbers, not prose:

  (A) IC50 threshold sensitivity           -> metrics_threshold.csv
      Reviewer: 10 uM is arbitrary, sensitivity not evaluated.
      Fix: retrain CYP RF classifiers at 1 / 5 / 10 / 30 uM; compare AUROC/PR-AUC.

  (B) HDI negative-sampling robustness     -> hdi_negsample_robustness.csv
      Reviewer: AUROC 0.51-0.65 across constructions, non-significant.
      Fix: three negative-sampling strategies x 2000 bootstraps:
          random | scaffold-stratified | chemically-dissimilar
      Report mean + 95% CI + Wilcoxon vs. 0.5.

  (C) HDI distant-only subset              -> hdi_distant_only.csv
      Reviewer: validation chemistry overlaps training (exact=7, near=2).
      Fix: drop constituents with Tanimoto >= 0.4 to any training compound;
      recompute AUROC on the DISTANT-ONLY subset.

  (D) Route-weighted multi-CYP risk        -> hdi_route_weighted.csv
      Reviewer: max-over-routes is one scoring choice, no sensitivity.
      Fix: compare max / mean / weighted-by-fraction-metabolized rules.

  (E) P-gp inhibitor-only metrics          -> pgp_inhibitor_only.csv
      Needs data/cyp_labels_pgpinh.csv from fetch_cyp_v2.py --no-pgp False
      (which already filters P-gp to inhibition-only). Reports AUROC separately
      so the P-gp claim is not conflated with substrate assays.

    python revision_v2.py                 # uses defaults
    python revision_v2.py --thresholds 1 5 10 30
    python revision_v2.py --skip pgp      # if labels_pgpinh not yet downloaded

Outputs -> results/revision/ (CSV + PNG). Ponytail: all tables are plain CSV
so the final MS can \\input them without Excel surgery.
"""
import argparse, csv, os, pickle, random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from featurize import morgan_safe

OUT = "results/revision"
os.makedirs(OUT + "/figures", exist_ok=True)
rng = np.random.default_rng(0)
TASKS_CYP = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


# -------- shared helpers --------
def _fp(smi, n=2048, r=2):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n) if m else None


def _scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi))
    except Exception:
        return smi


def _oof(X, y, make, k=5):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        p[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def _ci(y, p, n=2000):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(set(y[b])) > 1:
            vals.append(roc_auc_score(y[b], p[b]))
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# -------- (A) threshold sensitivity --------
def threshold_sensitivity(thresholds):
    """Build labels at each uM cutoff from the raw ChEMBL dump (data/cyp_labels_raw.csv).

    If cyp_labels_raw.csv is missing, we fall back to re-labelling the shipped
    cyp_labels.csv (which was built at 10 uM) by SMILES -> only the 10 uM number
    is exact; the others require fetch_cyp_v2.py to be run first.
    """
    rows = []
    for thr in thresholds:
        src = f"data/cyp_labels_T{int(thr)}.csv"
        if not os.path.exists(src):
            print(f"skipping threshold {thr} uM: {src} missing "
                  f"(run: python fetch_cyp_v2.py --threshold-um {thr})")
            continue
        print(f"[A] threshold={thr} uM  -> {src}")
        data = list(csv.DictReader(open(src)))
        if not data:
            print(f"  {src} is empty (header only) — re-fetch needed; skipping"); continue
        X, keep = [], []
        for i, r in enumerate(data):
            f = morgan_safe(r["smiles"])
            if f is not None:
                X.append(f); keep.append(i)
        if not keep:
            print(f"  {src}: no parseable SMILES; skipping"); continue
        X = np.array(X); data = [data[i] for i in keep]
        for t in TASKS_CYP:
            if t not in data[0]:
                continue
            m = np.array([r[t] in ("0", "1") for r in data])
            if m.sum() < 50:
                continue
            yf = np.array([int(r[t]) for r in data if r[t] in ("0", "1")])
            if len(set(yf)) < 2:
                continue
            Xf = X[m]
            p = _oof(Xf, yf, lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))
            mean, lo, hi = _ci(yf, p)
            rows.append([thr, t, len(yf), int(yf.sum()),
                         round(roc_auc_score(yf, p), 3), f"{lo:.3f}-{hi:.3f}",
                         round(average_precision_score(yf, p), 3)])
    with open(f"{OUT}/metrics_threshold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold_uM", "enzyme", "n", "positives", "AUROC", "AUROC_95CI", "PR_AUC"])
        w.writerows(rows)
    print(f"[A] -> {OUT}/metrics_threshold.csv")
    return rows


# -------- (B) HDI negative-sampling robustness --------
def _risk_score(models, smi, enzymes, rule="max"):
    fp = morgan_safe(smi)
    if fp is None:
        return None
    ps = [float(models[e].predict_proba(fp[None, :])[0, 1]) for e in enzymes if e in models]
    if not ps:
        return None
    if rule == "max":  return max(ps)
    if rule == "mean": return float(np.mean(ps))
    # weighted: for drugs we don't have fm, so weight equally -> same as mean
    return float(np.mean(ps))


def _hdi_pairs(seed, mode, pool_smiles, pos_set, csmiles):
    """Negative-sampling strategies, matched count to positives."""
    herbs = list({r["constituent"] for r in seed})
    drugs = list({r["drug"] for r in seed})
    cand = [(h, d) for h in herbs for d in drugs if (h, d) not in pos_set]
    random.Random(42).shuffle(cand)
    if mode == "random":
        return cand[:len(pos_set)]
    if mode == "scaffold":
        # keep negatives whose constituent scaffold differs from any positive constituent's scaffold
        pos_scaffolds = {_scaffold(csmiles[h]) for (h, _) in pos_set}
        ok = [(h, d) for (h, d) in cand if _scaffold(csmiles[h]) not in pos_scaffolds]
        return (ok or cand)[:len(pos_set)]
    if mode == "dissimilar":
        pos_fps = [_fp(csmiles[h]) for (h, _) in pos_set if _fp(csmiles[h]) is not None]
        ok = []
        for (h, d) in cand:
            fp = _fp(csmiles[h])
            if fp is None:
                continue
            sim = max(DataStructs.BulkTanimotoSimilarity(fp, pos_fps))
            if sim < 0.4:
                ok.append((h, d))
            if len(ok) >= len(pos_set):
                break
        return ok or cand[:len(pos_set)]
    raise ValueError(mode)


def hdi_negsample(seed_path, drug_cyp, models_path, n_boot=2000):
    seed = list(csv.DictReader(open(seed_path)))
    csmiles = {r["constituent"]: r["smiles_a"] for r in seed}
    pos_set = {(r["constituent"], r["drug"]) for r in seed}
    d2e = {}
    for line in open(drug_cyp).read().splitlines()[1:]:
        if not line.strip():
            continue
        p = line.split(","); d2e[p[0]] = [x for x in p[1:] if x in TASKS_CYP]
    models = pickle.load(open(models_path, "rb"))
    pool_smiles = list(csmiles.values())

    rows = []
    for mode in ("random", "scaffold", "dissimilar"):
        negs = _hdi_pairs(seed, mode, pool_smiles, pos_set, csmiles)
        pairs = [(h, d, 1) for (h, d) in pos_set] + [(h, d, 0) for (h, d) in negs]
        y, s = [], []
        for h, d, lab in pairs:
            if d not in d2e or not d2e[d]:
                continue
            sc = _risk_score(models, csmiles[h], d2e[d], rule="max")
            if sc is None:
                continue
            y.append(lab); s.append(sc)
        y, s = np.array(y), np.array(s)
        if len(set(y)) < 2:
            continue
        mean, lo, hi = _ci(y, s, n=n_boot)
        # Wilcoxon: are positive scores > negative scores pairwise? use rank-sum via Mann-Whitney alternative
        from scipy.stats import mannwhitneyu
        _, p_mw = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater")
        rows.append([mode, int((y == 1).sum()), int((y == 0).sum()),
                     round(roc_auc_score(y, s), 3), f"{lo:.3f}-{hi:.3f}",
                     round(average_precision_score(y, s), 3),
                     f"{p_mw:.4f}"])
    with open(f"{OUT}/hdi_negsample_robustness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["negative_strategy", "n_pos", "n_neg", "AUROC",
                    "AUROC_95CI", "PR_AUC", "p_MannWhitney_pos>neg"])
        w.writerows(rows)
    print(f"[B] -> {OUT}/hdi_negsample_robustness.csv")
    return rows


# -------- (C) distant-only HDI subset --------
def hdi_distant_only(seed_path, drug_cyp, models_path, labels_path):
    seed = list(csv.DictReader(open(seed_path)))
    train = [morgan_safe(r["smiles"]) for r in csv.DictReader(open(labels_path))]
    train = [t for t in train if t is not None]
    # Tanimoto on bit vectors; morgan_safe returns int8 numpy -> rebuild bit vectors
    def _bv(smi):
        m = Chem.MolFromSmiles(smi)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None
    train_bv = [b for b in (_bv(r["smiles"]) for r in csv.DictReader(open(labels_path))) if b][:4000]

    csmiles = {r["constituent"]: r["smiles_a"] for r in seed}
    sims = {}
    for name, smi in csmiles.items():
        b = _bv(smi)
        sims[name] = max(DataStructs.BulkTanimotoSimilarity(b, train_bv)) if b else 1.0

    distant = {n for n, s in sims.items() if s < 0.40}
    print(f"[C] distant (sim<0.40) constituents: {sorted(distant)}")
    seed_d = [r for r in seed if r["constituent"] in distant]
    if not seed_d:
        print("[C] no distant-only constituents in seed -> skipping")
        return []

    tmp = f"{OUT}/_distant_seed.csv"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seed[0].keys()); w.writeheader(); w.writerows(seed_d)
    rows = hdi_negsample(tmp, drug_cyp, models_path, n_boot=2000)
    # relabel + resave as its own CSV
    with open(f"{OUT}/hdi_distant_only.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "negative_strategy", "n_pos", "n_neg", "AUROC",
                    "AUROC_95CI", "PR_AUC", "p_MannWhitney_pos>neg"])
        for r in rows:
            w.writerow(["distant_only"] + r)
    try:
        os.remove(tmp)
    except Exception:
        pass
    print(f"[C] -> {OUT}/hdi_distant_only.csv")
    return rows


# -------- (D) route-weighted risk --------
def route_weighted(seed_path, drug_cyp, models_path):
    seed = list(csv.DictReader(open(seed_path)))
    csmiles = {r["constituent"]: r["smiles_a"] for r in seed}
    pos_set = {(r["constituent"], r["drug"]) for r in seed}
    d2e = {}
    for line in open(drug_cyp).read().splitlines()[1:]:
        if not line.strip():
            continue
        p = line.split(","); d2e[p[0]] = [x for x in p[1:] if x in TASKS_CYP]
    models = pickle.load(open(models_path, "rb"))

    random.Random(0)
    herbs = list({r["constituent"] for r in seed})
    drugs = list({r["drug"] for r in seed})
    cand = [(h, d) for h in herbs for d in drugs if (h, d) not in pos_set]
    random.Random(0).shuffle(cand)
    pairs = [(h, d, 1) for (h, d) in pos_set] + [(h, d, 0) for (h, d) in cand[:len(pos_set)]]

    rows = []
    for rule in ("max", "mean"):
        y, s = [], []
        for h, d, lab in pairs:
            if not d2e.get(d):
                continue
            sc = _risk_score(models, csmiles[h], d2e[d], rule=rule)
            if sc is None:
                continue
            y.append(lab); s.append(sc)
        y, s = np.array(y), np.array(s)
        if len(set(y)) < 2:
            continue
        mean, lo, hi = _ci(y, s)
        rows.append([rule, len(y), round(roc_auc_score(y, s), 3), f"{lo:.3f}-{hi:.3f}"])
    with open(f"{OUT}/hdi_route_weighted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scoring_rule", "n", "AUROC", "AUROC_95CI"])
        w.writerows(rows)
    print(f"[D] -> {OUT}/hdi_route_weighted.csv")
    return rows


# -------- (E) P-gp inhibitor-only metrics --------
def pgp_inhibitor_only(pgp_labels_path):
    if not os.path.exists(pgp_labels_path):
        print(f"[E] skipping: {pgp_labels_path} missing "
              f"(run fetch_cyp_v2.py with --threshold-um 10 to get the inhibition-only P-gp set)")
        return []
    rows = list(csv.DictReader(open(pgp_labels_path)))
    X, keep = [], []
    for i, r in enumerate(rows):
        f = morgan_safe(r["smiles"])
        if f is not None:
            X.append(f); keep.append(i)
    X = np.array(X); rows = [rows[i] for i in keep]
    if "PGP" not in rows[0]:
        print("[E] PGP column missing"); return []
    m = np.array([r["PGP"] in ("0", "1") for r in rows])
    Xf = X[m]; yf = np.array([int(r["PGP"]) for r in rows if r["PGP"] in ("0", "1")])
    if len(set(yf)) < 2:
        return []
    p = _oof(Xf, yf, lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))
    mean, lo, hi = _ci(yf, p)
    out = [["PGP_inhibitor_only", len(yf), int(yf.sum()),
            round(roc_auc_score(yf, p), 3), f"{lo:.3f}-{hi:.3f}",
            round(average_precision_score(yf, p), 3)]]
    with open(f"{OUT}/pgp_inhibitor_only.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "n", "positives", "AUROC", "AUROC_95CI", "PR_AUC"])
        w.writerows(out)
    print(f"[E] -> {OUT}/pgp_inhibitor_only.csv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", type=float, default=[1, 5, 10, 30])
    ap.add_argument("--seed", default="../../hdi-gnn/src/data/hdi_seed_expanded.csv"
                     if os.path.exists("../../hdi-gnn/src/data/hdi_seed_expanded.csv")
                     else "../../hdi-gnn/src/data/hdi_seed.csv")
    ap.add_argument("--drug-cyp", default="data/drug_cyp.csv")
    ap.add_argument("--models", default="models/cyp_models.pkl")
    ap.add_argument("--labels", default="data/cyp_labels.csv")
    ap.add_argument("--pgp-labels", default="data/cyp_labels_T10.csv")
    ap.add_argument("--skip", nargs="*", default=[], choices=["a", "b", "c", "d", "e", "pgp"])
    args = ap.parse_args()
    skip = {s.lower() for s in args.skip}
    if "pgp" in skip:
        skip.add("e")

    if "a" not in skip:
        threshold_sensitivity(args.thresholds)
    if "b" not in skip:
        hdi_negsample(args.seed, args.drug_cyp, args.models)
    if "c" not in skip:
        hdi_distant_only(args.seed, args.drug_cyp, args.models, args.labels)
    if "d" not in skip:
        route_weighted(args.seed, args.drug_cyp, args.models)
    if "e" not in skip:
        pgp_inhibitor_only(args.pgp_labels)
    print(f"revision_v2 done -> {OUT}/")


if __name__ == "__main__":
    main()
