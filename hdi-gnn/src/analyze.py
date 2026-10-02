"""Full publication analysis: baselines vs GNN, honest CV, figures + tables.

    python analyze.py            # builds data/hdi.csv if missing, writes results/

Outputs (results/):
  metrics.csv, tables/*.csv, figures/*.png, RESULTS.md

Evaluation is out-of-fold (OOF) prediction pooling — robust on tiny data:
  - random 5-fold stratified CV (all models)
  - leave-one-herb-out (LOHO) for the GNN = cold-start / unseen-herb generalization
    (THE split reviewers demand for herb-drug prediction)

ponytail: OOF-pooled AUROC, no bootstrap CI. add CI when the dataset is large
enough that per-fold variance is the question (hundreds+ pairs).
"""
import csv
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from torch_geometric.data import Batch

from featurize import morgan, smiles_to_graph, NUM_ATOM_FEATURES
from data import HDIPairs, collate
from model import HDIModel
import build_dataset

DATA = "data/hdi.csv"
SEED = "data/hdi_seed.csv"
RES = "results"
RNG = 0


def load():
    rows = list(csv.DictReader(open(DATA)))
    sa = [r["smiles_a"] for r in rows]
    sb = [r["smiles_b"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    # herb group per row: match smiles_a back to constituent name from the seed
    name = {r["smiles_a"]: r["constituent"] for r in csv.DictReader(open(SEED))}
    groups = np.array([name.get(s, "unknown") for s in sa])
    X = np.array([np.concatenate([morgan(a), morgan(b)]) for a, b in zip(sa, sb)])
    return sa, sb, y, groups, X


def oof_baseline(model_fn, X, y, splitter, groups=None):
    pred = np.zeros(len(y), dtype=float)
    for tr, te in splitter.split(X, y, groups):
        m = model_fn().fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def gnn_oof(sa, sb, y, splitter, groups=None, epochs=30):
    graphs = [(smiles_to_graph(a), smiles_to_graph(b)) for a, b in zip(sa, sb)]
    yt = torch.tensor(y, dtype=torch.float)
    pred = np.zeros(len(y), dtype=float)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for tr, te in splitter.split(np.zeros(len(y)), y, groups):
        model = HDIModel(NUM_ATOM_FEATURES).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        a = Batch.from_data_list([graphs[i][0] for i in tr]).to(dev)
        b = Batch.from_data_list([graphs[i][1] for i in tr]).to(dev)
        yy = yt[tr].to(dev)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            loss_fn(model(a, b), yy).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ta = Batch.from_data_list([graphs[i][0] for i in te]).to(dev)
            tb = Batch.from_data_list([graphs[i][1] for i in te]).to(dev)
            pred[te] = torch.sigmoid(model(ta, tb)).cpu().numpy()
    return pred


def metrics(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p)


def auroc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y[s])) < 2:
            continue
        vals.append(roc_auc_score(y[s], p[s]))
    return round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)


def main():
    os.makedirs(f"{RES}/figures", exist_ok=True)
    os.makedirs(f"{RES}/tables", exist_ok=True)
    if not os.path.exists(DATA):
        build_dataset.build(ratio=2)
    sa, sb, y, groups, X = load()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    baselines = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, random_state=RNG),
        "HistGB": lambda: HistGradientBoostingClassifier(random_state=RNG),
        "LogReg": lambda: LogisticRegression(max_iter=1000),
    }
    preds = {name: oof_baseline(fn, X, y, skf) for name, fn in baselines.items()}
    preds["GNN"] = gnn_oof(sa, sb, y, skf)

    # cold-start: leave-one-herb-out, GNN only
    logo = LeaveOneGroupOut()
    gnn_loho = gnn_oof(sa, sb, y, logo, groups=groups)

    # ---- metrics table ----
    hdr = ["model", "split", "AUROC", "AUROC_95CI_lo", "AUROC_95CI_hi", "AUPRC"]
    rows = []
    for name, p in preds.items():
        au, ap = metrics(y, p); lo, hi = auroc_ci(y, p)
        rows.append((name, "random-5fold", round(au, 3), lo, hi, round(ap, 3)))
    au, ap = metrics(y, gnn_loho); lo, hi = auroc_ci(y, gnn_loho)
    rows.append(("GNN", "leave-one-herb-out", round(au, 3), lo, hi, round(ap, 3)))
    for path in (f"{RES}/metrics.csv", f"{RES}/tables/table2_model_comparison.csv"):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

    # ---- table 1: dataset stats ----
    herbs, hcount = np.unique(groups, return_counts=True)
    with open(f"{RES}/tables/table1_dataset_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total pairs", len(y)])
        w.writerow(["positives", int(y.sum())])
        w.writerow(["negatives", int((y == 0).sum())])
        w.writerow(["unique herbs (constituents)", len(herbs)])
    # table S1 = provenance seed, copied verbatim
    with open(SEED) as s, open(f"{RES}/tables/tableS1_provenance.csv", "w") as d:
        d.write(s.read())

    # ---- figures ----
    plt.figure()
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — random 5-fold CV"); plt.legend()
    plt.savefig(f"{RES}/figures/fig1_roc.png", dpi=200, bbox_inches="tight"); plt.close()

    plt.figure()
    for name, p in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — random 5-fold CV"); plt.legend()
    plt.savefig(f"{RES}/figures/fig2_pr.png", dpi=200, bbox_inches="tight"); plt.close()

    plt.figure()
    names = [r[0] + ("*" if r[1].startswith("leave") else "") for r in rows]
    plt.bar(names, [r[2] for r in rows])
    plt.ylabel("AUROC"); plt.ylim(0, 1); plt.axhline(0.5, color="k", ls="--", lw=0.8)
    plt.title("Model comparison (*=cold-herb)"); plt.xticks(rotation=30, ha="right")
    plt.savefig(f"{RES}/figures/fig3_model_compare.png", dpi=200, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.bar(herbs, hcount)
    plt.ylabel("pairs"); plt.title("Pairs per herb constituent"); plt.xticks(rotation=30, ha="right")
    plt.savefig(f"{RES}/figures/fig4_dataset.png", dpi=200, bbox_inches="tight"); plt.close()

    # embedding scatter: train GNN on all data, PCA of pair embeddings
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HDIModel(NUM_ATOM_FEATURES).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ds = HDIPairs(DATA)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
    for _ in range(40):
        model.train()
        for a, b, yy in dl:
            opt.zero_grad(); loss_fn(model(a.to(dev), b.to(dev)), yy.to(dev)).backward(); opt.step()
    model.eval()
    embs = []
    with torch.no_grad():
        for a, b in [(smiles_to_graph(x), smiles_to_graph(z)) for x, z in zip(sa, sb)]:
            za = model.enc(Batch.from_data_list([a]).to(dev))
            zb = model.enc(Batch.from_data_list([b]).to(dev))
            embs.append(torch.cat([za, zb], 1).cpu().numpy()[0])
    emb2 = PCA(n_components=2, random_state=RNG).fit_transform(np.array(embs))
    plt.figure()
    for lab, mk in [(1, "o"), (0, "x")]:
        s = y == lab
        plt.scatter(emb2[s, 0], emb2[s, 1], marker=mk, label=("interact" if lab else "no-interact"))
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("GNN pair embeddings (PCA)"); plt.legend()
    plt.savefig(f"{RES}/figures/fig5_embedding.png", dpi=200, bbox_inches="tight"); plt.close()

    # ---- results note ----
    with open(f"{RES}/RESULTS.md", "w") as f:
        f.write("# Results (auto-generated)\n\n")
        f.write("| model | split | AUROC | 95% CI | AUPRC |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]}-{r[4]} | {r[5]} |\n")
        f.write(f"\nDataset: {len(y)} pairs, {int(y.sum())} pos / {int((y==0).sum())} neg, "
                f"{len(herbs)} herbs.\n\n")
        f.write("> CAVEAT: numbers reflect the current seed size. They become "
                "publication-grade only after the labeled set is expanded (hundreds+ pairs). "
                "Rerun `python analyze.py` after growing data/hdi_seed.csv.\n")
    print("done -> results/ (metrics.csv, tables/, figures/, RESULTS.md)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
