"""Train/eval the HDI GNN. Runs on bundled sample data if no CSV is given.

    python train.py                     # smoke test on data/sample_hdi.csv
    python train.py data/your_hdi.csv   # real run
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from data import HDIPairs, collate, make_sample
from featurize import NUM_ATOM_FEATURES
from model import HDIModel


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    ys, ps = [], []
    for a, b, y in dl:
        p = torch.sigmoid(model(a.to(dev), b.to(dev))).cpu()
        ys += y.tolist()
        ps += p.tolist()
    return roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")


def run(csv_path, epochs=10, bs=32, lr=1e-3):
    ds = HDIPairs(csv_path)
    n_val = max(1, len(ds) // 5)
    tr, va = random_split(ds, [len(ds) - n_val, n_val])
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(va, batch_size=bs, collate_fn=collate)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HDIModel(NUM_ATOM_FEATURES).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for a, b, y in dl_tr:
            a, b, y = a.to(dev), b.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(a, b), y).backward()
            opt.step()
        print(f"epoch {ep + 1}  val_auc {evaluate(model, dl_va, dev):.3f}")
    return model


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_hdi.csv"
    if not os.path.exists(path):
        print(f"{path} missing -> writing sample")
        make_sample(path)
    run(path, epochs=5)
    print("OK: pipeline ran end to end.")  # ponytail check: fails loudly if any step breaks
