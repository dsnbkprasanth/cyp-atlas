"""HDI pair dataset. CSV columns: smiles_a, smiles_b, label (0/1)."""
import csv
import os
import random
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from featurize import smiles_to_graph


class HDIPairs(Dataset):
    def __init__(self, csv_path):
        with open(csv_path) as f:
            self.rows = [
                (r["smiles_a"], r["smiles_b"], int(r["label"]))
                for r in csv.DictReader(f)
            ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, b, y = self.rows[i]
        return smiles_to_graph(a), smiles_to_graph(b), torch.tensor(y, dtype=torch.float)


def collate(batch):
    a, b, y = zip(*batch)
    return Batch.from_data_list(a), Batch.from_data_list(b), torch.stack(y)


# ponytail: sample rows are random SMILES pairs with random labels — a pipeline
# smoke-test only, NO real signal. Replace data/sample_hdi.csv with the curated
# HDI benchmark (herb constituent SMILES x co-med SMILES, interaction 0/1).
def make_sample(path, n=200):
    smis = [
        "CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "O=C(O)c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles_a", "smiles_b", "label"])
        for _ in range(n):
            w.writerow([random.choice(smis), random.choice(smis), random.randint(0, 1)])
