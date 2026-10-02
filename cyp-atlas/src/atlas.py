"""Apply trained CYP models to the IMPPAT phytochemical library -> predicted atlas.

    python atlas.py ../imppat_phytocompounds.sdf

Writes data/cyp_atlas.csv: name, smiles, P(inhibit) for each CYP enzyme.
This predicted atlas of the phytochemical space is the paper's core resource.

ponytail: predictions, not assays. spot-validate top hits before claims.
"""
import csv
import pickle
import sys
import numpy as np
from rdkit import Chem
from featurize import morgan_safe

MODELS = "models/cyp_models.pkl"
OUT = "data/cyp_atlas.csv"


def main(sdf):
    models = pickle.load(open(MODELS, "rb"))
    tasks = list(models)
    names, smis, feats = [], [], []
    for m in Chem.SDMolSupplier(sdf):
        if m is None:
            continue
        smi = Chem.MolToSmiles(m)
        fp = morgan_safe(smi)
        if fp is None:
            continue
        names.append(m.GetProp("_Name") if m.HasProp("_Name") else "?")
        smis.append(smi)
        feats.append(fp)
    X = np.array(feats)
    preds = {t: models[t].predict_proba(X)[:, 1] for t in tasks}
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "smiles"] + tasks)
        for i, nm in enumerate(names):
            w.writerow([nm, smis[i]] + [round(float(preds[t][i]), 3) for t in tasks])
    assert len(names) > 0
    print(f"atlas: {len(names)} phytochemicals x {len(tasks)} enzymes -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../imppat_phytocompounds.sdf")
