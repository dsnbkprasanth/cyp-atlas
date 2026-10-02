"""Download large public CYP450 inhibition data from ChEMBL -> data/cyp_labels.csv.

Runs on YOUR machine (needs internet to ChEMBL / EBI).

    python fetch_cyp.py           # real download (thousands of compounds x 5 CYPs)
    python fetch_cyp.py --demo    # tiny synthetic file to test the pipeline offline

Source: ChEMBL bioactivity, IC50 vs each CYP (target IDs verified below).
Label: inhibitor(1) if pChEMBL >= 5 (IC50 <= 10 uM), else 0. Aggregated per compound.
Output columns: smiles, CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4  (blank = untested)

ponytail: IC50 only, pChEMBL>=5 cutoff. add Ki/other assays or shift the cutoff if
your endpoint needs it — one edit in classify()/STD_TYPES.
"""
import csv
import json
import os
import sys
import time
import urllib.request

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGETS = {  # verified via ChEMBL target search
    "CYP1A2": "CHEMBL3356", "CYP2C9": "CHEMBL3397", "CYP2C19": "CHEMBL3622",
    "CYP2D6": "CHEMBL289", "CYP3A4": "CHEMBL340",
    "PGP": "CHEMBL4302",   # P-glycoprotein 1 / ABCB1 / MDR1 (ponytail: verify on ChEMBL if 0 rows)
}
STD_TYPE = "IC50"
OUT = "data/cyp_labels.csv"


def _get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def classify(a):
    pv = a.get("pchembl_value")
    if pv not in (None, ""):
        return 1 if float(pv) >= 5 else 0
    sv, unit = a.get("standard_value"), a.get("standard_units")
    if sv not in (None, "") and unit == "nM":
        return 1 if float(sv) <= 10000 else 0
    return None


def fetch_target(cid):
    """{canonical_smiles: label} aggregated (mean>=0.5) over all IC50 measurements."""
    acc, offset = {}, 0
    while True:
        d = _get(f"{BASE}?target_chembl_id={cid}&standard_type={STD_TYPE}&limit=1000&offset={offset}")
        acts = d.get("activities", [])
        for a in acts:
            smi, lab = a.get("canonical_smiles"), classify(a)
            if smi and lab is not None:
                acc.setdefault(smi, []).append(lab)
        total = d["page_meta"]["total_count"]
        offset += 1000
        print(f"    {cid}: {min(offset, total)}/{total}")
        if offset >= total or not acts:
            break
        time.sleep(0.2)
    return {s: (1 if sum(v) / len(v) >= 0.5 else 0) for s, v in acc.items()}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tasks = {}  # smiles -> {enzyme: label}
    for enz, cid in TARGETS.items():
        print(f"fetching {enz} ({cid}) ...")
        for smi, lab in fetch_target(cid).items():
            tasks.setdefault(smi, {})[enz] = lab
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles"] + list(TARGETS))
        for smi, labs in tasks.items():
            w.writerow([smi] + [labs.get(t, "") for t in TARGETS])
    print(f"wrote {OUT}: {len(tasks)} unique compounds")


def demo(path=OUT, n=300):
    import random
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            "O=C(O)c1ccccc1O", "COc1ccccc1", "CCN(CC)CC", "c1ccncc1"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles"] + list(TARGETS))
        for _ in range(n):
            w.writerow([random.choice(base)] + [random.choice([0, 1, ""]) for _ in TARGETS])
    print(f"demo -> {path} ({n} rows)")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
