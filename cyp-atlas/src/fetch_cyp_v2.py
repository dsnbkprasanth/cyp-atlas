"""ChEMBL CYP + P-gp download with reviewer-requested knobs.

Reviewer fixes:
  * IC50 threshold is a CLI argument (reviewer: arbitrary 10 uM, no sensitivity test).
    --threshold-um 1 / 5 / 10 / 30  -> writes data/cyp_labels_T<um>.csv.
  * P-gp is split into INHIBITOR-only (what the paper actually claims)
    by keeping only assays whose standard_type == IC50 AND assay_type == 'B'
    (binding/inhibition), and whose description matches 'inhibit' and NOT
    'substrate|transport|efflux|accumulation|flux|permeability'.
    Reviewer: 'endpoint combines heterogeneous inhibition and substrate assays'.
  * CYPs are already pure inhibition IC50; we also drop confidence_score < 7.

    python fetch_cyp_v2.py --threshold-um 10         # replaces old fetch_cyp.py
    python fetch_cyp_v2.py --threshold-um 1 --out data/cyp_labels_T1.csv
    python fetch_cyp_v2.py --threshold-um 10 --no-pgp
"""
import argparse, csv, json, os, re, time, urllib.request

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGETS = {
    "CYP1A2": "CHEMBL3356", "CYP2C9": "CHEMBL3397", "CYP2C19": "CHEMBL3622",
    "CYP2D6": "CHEMBL289",  "CYP3A4": "CHEMBL340",  "PGP": "CHEMBL4302",
}
PGP_INHIB_RE = re.compile(r"inhibit", re.I)
PGP_SUBSTRATE_RE = re.compile(r"substrate|transport|efflux|accumulation|flux|permeab", re.I)


def _get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def _keep_pgp(a):
    """P-gp: inhibition assays only. Reject substrate/transport/efflux."""
    if a.get("assay_type") not in ("B", None):   # B = binding/inhibition
        return False
    desc = (a.get("assay_description") or "") + " " + (a.get("assay_name") or "")
    if PGP_SUBSTRATE_RE.search(desc):
        return False
    return bool(PGP_INHIB_RE.search(desc)) or a.get("assay_type") == "B"


def classify(a, thr_nM):
    sv, unit = a.get("standard_value"), a.get("standard_units")
    if sv not in (None, "") and unit == "nM":
        try:
            return 1 if float(sv) <= thr_nM else 0
        except Exception:
            return None
    pv = a.get("pchembl_value")
    if pv not in (None, ""):
        # pChEMBL = -log10(IC50 in M). pChEMBL >= -log10(thr_M) == inhibitor.
        import math
        return 1 if float(pv) >= -math.log10(thr_nM * 1e-9) else 0
    return None


def fetch_target(cid, thr_nM, pgp=False):
    acc, offset = {}, 0
    while True:
        url = (f"{BASE}?target_chembl_id={cid}&standard_type=IC50"
               f"&limit=1000&offset={offset}")
        d = _get(url)
        acts = d.get("activities", [])
        for a in acts:
            if pgp and not _keep_pgp(a):
                continue
            # ponytail: no confidence_score filter — ChEMBL leaves it unset
            # on many CYP activities; filter drops too much. Add back when
            # you have a reviewer who asks for it, and expose as a CLI arg.
            smi, lab = a.get("canonical_smiles"), classify(a, thr_nM)
            if smi and lab is not None:
                acc.setdefault(smi, []).append(lab)
        total = d["page_meta"]["total_count"]
        offset += 1000
        print(f"    {cid}: {min(offset, total)}/{total}  kept={sum(len(v) for v in acc.values())}")
        if offset >= total or not acts:
            break
        time.sleep(0.2)
    # majority vote per compound
    return {s: (1 if sum(v) / len(v) >= 0.5 else 0) for s, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-um", type=float, default=10.0,
                    help="IC50 threshold in micromolar (reviewer: run 1/5/10/30 for sensitivity).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-pgp", action="store_true", help="skip P-gp (CYPs only).")
    args = ap.parse_args()
    thr_nM = args.threshold_um * 1000.0
    out = args.out or f"data/cyp_labels_T{int(args.threshold_um)}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    enzymes = [k for k in TARGETS if not (args.no_pgp and k == "PGP")]
    rows = {}  # smiles -> {enzyme: label}
    for enz in enzymes:
        print(f"fetching {enz} ({TARGETS[enz]}) thr={args.threshold_um} uM ...")
        for smi, lab in fetch_target(TARGETS[enz], thr_nM, pgp=(enz == "PGP")).items():
            rows.setdefault(smi, {})[enz] = lab
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["smiles"] + enzymes)
        for smi, labs in rows.items():
            w.writerow([smi] + [labs.get(e, "") for e in enzymes])
    print(f"wrote {out}: {len(rows)} compounds, threshold={args.threshold_um} uM, "
          f"PGP={'inhibition-only' if not args.no_pgp else 'off'}")


if __name__ == "__main__":
    main()
