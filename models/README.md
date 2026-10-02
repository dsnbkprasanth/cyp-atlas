# Model artifacts

The trained per-enzyme model bundle is **not stored in this Git repository** because
of its size (~333 MB), which exceeds GitHub's 100 MB per-file limit.

## `cyp_models.pkl`

- **Contents:** six trained scikit-learn `RandomForestClassifier` models
  (`n_estimators=300`, `random_state=0`), one per label:
  `CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4, PGP`.
- **Features:** Morgan/RDKit fingerprints of compound SMILES (see `cyp-atlas/src/featurize.py`).
- **Training data:** `cyp-atlas/src/data/cyp_labels.csv` (14,637 compounds; ChEMBL-derived,
  pChEMBL ≥ 5, IC50 assays). SHA-256 of the training data is in `metadata/CHECKSUMS.sha256`.
- **Seed:** `random_state=0`.
- **Approx. size:** 333,783,159 bytes.

## How to obtain it

Regenerate exactly from source (no download needed):

```bash
cd cyp-atlas/src
python fetch_cyp.py          # or: python fetch_cyp.py --demo   (offline synthetic smoke test)
python train_cyp.py          # writes models/cyp_models.pkl + results/metrics.csv
```

Or download the archived checkpoint from the Zenodo release once published
(DOI to be added here after archival).

> After regenerating or downloading, verify the checkpoint's SHA-256 and record
> it alongside the release. Tree-model training is deterministic given the fixed
> seed and the same library versions.
