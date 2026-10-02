# Source provenance

Traceability for every external-derived dataset. Checksums and sizes are in
`DATA_MANIFEST.csv` / `DATA_MANIFEST.json`.

## CYP450 / P-gp inhibition labels — `cyp-atlas/src/data/cyp_labels.csv`
- **Source:** ChEMBL bioactivity database.
- **Source version:** ChEMBL v34.
- **Retrieval:** `cyp-atlas/src/fetch_cyp.py` (stdlib `urllib`; `--demo` produces an offline synthetic set).
- **Filter criteria:** IC50 assays against CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4, and P-glycoprotein; activity binarized at pChEMBL ≥ 5.
- **License:** ChEMBL data is released under CC BY-SA 3.0. Attribute ChEMBL/EMBL-EBI.
- **Identifiers to preserve when extending:** chembl_id, assay_id, target_id, molecule_id, activity_id (retain these if you re-pull, for full auditability).

## Phytochemical structures — input to the atlas (`imppat_phytocompounds.sdf`, external)
- **Source:** IMPPAT 2.0 (Indian Medicinal Plants, Phytochemistry And Therapeutics), https://cb.imsc.res.in/imppat/
- **Redistribution:** NOT included in this repository. See `data/external/README.md` for retrieval.
- **Derived, included output:** `cyp-atlas/src/data/cyp_atlas.csv` (model-scored predictions over these structures) is released here.
- **License:** follow IMPPAT 2.0 terms of use; cite IMPPAT.

## Curated HDI seed — `hdi-gnn/src/data/hdi_seed.csv`
- **Positives:** hand-curated, cited herb–drug interactions; one PMID per pair recorded in the file.
- **Constituent SMILES:** ChEMBL, with hyperforin/bergamottin from NCI CACTUS.
- **Drug SMILES:** ChEMBL.
- **Negatives:** not in the seed; `build_dataset.py` adds *assumed*-negative co-administration pairs (documented sampling rule in that script) to produce `hdi.csv`.
- **Note:** provenance table `hdi-gnn/src/results/tables/tableS1_provenance.csv` mirrors this seed.

## Derived files
- `cyp_atlas_domain.csv` — from `cyp_atlas.csv` via `domain.py` (applicability domain).
- `hdi.csv` — from `hdi_seed.csv` via `build_dataset.py`.
- `sample_hdi.csv` — synthetic (random labels) for smoke testing only.

## Drug → enzyme routes — `cyp-atlas/src/data/drug_cyp.csv`
- Manually curated CYP/transporter substrate/route assignments for the validation drugs (includes P-gp routes; induction is not modelled — stated as a limitation).
