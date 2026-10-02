# Data dictionary

Missing values are empty strings unless noted. All SMILES are as retrieved from the
cited source (not re-canonicalized). See `DATA_MANIFEST.csv` for checksums, sizes,
and row/column counts, and `SOURCE_PROVENANCE.md` for source and license details.

## `cyp-atlas/src/data/cyp_labels.csv`
Per-compound CYP450/P-gp inhibition training labels (14,637 rows).

| column | type | meaning |
|---|---|---|
| smiles | string | Compound SMILES (ChEMBL-derived) |
| CYP1A2 | int {0,1} | 1 = inhibitor of CYP1A2 (pChEMBL ≥ 5), 0 = not |
| CYP2C9 | int {0,1} | inhibitor of CYP2C9 |
| CYP2C19 | int {0,1} | inhibitor of CYP2C19 |
| CYP2D6 | int {0,1} | inhibitor of CYP2D6 |
| CYP3A4 | int {0,1} | inhibitor of CYP3A4 |
| PGP | int {0,1} | inhibitor of P-glycoprotein (ABCB1) |

## `cyp-atlas/src/data/cyp_atlas.csv`
Predicted CYP-inhibition atlas over IMPPAT phytochemicals (17,963 rows) — the release resource.

| column | type | meaning |
|---|---|---|
| name | string | Phytochemical name (IMPPAT) |
| smiles | string | Phytochemical SMILES (IMPPAT) |
| CYP1A2..CYP3A4, PGP | float [0,1] | Predicted probability of inhibition per enzyme/transporter |

## `cyp-atlas/src/data/cyp_atlas_domain.csv`
Atlas with applicability-domain flags (17,963 rows). Columns as `cyp_atlas.csv`, plus:

| column | type | meaning |
|---|---|---|
| max_train_tanimoto | float [0,1] | Max Tanimoto similarity to any training compound |
| in_domain | int {0,1} | 1 = inside applicability domain (above similarity threshold) |

## `cyp-atlas/src/data/drug_cyp.csv`
Drug → CYP/transporter route lookup for mechanistic HDI risk (13 rows).

| column | type | meaning |
|---|---|---|
| drug | string | Conventional drug name |
| enzyme | string | One or more CYP/transporter routes (comma-separated within the field) |

## `hdi-gnn/src/data/hdi_seed.csv`
Curated seed of cited positive herb–drug interactions (55 rows). Each row = one documented interaction (label always 1).

| column | type | meaning |
|---|---|---|
| constituent | string | Herb constituent (phytochemical) name |
| drug | string | Co-administered conventional drug |
| smiles_a | string | Constituent SMILES (ChEMBL / PubChem / NCI CACTUS) |
| smiles_b | string | Drug SMILES |
| label | int {1} | 1 = documented interaction (positives only in the seed) |
| mechanism | string | Free-text mechanism note |
| pmid | int | PubMed ID supporting the interaction |

## `hdi-gnn/src/data/hdi.csv`
Train-ready set built from the seed plus sampled assumed-negatives (156 rows).

| column | type | meaning |
|---|---|---|
| smiles_a | string | Constituent SMILES |
| smiles_b | string | Drug SMILES |
| label | int {0,1} | 1 = interacting (from seed), 0 = assumed-negative (sampled pair) |

## `hdi-gnn/src/data/sample_hdi.csv`
Synthetic sample data for the pipeline smoke test (200 rows). **Labels are random on
purpose** — proves the plumbing, not the science. Same three columns as `hdi.csv`.
