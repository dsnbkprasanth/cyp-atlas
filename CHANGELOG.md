# Changelog

## v1.1.0 — 2026-10-02

Reviewer-response pipeline added. Code-only + frozen data; no manuscript artefacts in repo.

- NEW cyp-atlas/src/fetch_cyp_v2.py — configurable IC50 threshold (--threshold-um); P-gp restricted to inhibition assays (assay_type=B; description-filtered; substrate/transport/efflux rejected).
- NEW cyp-atlas/src/domain_v2.py — ships cyp_atlas_indomain.csv as primary deliverable; AD sweep 0.20..0.50.
- NEW cyp-atlas/src/revision_v2.py — threshold sensitivity, 3 negative-sampling strategies with bootstrap CI, distant-only HDI subset, route-weighted scoring, P-gp inhibitor-only.
- NEW cyp-atlas/src/validate_available.py — headline validation on expanded seed with PGP route included; per-drug and per-isoform stratification.
- NEW hdi-gnn/src/data/hdi_seed_expanded.csv — 50 documented HDI pairs, 32 constituents, PMID per pair.
- NEW cyp-atlas/src/run_revision.sh — one-command reproduction.
- NEW data/frozen/ — frozen CSV outputs backing every claim in metadata/DATA_MANIFEST.csv.
- Regenerated metadata/CHECKSUMS.sha256 across the entire release tree.

## v1.0.0 — 2026-09-16

Initial public deposition.

- CYP-Atlas source code released (per-enzyme CYP/P-gp models, atlas generation,
  mechanistic HDI risk, applicability domain, benchmarks, revision analyses).
- HDI-GNN source code released (twin-GCN, baselines, dataset build, IMPPAT screen).
- Frozen datasets released: CYP training labels (14,637 compounds), predicted
  CYP-inhibition atlas (17,963 phytochemicals) with applicability-domain flags,
  and the curated HDI seed benchmark (55 cited positive pairs).
