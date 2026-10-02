# CYP-Atlas + HDI-GNN

Mechanism-grounded prediction of herb–drug interaction (HDI) risk across the
phytochemical space. This repository bundles two linked projects:

- **`cyp-atlas/`** — the headline pipeline. Train per-enzyme CYP450/P-gp inhibition
  models on abundant public ChEMBL bioactivity data, predict a CYP-inhibition
  **atlas** over ~18k IMPPAT phytochemicals, then infer mechanistic, patient-specific
  HDI risk and validate it against curated HDI pairs.
- **`hdi-gnn/`** — a structure-based graph neural network and baseline benchmark
  over a curated seed of cited herb–drug interactions (`cyp-atlas` reuses this seed
  as its validation set).

## Requirements

- Python 3.11, CPU is sufficient for `cyp-atlas` (random forests) and the `hdi-gnn`
  smoke test. `hdi-gnn`'s GNN can optionally use a CUDA build of PyTorch.
- Dependencies: `cyp-atlas/requirements.txt`, `hdi-gnn/requirements.txt` (or `hdi-gnn/environment.yml`).

## Installation

```bash
# cyp-atlas
conda create -n cyp-atlas python=3.11 -y && conda activate cyp-atlas
pip install -r cyp-atlas/requirements.txt

# hdi-gnn (separate env)
conda env create -f hdi-gnn/environment.yml && conda activate hdi-gnn
```

## Data sources & availability

- **CYP inhibition labels** — ChEMBL v34, IC50 assays, pChEMBL ≥ 5 (CC BY-SA 3.0).
  Regenerable via `cyp-atlas/src/fetch_cyp.py`. Frozen copy included.
- **Predicted CYP atlas** (`cyp-atlas/src/data/cyp_atlas.csv`) — released here.
- **Curated HDI seed** (`hdi-gnn/src/data/hdi_seed.csv`) — 55 cited positive pairs, one PMID each.
- **IMPPAT phytocompound SDF** — external, **not redistributed**; see `data/external/README.md`.
- **Trained model bundle** (`cyp_models.pkl`, ~333 MB) — **not in git**; regenerate or
  fetch from the archival release; see `models/README.md`.

Full provenance, checksums, and a data dictionary are in `metadata/`.

## Usage (verified commands, run from each project's `src/`)

CYP-Atlas:
```bash
cd cyp-atlas/src
python fetch_cyp.py --demo            # offline synthetic smoke test (no network)
python fetch_cyp.py                   # real CYP IC50 data from ChEMBL -> data/cyp_labels.csv
python train_cyp.py                   # per-enzyme models + CV AUROC -> models/, results/metrics.csv
python atlas.py ../../imppat_phytocompounds.sdf   # predicted atlas -> data/cyp_atlas.csv
python risk.py ../../hdi-gnn/src/data/hdi_seed.csv  # mechanistic HDI risk, validated
python benchmark.py                   # RF vs LogReg vs HistGB, calibration/Brier
python domain.py                      # applicability domain -> data/cyp_atlas_domain.csv
python summarize.py                   # atlas prevalence + risk CI/perm-p -> figures
python revision_analysis.py           # reviewer-facing revision analyses -> results/revision/
```

HDI-GNN:
```bash
cd hdi-gnn/src
python train.py                       # smoke test on bundled sample (val_auc ~0.5 by design)
python build_dataset.py 2             # seed positives + 2:1 assumed-negatives -> data/hdi.csv
python train.py data/hdi.csv          # train on the curated seed
python analyze.py                     # baselines vs GNN, CV + cold-herb -> results/
python screen.py ../imppat_phytocompounds.sdf 200   # score IMPPAT x drug pairs
```

## Expected outputs

Frozen example outputs are included under each `src/results/` (metrics, tables,
figures). Key released numbers: per-enzyme CYP AUROC 0.86–0.91 (random split),
0.81–0.86 (scaffold split) — `cyp-atlas/src/results/metrics.csv`; mechanistic HDI
risk AUROC 0.591 (95% CI 0.482–0.698) on 55 curated pairs —
`cyp-atlas/src/results/risk_summary.csv`. HDI-GNN metrics are seed-scale and
honestly weak until the benchmark is grown — `hdi-gnn/src/results/metrics.csv`.

## Reproduction

```bash
git clone <repository>
cd <repository>
git checkout v1.0.0
# then install per project and run the commands above
```

See `docs/REPRODUCIBILITY.md` for the stage-by-stage input→output chain and
`metadata/REPRODUCIBILITY_MANIFEST.json` for seeds and entry points.

## Reproducibility ceilings (honest)

Labels are ChEMBL IC50 at pChEMBL ≥ 5; CYP **induction** (e.g. St John's wort) is
not modelled and is stated as a limitation; atlas values are predictions — spot-validate
top hits. The HDI-GNN dataset is a small seed to extend, not a final model.

## Version

v1.0.0 — see `CHANGELOG.md`. Licensed MIT (code); third-party data keeps its own
terms (see `LICENSE` and `metadata/SOURCE_PROVENANCE.md`).


## v1.1.0 reproducibility quick-start

```bash
cd cyp-atlas/src
# option A: regenerate everything (needs internet for ChEMBL and the IMPPAT SDF)
bash run_revision.sh

# option B: validate on frozen on-disk data (no network)
python validate_available.py --seed ../../hdi-gnn/src/data/hdi_seed_expanded.csv
```

The frozen inputs and outputs that back the headline numbers are under
`data/frozen/` with SHA-256 checksums in `metadata/CHECKSUMS.sha256`.
See `metadata/DATA_MANIFEST.csv` for a per-file summary.
