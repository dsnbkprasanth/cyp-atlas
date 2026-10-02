# Reproducibility chain

Each stage lists its input(s) → output(s). Seeds are fixed (`random_state=0`,
`np.random.default_rng(0)`) for all tree models and splits; see
`metadata/REPRODUCIBILITY_MANIFEST.json`.

## CYP-Atlas

```
ChEMBL IC50 bioactivity  (external, network)
    │  fetch_cyp.py
    ▼
cyp_labels.csv  (14,637 compounds × 6 enzyme/transporter labels)
    │  train_cyp.py   (per-enzyme RandomForest, n_estimators=300, seed=0; Morgan/RDKit FP)
    ▼
cyp_models.pkl (~333 MB, excluded) + results/metrics.csv  (random + scaffold AUROC)
    │  atlas.py  ← imppat_phytocompounds.sdf (external, excluded)
    ▼
cyp_atlas.csv  (17,963 phytochemicals × 6 predicted probabilities)   [RELEASE RESOURCE]
    │  domain.py   (Tanimoto to training set)
    ▼
cyp_atlas_domain.csv  (+ max_train_tanimoto, in_domain)
    │  risk.py  ← hdi_seed.csv + drug_cyp.csv   (multi-CYP/P-gp mechanistic score)
    ▼
results/risk_validation.csv   → summarize.py → risk_summary.csv (AUROC, 95% CI, perm-p), figures
    │  benchmark.py / revision_analysis.py
    ▼
results/benchmark.csv, results/revision/*  (model comparison, calibration, AD reliability)
```

## HDI-GNN

```
hdi_seed.csv  (55 cited positive HDI pairs, PMID each)
    │  build_dataset.py N   (adds N:1 assumed-negatives)
    ▼
hdi.csv  (156 rows: positives + assumed-negatives)
    │  train.py  (twin-GCN + MLP head)          ┐
    │  analyze.py (RF/HistGB/LogReg baselines vs GNN, 5-fold + leave-one-herb-out)
    ▼                                            ┘
results/metrics.csv, tables/ (dataset stats, model comparison, provenance=Table S1), figures/
    │  screen.py ← imppat_phytocompounds.sdf (external, excluded)
    ▼
results/screen_top_hits.csv  (predicted phytochemical × drug pairs; PREDICTIONS — confirm before any claim)
```

## Verification notes

- Tree-model results are deterministic given the fixed seed and matching library versions.
- The HDI-GNN torch model does not set a global torch seed; its metrics are noisy at seed scale by design.
- To verify frozen data integrity: `sha256sum -c metadata/CHECKSUMS.sha256` from the repo root.
- No clean-room reproduction has been executed as part of packaging; the owner should
  run one in a fresh environment before tagging the release (see the checklist).
