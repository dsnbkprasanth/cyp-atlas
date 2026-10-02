# HDI validation — summary (on-disk data only)

Routes used: **['CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4', 'PGP']** (PGP route included where applicable).

## Overall (every strategy x rule)

| strategy | rule | n+ | n- | AUROC | 95% CI | PR-AUC | p(pos>neg) |
|---|---|---|---|---|---|---|---|
| random | max | 48 | 45 | 0.539 | 0.419-0.660 | 0.565 | 0.2579 |
| random | mean | 48 | 45 | 0.508 | 0.389-0.625 | 0.52 | 0.4510 |
| scaffold | max | 48 | 45 | 0.539 | 0.423-0.659 | 0.565 | 0.2579 |
| scaffold | mean | 48 | 45 | 0.508 | 0.393-0.627 | 0.52 | 0.4510 |
| dissimilar | max | 48 | 45 | 0.539 | 0.414-0.657 | 0.565 | 0.2579 |
| dissimilar | mean | 48 | 45 | 0.508 | 0.390-0.624 | 0.52 | 0.4510 |

## Headline

**random negatives, max rule: AUROC 0.539 (95% CI 0.419-0.660), PR-AUC 0.565, p=0.2579.**

## Per-isoform subset AUROC (random negatives, isoform-only score)

| isoform | n | pos | AUROC | 95% CI |
|---|---|---|---|---|
| CYP1A2 | 16 | 14 | 0.25 | 0.000-0.667 |
| CYP2C9 | 16 | 14 | 0.536 | 0.000-1.000 |
| CYP2D6 | 10 | 7 | 0.238 | 0.000-0.625 |
| CYP3A4 | 86 | 47 | 0.48 | 0.357-0.606 |
| PGP | 40 | 11 | 0.563 | 0.355-0.764 |

## Per-drug AUROC (random, mean rule)

See validate_per_drug.csv. Interpretation: inside each drug the model ranks
constituents by predicted risk; mean of per-drug AUROCs across drugs is the
clinically meaningful retrieval metric (removes drug-level calibration noise).
