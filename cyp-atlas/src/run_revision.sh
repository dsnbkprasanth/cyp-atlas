#!/usr/bin/env bash
# One command to regenerate every reviewer-facing number.
# Run from cyp-atlas/src.
#
# Requires internet for fetch_cyp_v2.py; skip (A) sensitivity if offline.
set -euo pipefail

python train_cyp.py                                  # original CYP models (10 uM baseline)
python atlas.py ../imppat_phytocompounds.sdf         # original full predicted atlas
python domain_v2.py                                  # AD filter + in-domain atlas + sweep
python risk.py ../../hdi-gnn/src/data/hdi_seed_expanded.csv   # expanded HDI validation

# Reviewer-facing re-runs: threshold sensitivity + negative-sample robustness +
# distant-only subset + route-weighted scoring + P-gp inhibitor-only.
#
# For threshold sensitivity (A) and P-gp inhibitor-only (E), labels at
# other thresholds must exist. Uncomment the block if internet is available:
#
# for T in 1 5 10 30; do
#   python fetch_cyp_v2.py --threshold-um $T
# done

python revision_v2.py \
    --seed ../../hdi-gnn/src/data/hdi_seed_expanded.csv \
    --thresholds 1 5 10 30

echo
echo "All reviewer-facing artefacts live in results/revision/."
echo "Primary atlas (reviewer-grade): data/cyp_atlas_indomain.csv"
