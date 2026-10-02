# Reproducibility test — v1.1.0

Not yet executed on a clean machine (user action — see release procedure).

Suggested test:

```bash
git clone <repository> clean-room && cd clean-room && git checkout v1.1.0
conda create -n cyp-atlas python=3.11 -y && conda activate cyp-atlas
pip install -r cyp-atlas/requirements.txt
cd cyp-atlas/src
python validate_available.py --seed ../../hdi-gnn/src/data/hdi_seed_expanded.csv
diff -q results/revision/validate_overall.csv ../../data/frozen/validate_overall.csv
```

Expected: `diff` prints nothing (numerical reproducibility of frozen outputs
against the on-disk pipeline run with seed 0). Tree models (RandomForest) and
splits use `random_state=0` throughout.
