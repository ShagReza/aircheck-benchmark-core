# aircheck-benchmark-core

Shared code for the AIRCHECK model benchmark, installed into every model's environment so all models use
the same cross-validation folds and the same metrics.

| Module | What it holds |
|---|---|
| `folds.py` | fold methods (`random`, `stratified`, `cluster`), `make_folds` (build + save), `load_folds` |
| `metrics.py` | `compute_metrics` (hit@200/100/50, AUROC, PR AUC, precision, recall, specificity, accuracy), `score_predictions` |
| `cli.py` | the `aircheck-bench` command: `make-folds`, `score` |

## Install

While developing (edits to this repo take effect without reinstalling):
```bash
pip install -e ~/repos/aircheck-benchmark-core
```
For benchmark runs, pin a tag (e.g. in a fork's `aircheck/requirements-aircheck.txt`):
```
aircheck-benchmark-core @ git+ssh://git@github.com/ShagReza/aircheck-benchmark-core.git@v0.1.0
```

## Make folds

```bash
aircheck-bench make-folds --train train.parquet --out folds/ --method cluster
aircheck-bench make-folds --train train.parquet --out folds/ --method stratified --n-folds 5
aircheck-bench make-folds --help   # all options
```
Writes, in `--out`:
```
<train name>_<method>_cv_data.csv    SMILES, LABEL, source_row: the rows the fold indices refer to
<train name>_<method>_cv_folds.npz   fold0_train, fold0_test, fold1_train, ... (0-based rows of the CSV)
<train name>_<method>_cv_folds.json  method, settings, input file checksum, core version, per-fold counts
```
`cluster` reads the `ECFP4` column (`--fingerprint-col`) and follows `cv_method: 'cluster'` in
`ml-pipeline-aircheck` (`_build_cluster_folds`), giving the same folds; its defaults are those of the
pipeline except `--n-folds` (5 here, 10 there). The CSV and `.npz` have the same layout as the files from
the forks' `cluster_cv_folds.ipynb`.

## Score predictions

Models write predictions only; the metrics are computed here, the same way for every model.
```bash
aircheck-bench score --predictions predictions.csv --out metrics.csv
aircheck-bench score --predictions predictions.csv --folds folds/train_cluster_cv_folds.npz   # also check the folds
```
Predictions file (CSV or parquet, one row per molecule):

| Column | |
|---|---|
| `split` | `fold0`, `fold1`, ... (CV fold test sets) or `test` (final test set). Optional: missing = all `test` |
| `LABEL` | 0/1 |
| `score` | higher = more likely a hit |
| `row` | optional: row in the folds CSV (CV) or the test file. Needed for `--folds` |

Other columns (SMILES, pred, rank, ...) are ignored, so the forks' current `predictions.csv` files can be
scored as they are. The output has one row per split, plus `cv_mean` and `cv_std` over the folds. With
`--folds`, it stops with an error unless each fold's predictions are exactly that fold's test rows.

## Use from Python

```python
from aircheck_benchmark_core.folds import load_folds
from aircheck_benchmark_core.metrics import compute_metrics

for train_idx, test_idx in load_folds("folds/train_cluster_cv_folds.npz"):
    ...
metrics = compute_metrics(labels, scores)
```

## Add a metric

Write a function in `metrics.py` that takes `(labels, scores, **settings)` and returns `{name: value}`, then
add it to `METRIC_GROUPS`. It appears in every `compute_metrics` result and in `aircheck-bench score`.

## Add a fold method

Write a function in `folds.py` that takes `labels` (plus its settings, with defaults) and returns a list of
`Fold(train, test)`, then add it to `METHODS`. It becomes available as `--method <name>`.
