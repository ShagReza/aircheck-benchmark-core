"""Benchmark metrics: the same numbers for every model.

`compute_metrics` moved from the AIRCHECK notebooks of the MMELON and MAMMAL forks, where it had been copied
into 14 notebooks in 3 slightly different versions. Add a metric by writing a function that takes
(labels, scores, settings) and returns {name: value}, then adding it to `METRIC_GROUPS`.

Predictions file read by `score_predictions` / `aircheck-bench score` (CSV or parquet, one row per molecule):
  split   fold0, fold1, ... (CV fold test sets) or test (final test set); optional, missing = all "test"
  LABEL   0/1
  score   higher = more likely a hit
  row     optional: row in the folds CSV (CV) or test file; needed to check the folds with `folds_npz`
Other columns (SMILES, pred, rank, ...) are ignored.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HIT_KS = (200, 100, 50)  # hit@K = number of label-1 molecules among the K highest-scored
THRESHOLD = 0.5  # score above this -> predicted 1 (only for precision, recall, specificity, accuracy)


def hit_metrics(labels, scores, hit_ks=HIT_KS, **_) -> dict:
    top = np.argsort(-scores, kind="stable")
    return {f"hit@{k}": int(labels[top[:k]].sum()) for k in hit_ks}


def ranking_metrics(labels, scores, **_) -> dict:
    both_classes = len(set(labels.tolist())) == 2
    return {
        "rocauc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else float("nan"),
    }


def threshold_metrics(labels, scores, threshold=THRESHOLD, **_) -> dict:
    pred = scores > threshold
    tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
    tn, fn = int((~pred & (labels == 0)).sum()), int((~pred & (labels == 1)).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(labels) if len(labels) else float("nan"),
    }


METRIC_GROUPS = [hit_metrics, ranking_metrics, threshold_metrics]  # output order = this order


def compute_metrics(labels, scores, hit_ks=HIT_KS, threshold=THRESHOLD) -> dict:
    """All metrics for one set of 0/1 labels and scores."""
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores, dtype=float)
    metrics = {}
    for group in METRIC_GROUPS:
        metrics.update(group(labels, scores, hit_ks=hit_ks, threshold=threshold))
    return metrics


def _split_order(split):
    m = re.fullmatch(r"fold(\d+)", str(split))
    return (0, int(m.group(1)), "") if m else (1, 0, str(split))


def check_folds(df, folds_npz, split_col="split", row_col="row"):
    """Raise if the fold predictions do not cover exactly each fold's test rows of the folds .npz."""
    from .folds import load_folds

    if row_col not in df:
        raise ValueError(f"Checking against folds needs a '{row_col}' column")
    folds = load_folds(folds_npz)
    found = sorted({s for s in df[split_col] if str(s).startswith("fold")}, key=_split_order)
    expected = [f"fold{i}" for i in range(len(folds))]
    if found != expected:
        raise ValueError(f"Fold splits {found} do not match the {len(folds)} folds in {folds_npz}")
    for i, (_, test_idx) in enumerate(folds):
        rows = df.loc[df[split_col] == f"fold{i}", row_col].to_numpy()
        if len(rows) != len(test_idx) or not np.array_equal(np.sort(rows), np.sort(test_idx)):
            raise ValueError(f"fold{i}: the predicted rows are not that fold's test rows in {folds_npz}")


def score_predictions(df, split_col="split", label_col="LABEL", score_col="score", hit_ks=HIT_KS,
                      threshold=THRESHOLD, folds_npz=None, row_col="row") -> pd.DataFrame:
    """Metrics per split, plus cv_mean and cv_std over the fold splits (if there are any)."""
    missing = [c for c in (label_col, score_col) if c not in df]
    if missing:
        raise ValueError(f"Predictions have no column(s) {missing}; available: {list(df.columns)}")
    if split_col not in df:
        df = df.assign(**{split_col: "test"})
    df = df.assign(**{split_col: df[split_col].astype(str)})
    if folds_npz is not None and df[split_col].str.fullmatch(r"fold\d+").any():
        check_folds(df, folds_npz, split_col, row_col)

    rows = []
    for split in sorted(df[split_col].unique(), key=_split_order):
        g = df[df[split_col] == split]
        rows.append({"split": split, "n": len(g), "n_pos": int(g[label_col].astype(int).sum()),
                     **compute_metrics(g[label_col], g[score_col], hit_ks, threshold)})
    table = pd.DataFrame(rows)

    fold_rows = table[table["split"].str.fullmatch(r"fold\d+")].drop(columns="split")
    if len(fold_rows):
        summary = pd.DataFrame([{"split": "cv_mean", **fold_rows.mean()},
                                {"split": "cv_std", **fold_rows.std()}])  # std over folds, ddof=1 as in the forks
        table = pd.concat([table, summary], ignore_index=True)
    return table


def read_predictions(path) -> pd.DataFrame:
    path = str(path)
    return pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)
