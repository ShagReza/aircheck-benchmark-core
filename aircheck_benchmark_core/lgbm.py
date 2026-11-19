"""LightGBM on embeddings: the same for every model, since only the embeddings differ.

Each fork's `embed_lgbm` method embeds its molecules and calls `train_and_score` here, so the LightGBM
settings and the fold handling are shared instead of copied into every fork. Moved from the forks'
`MMELON_embedding_lgbm_cv.ipynb` / `MAMMAL_embedding_lgbm_cv.ipynb`, where it gave the same numbers.

Needs lightgbm, which the model environments have (`pip install aircheck-benchmark-core[lgbm]` otherwise).
Nothing else in this package imports it, so core works without lightgbm installed.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np

PARAMS = {
    "default": {},
    "suggested": dict(scale_pos_weight=8, n_estimators=3000, learning_rate=0.02, num_leaves=31,
                      min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                      reg_alpha=0.5, reg_lambda=5.0),
}


def params_for(labels, choice="suggested", balance_classes=True) -> dict:
    """The named parameters, with scale_pos_weight set from these labels when balancing the classes."""
    if choice not in PARAMS:
        raise ValueError(f"lgbm_params must be one of {list(PARAMS)}, not {choice!r}")
    params = dict(PARAMS[choice])
    if balance_classes:
        labels = np.asarray(labels).astype(int)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if not (n_pos and n_neg):
            raise ValueError("training rows need both labels to balance the classes")
        params["scale_pos_weight"] = round(n_neg / n_pos, 4)
    return params


def fit(X, y, choice="suggested", balance_classes=True, seed=42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **params_for(y, choice, balance_classes))
    model.fit(X, np.asarray(y).astype(int))
    return model


def train_and_score(X_train, valid_train, labels, folds, X_test, valid_test, models_dir,
                    choice="suggested", balance_classes=True, seed=42) -> dict:
    """One model per fold plus one on all training rows, and the scores they give.

    X_train holds only the molecules that could be embedded; `valid_train` says which training rows those
    are (the same for X_test and `valid_test`). Fold indices are rows of the folds CSV, as in `folds.py`.

    Returns {"fold<i>": scores of that fold's test rows, "test_ensemble": the fold models averaged on the
    test file, "test": the model trained on all rows}, with NaN where a molecule has no embedding.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    valid_train, valid_test = np.asarray(valid_train, dtype=bool), np.asarray(valid_test, dtype=bool)
    labels = np.asarray(labels).astype(int)
    position = np.full(len(valid_train), -1)  # training row -> row of X_train
    position[valid_train] = np.arange(valid_train.sum())

    scores, fold_test_scores = {}, []
    for i, (train_idx, test_idx) in enumerate(folds):
        kept = train_idx[valid_train[train_idx]]  # rows without an embedding are left out of training
        model = fit(X_train[position[kept]], labels[kept], choice, balance_classes, seed)
        model.booster_.save_model(models_dir / f"fold_{i}.txt")
        fold_scores = np.full(len(test_idx), np.nan)
        has = valid_train[test_idx]
        fold_scores[has] = model.predict_proba(X_train[position[test_idx[has]]])[:, 1]
        scores[f"fold{i}"] = fold_scores
        fold_test_scores.append(model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([]))

    all_rows = np.where(valid_train)[0]
    full = fit(X_train[position[all_rows]], labels[all_rows], choice, balance_classes, seed)
    full.booster_.save_model(models_dir / "full.txt")
    for split, values in (("test_ensemble", np.mean(fold_test_scores, axis=0)),
                          ("test", full.predict_proba(X_test)[:, 1] if len(X_test) else np.array([]))):
        scores[split] = np.full(len(valid_test), np.nan)
        scores[split][valid_test] = values
    return scores
