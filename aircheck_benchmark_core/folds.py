"""Cross-validation folds: made once per dataset, saved to files, and used by every model.

Methods (add a new one by writing a function that returns a list of `Fold` and adding it to `METHODS`):
  random      KFold on shuffled rows
  stratified  StratifiedKFold: each fold keeps the label ratio
  cluster     cluster-held-out folds, as `cv_method: 'cluster'` in ml-pipeline-aircheck
              (`_build_cluster_folds` in model_utils.py) and the forks' `cluster_cv_folds.ipynb`

Output of `make_folds` (in `out_dir`):
  <train name>_<method>_cv_data.csv    SMILES, LABEL, source_row: the rows the fold indices refer to
  <train name>_<method>_cv_folds.npz   fold0_train, fold0_test, fold1_train, ... (0-based rows of the CSV)
  <train name>_<method>_cv_folds.json  method, settings, input file checksum, core version, per-fold counts
"""

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold, StratifiedKFold

from . import __version__


@dataclass
class Fold:
    train: np.ndarray  # 0-based row indices
    test: np.ndarray
    info: dict = field(default_factory=dict)  # method-specific details, saved in the JSON


def random_folds(labels, n_folds=5, seed=42) -> list:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [Fold(tr, te) for tr, te in kf.split(np.zeros(len(labels)))]


def stratified_folds(labels, n_folds=5, seed=42) -> list:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [Fold(tr, te) for tr, te in skf.split(np.zeros(len(labels)), labels)]


def tanimoto_distance(X) -> np.ndarray:
    """Pairwise Tanimoto distance between binary fingerprints (rows of X)."""
    X = (np.asarray(X) > 0).astype(np.int32)  # int32 counts, so large fingerprints cannot overflow
    intersection = X @ X.T
    bit_counts = X.sum(axis=1, keepdims=True)
    union = bit_counts + bit_counts.T - intersection
    sim = np.divide(intersection, union, out=np.zeros(intersection.shape, dtype=np.float32),
                    where=union != 0).astype(np.float32)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster_folds(labels, fingerprints, n_folds=5, n_clusters=20, linkage="average", test_pos_per_fold=100,
                  test_neg_fraction=0.25, seed=42) -> list:
    """Cluster-held-out folds.

    1. Cluster only the positives: Tanimoto distance, agglomerative clustering into `n_clusters` groups.
    2. Each fold's test positives come from whole clusters (largest unused cluster first, topped up with the
       smallest ones until there are `test_pos_per_fold` positives). Every positive in those clusters is
       removed from that fold's training set, so each fold is tested on chemistry it has not trained on.
    3. Test negatives are a random `test_neg_fraction` of all negatives, drawn fresh for each fold (so they
       can repeat across folds); the remaining negatives are used for training.
    4. Up to `n_folds` folds are built, stopping early when unused clusters run out.

    `fingerprints` holds one row per positive only (in the order of the positives in `labels`).
    Defaults are those of ml-pipeline-aircheck, except n_folds (10 there).
    """
    from sklearn.cluster import AgglomerativeClustering

    labels = np.asarray(labels).astype(int)
    pos_idx, neg_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
    if len(pos_idx) < test_pos_per_fold:
        raise ValueError(f"Only {len(pos_idx)} positives; need at least test_pos_per_fold = {test_pos_per_fold}")

    cluster_labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=linkage
    ).fit_predict(tanimoto_distance(fingerprints))
    cluster_to_pos = {cid: pos_idx[cluster_labels == cid] for cid in np.unique(cluster_labels)}

    rng = np.random.default_rng(seed)
    unused = set(cluster_to_pos)
    groups = []  # (test positives, excluded positives, cluster ids) per fold
    while len(groups) < n_folds:
        available = sorted(((c, cluster_to_pos[c]) for c in unused), key=lambda x: len(x[1]), reverse=True)
        if not available or sum(len(m) for _, m in available) < test_pos_per_fold:
            break
        # Start from the largest unused cluster; add the smallest others until there are enough positives
        seed_cid, seed_members = available[0]
        selected, members = [seed_cid], list(seed_members)
        if len(members) < test_pos_per_fold:
            for cid, m in sorted(((c, cluster_to_pos[c]) for c in unused if c != seed_cid), key=lambda x: len(x[1])):
                if len(members) >= test_pos_per_fold:
                    break
                selected.append(cid)
                members.extend(m)
        if len(members) < test_pos_per_fold:
            break
        members = np.array(members)
        test_pos = np.sort(rng.choice(members, size=test_pos_per_fold, replace=False))
        groups.append((test_pos, np.sort(members), selected))
        # One at a time, as in ml-pipeline-aircheck: `unused -= set(selected)` leaves the set in another
        # iteration order, which picks a different cluster when two have the same size
        for cid in selected:
            unused.discard(cid)
    if not groups:
        raise ValueError("No folds could be built from the clusters; try a smaller test_pos_per_fold")

    rng = np.random.default_rng(seed)  # ml-pipeline-aircheck uses a fresh RNG with the same seed for the negatives
    n_test_neg = int(round(len(neg_idx) * test_neg_fraction))
    folds = []
    for test_pos, excluded_pos, selected in groups:
        test_neg = rng.choice(neg_idx, size=n_test_neg, replace=False)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_neg] = False
        train_mask[excluded_pos] = False
        folds.append(Fold(np.where(train_mask)[0], np.sort(np.concatenate([test_pos, test_neg])),
                          info={"clusters": [int(c) for c in selected],
                                "held_out_pos": len(excluded_pos) - len(test_pos)}))
    return folds


METHODS = {"random": random_folds, "stratified": stratified_folds, "cluster": cluster_folds}
NEEDS_FINGERPRINTS = {"cluster"}


def read_columns(path, columns, nrows=None) -> pd.DataFrame:
    """Read only the given columns from a parquet or CSV file."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        available = pq.ParquetFile(path).schema_arrow.names
        missing = [c for c in columns if c not in available]
        if missing:
            raise ValueError(f"{path.name} has no column(s) {missing}; available: {available}")
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, usecols=columns)
    return df.head(nrows) if nrows is not None else df


def fingerprint_bits(values) -> np.ndarray:
    """Fingerprints stored as arrays or as comma-separated strings -> 0/1 uint8 matrix."""
    rows = [np.fromstring(v, sep=",", dtype=np.float32) if isinstance(v, str) else np.asarray(v) for v in values]
    return (np.vstack(rows) > 0).astype(np.uint8)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_folds(train_path, out_dir, method="cluster", smiles_col="SMILES", label_col="LABEL",
               fingerprint_col="ECFP4", nrows=None, **params) -> dict:
    """Build folds for one training file and save the CSV, .npz and .json. Returns their paths.

    `params` go to the method function (e.g. n_folds, seed, n_clusters); missing ones take its defaults.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'; choose from {sorted(METHODS)}")
    fn = METHODS[method]
    accepted = inspect.signature(fn).parameters
    unknown = [p for p in params if p not in accepted]
    if unknown:
        raise ValueError(f"Method '{method}' does not take {unknown}")
    settings = {name: p.default for name, p in accepted.items() if p.default is not inspect.Parameter.empty}
    settings.update(params)

    train_path, out_dir = Path(train_path).expanduser(), Path(out_dir).expanduser()
    needs_fp = method in NEEDS_FINGERPRINTS
    df = read_columns(train_path, [smiles_col, label_col] + ([fingerprint_col] if needs_fp else []), nrows)
    df["source_row"] = np.arange(len(df))  # row in the original file
    # Drop rows without a label (and, for cluster folds, without a fingerprint), then renumber
    df = df.dropna(subset=[label_col] + ([fingerprint_col] if needs_fp else [])).reset_index(drop=True)

    labels = df[label_col].astype(int).to_numpy()
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError(f"Labels must be 0/1, found {np.unique(labels)}")
    if needs_fp:
        pos = labels == 1  # only the positives are clustered
        folds = fn(labels, fingerprint_bits(df.loc[pos, fingerprint_col]), **params)
        df = df.drop(columns=[fingerprint_col])
    else:
        folds = fn(labels, **params)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{train_path.stem}_{method}_cv"
    paths = {"data": out_dir / f"{stem}_data.csv", "folds": out_dir / f"{stem}_folds.npz",
             "info": out_dir / f"{stem}_folds.json"}

    df[[smiles_col, label_col, "source_row"]].to_csv(paths["data"], index=False)
    np.savez(paths["folds"], **{f"fold{i}_{part}": idx for i, f in enumerate(folds)
                               for part, idx in (("train", f.train), ("test", f.test))})
    info = {
        "core_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "settings": settings,
        "train_file": str(train_path.resolve()),
        "train_sha256": file_sha256(train_path),
        "columns": {"smiles": smiles_col, "label": label_col, **({"fingerprint": fingerprint_col} if needs_fp else {})},
        "nrows": nrows,
        "n_rows": len(df),
        "n_pos": int(labels.sum()),
        "n_folds": len(folds),
        "folds": [{"fold": i, "train": len(f.train), "test": len(f.test), "train_pos": int(labels[f.train].sum()),
                   "test_pos": int(labels[f.test].sum()), **f.info} for i, f in enumerate(folds)],
    }
    paths["info"].write_text(json.dumps(info, indent=2) + "\n")
    return paths


def load_folds(npz_path) -> list:
    """Read a folds .npz back as a list of (train indices, test indices)."""
    data = np.load(npz_path)
    n = len([k for k in data.files if k.endswith("_train")])
    return [(data[f"fold{i}_train"], data[f"fold{i}_test"]) for i in range(n)]
