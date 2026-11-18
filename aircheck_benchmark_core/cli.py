"""Command line: `aircheck-bench make-folds | score | report ...` (add --help for the options)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .folds import METHODS, make_folds
from .metrics import HIT_KS, THRESHOLD, read_predictions, score_predictions
from .report import make_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aircheck-bench", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-folds", help="build cross-validation folds for a training file")
    p.add_argument("--train", required=True, help="training data (parquet or CSV)")
    p.add_argument("--out", required=True, help="folder for the CSV, .npz and .json")
    p.add_argument("--method", default="cluster", choices=sorted(METHODS))
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="LABEL")
    p.add_argument("--fingerprint-col", default="ECFP4", help="used by cluster folds only")
    p.add_argument("--nrows", type=int, help="use only the first N rows")
    # Method settings: left out -> the method's own default
    p.add_argument("--n-folds", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--n-clusters", type=int, help="cluster only")
    p.add_argument("--linkage", help="cluster only")
    p.add_argument("--test-pos-per-fold", type=int, help="cluster only")
    p.add_argument("--test-neg-fraction", type=float, help="cluster only")

    p = sub.add_parser("score", help="compute the metrics from a predictions file")
    p.add_argument("--predictions", required=True, help="predictions (CSV or parquet); see metrics.py for the columns")
    p.add_argument("--out", help="save the metrics table here (CSV)")
    p.add_argument("--folds", help="folds .npz: check that each fold's predictions are exactly its test rows")
    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="LABEL")
    p.add_argument("--score-col", default="score")
    p.add_argument("--row-col", default="row")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--hit-ks", type=int, nargs="+", default=list(HIT_KS))

    p = sub.add_parser("report", help="score every run under a results folder and write report.md + summary CSV")
    p.add_argument("--results", required=True, help="folder searched for run.json files (one model or all models)")
    p.add_argument("--out", help="report path (default: <results>/report.md); the CSV goes next to it")
    p.add_argument("--train", help="train file, for the Data section")
    p.add_argument("--test", help="test file, for the Data section")
    p.add_argument("--folds", help="folds .npz: check every run's fold predictions and describe the folds")
    p.add_argument("--title", help="report title (default: the results folder name)")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="LABEL")

    args = parser.parse_args(argv)
    if args.command == "report":
        paths = make_report(args.results, out=args.out, train_path=args.train, test_path=args.test,
                            folds_npz=args.folds, title=args.title, smiles_col=args.smiles_col,
                            label_col=args.label_col)
        for path in paths.values():
            print(f"Saved {path}")
    elif args.command == "score":
        table = score_predictions(read_predictions(args.predictions), split_col=args.split_col,
                                  label_col=args.label_col, score_col=args.score_col, hit_ks=args.hit_ks,
                                  threshold=args.threshold, folds_npz=args.folds, row_col=args.row_col)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            # Whole numbers (counts, hit@K) without decimals, the rest with 4
            print(table.to_string(index=False, float_format=lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.4f}"))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"Saved {args.out}")
    elif args.command == "make-folds":
        params = {k: getattr(args, k) for k in ("n_folds", "seed", "n_clusters", "linkage", "test_pos_per_fold",
                                                 "test_neg_fraction") if getattr(args, k) is not None}
        paths = make_folds(args.train, args.out, method=args.method, smiles_col=args.smiles_col,
                           label_col=args.label_col, fingerprint_col=args.fingerprint_col, nrows=args.nrows, **params)
        info = json.loads(paths["info"].read_text())
        for f in info["folds"]:
            print(f"fold {f['fold']}: train={f['train']:,} ({f['train_pos']} pos)  test={f['test']:,} ({f['test_pos']} pos)")
        for path in paths.values():
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
