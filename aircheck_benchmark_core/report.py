"""Benchmark report: one Markdown report and one CSV for any number of runs (methods and models).

Every run (one method of one model on one dataset) writes a folder with two files:
  run.json          {"model": "mammal", "method": "finetune",          required
                     "name": "MAMMAL fine-tuned",                       display name (default "model / method")
                     "settings": {"epochs": 10, ...},                   shown in Setup
                     "report_extras": {"selected epoch": 5},            extra columns in the CV table
                     "hardware": "Tesla T4", "fork_commit": "abc1234", "core_version": "0.1.0"}   all optional
  predictions.csv   split, LABEL, score, row (see metrics.py); or predictions.parquet

`make_report` finds every run.json under a folder, so the same call reports on one model's methods
(results/<dataset>/<model>/) or on every model (results/<dataset>/).
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .folds import read_columns
from .metrics import HIT_KS, read_predictions, score_predictions

TABLE_METRICS = [*(f"hit@{k}" for k in HIT_KS), "rocauc", "pr_auc"]  # the columns shown in the report
HEADERS = {"rocauc": "AUROC", "pr_auc": "PR AUC"}
DASH = "—"


def find_runs(results_dir) -> list:
    """Every run under `results_dir`: its run.json contents plus the folder."""
    runs = []
    for path in sorted(Path(results_dir).expanduser().rglob("run.json")):
        info = json.loads(path.read_text())
        missing = [k for k in ("model", "method") if k not in info]
        if missing:
            raise ValueError(f"{path} has no {missing}")
        preds = [p for p in (path.parent / "predictions.csv", path.parent / "predictions.parquet") if p.exists()]
        if not preds:
            raise ValueError(f"{path.parent} has a run.json but no predictions.csv or predictions.parquet")
        runs.append({**info, "name": info.get("name", f"{info['model']} / {info['method']}"),
                     "folder": path.parent, "predictions": preds[0]})
    if not runs:
        raise ValueError(f"No run.json found under {results_dir}")
    return runs


def _md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value, metric) -> str:
    if value is None or pd.isna(value):
        return DASH
    return f"{value:.0f}" if metric.startswith("hit@") else f"{value:.3f}"


def _fmt_cv(mean, std, metric) -> str:
    if mean is None or pd.isna(mean):
        return DASH
    digits = 1 if metric.startswith("hit@") else 3
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def _data_section(train_path, test_path, folds_npz, smiles_col, label_col) -> str:
    rows, smiles = [], {}
    for name, path in (("Train", train_path), ("Test", test_path)):
        if path is None:
            continue
        df = read_columns(path, [smiles_col, label_col])
        smiles[name] = set(df[smiles_col].dropna())
        rows.append((name, f"`{Path(path).name}`", f"{len(df):,}", f"{int(df[label_col].sum()):,}",
                     f"{100 * df[label_col].mean():.2f}%"))
    bullets = []
    if folds_npz is not None:
        info_path = Path(str(folds_npz).replace("_folds.npz", "_folds.json"))
        if info_path.exists():
            info = json.loads(info_path.read_text())
            settings = ", ".join(f"{k}={v}" for k, v in info["settings"].items())
            test_pos = ", ".join(str(f["test_pos"]) for f in info["folds"])
            bullets.append(f"- **Folds:** {info['method']}, {info['n_folds']} fold(s) ({settings}); "
                           f"test actives per fold: {test_pos}. File: `{Path(folds_npz).name}`.")
        else:
            bullets.append(f"- **Folds:** `{Path(folds_npz).name}`.")
    if len(smiles) == 2:
        bullets.append(f"- **Molecules in both train and test:** {len(smiles['Train'] & smiles['Test']):,}.")
    parts = [_md_table(["", "File", "Molecules", "Actives", "Active rate"], rows)] if rows else []
    return "\n\n".join(parts + (["\n".join(bullets)] if bullets else []))


def make_report(results_dir, out=None, train_path=None, test_path=None, folds_npz=None, title=None,
                smiles_col="SMILES", label_col="LABEL") -> dict:
    """Score every run under `results_dir` and write report.md and summary.csv. Returns their paths.

    With `folds_npz`, every run's fold predictions are checked to be exactly that file's fold test rows.
    """
    results_dir = Path(results_dir).expanduser()
    out = Path(out).expanduser() if out else results_dir / "report.md"
    runs = find_runs(results_dir)

    tables = []
    for run in runs:
        table = score_predictions(read_predictions(run["predictions"]), folds_npz=folds_npz)
        tables.append(table.assign(name=run["name"], model=run["model"], method=run["method"]))
    summary = pd.concat(tables, ignore_index=True)
    summary = summary[["name", "model", "method", *[c for c in summary.columns if c not in ("name", "model", "method")]]]

    # Setup: one row per run
    setup_rows = [(r["name"], f"`{r['model']}`", f"`{r['method']}`",
                   ", ".join(f"{k}={v}" for k, v in r.get("settings", {}).items()) or DASH,
                   r.get("hardware", DASH), f"`{r['fork_commit']}`" if r.get("fork_commit") else DASH,
                   r.get("core_version", DASH)) for r in runs]
    setup_md = _md_table(["Run", "Model", "Method", "Settings", "Hardware", "Fork commit", "Core version"], setup_rows)

    # Cross-validation: mean ± std over folds, for runs with fold predictions
    extras = list(dict.fromkeys(k for r in runs for k in r.get("report_extras", {})))
    cv_rows = []
    for run, table in zip(runs, tables):
        stats = table.set_index("split")
        if "cv_mean" not in stats.index:
            continue
        cv_rows.append((run["name"], *[run.get("report_extras", {}).get(k, DASH) for k in extras],
                        *[_fmt_cv(stats.at["cv_mean", m], stats.at["cv_std", m], m) for m in TABLE_METRICS]))
    cv_md = (_md_table(["Run", *extras, *[HEADERS.get(m, m) for m in TABLE_METRICS]], cv_rows) +
             "\n\nMean ± std over folds (no std with one fold).") if cv_rows else "_No runs with fold predictions._"

    # Test: every non-fold split (e.g. test, test_ensemble)
    test_rows = []
    for run, table in zip(runs, tables):
        for _, row in table[~table["split"].str.fullmatch(r"fold\d+|cv_mean|cv_std")].iterrows():
            test_rows.append((run["name"], f"`{row['split']}`", *[_fmt(row[m], m) for m in TABLE_METRICS]))
    test_md = (_md_table(["Run", "Split", *[HEADERS.get(m, m) for m in TABLE_METRICS]], test_rows)
               if test_rows else "_No runs with test predictions._")

    sections = [f"# Report: {title or results_dir.name}",
                f"*Generated by `aircheck-bench report` (core {__version__}) on {datetime.now():%Y-%m-%d %H:%M} "
                f"from `{results_dir}`: {len(runs)} run(s). Rerunning it regenerates this file.*"]
    data_md = _data_section(train_path, test_path, folds_npz, smiles_col, label_col)
    if data_md:
        sections += ["## Data", data_md]
    sections += ["## Setup", setup_md, "## Cross-validation results (train)", cv_md, "## Test results", test_md]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(sections) + "\n")
    csv_path = out.with_name(f"{out.stem}_summary.csv")  # every split and metric of every run
    summary.to_csv(csv_path, index=False)
    return {"report": out, "summary": csv_path}
