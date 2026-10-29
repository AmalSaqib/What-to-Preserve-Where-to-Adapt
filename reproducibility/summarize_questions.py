#!/usr/bin/env python3
"""Rebuild the result tables for the paper's three research questions.

Q1  Adaptation depth: forgetting and current-task Dice as the trainable region
    expands, including the approximate matched-RMS and MRI-to-MRI comparisons.
Q2  Learning-rate allocation: shared versus scale-aware rates at fixed depth.
Q3  Smaller learning rate: uniformly reduced rates versus scale-aware rates.

Rows and their experiments are defined in ``experiments.json`` under
``questions``. For every experiment the script reads the final checkpoint and
the held-out evaluation outputs, then reports per-seed values and, for rows
with several seeds, the mean and sample standard deviation (ddof=1). Standard
deviations describe training-seed variability, not held-out-case variability.

Checkpoints are loaded with PyTorch pickle deserialization; only use
checkpoints that you produced or otherwise trust.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproducibility.bootstrap_metrics import load_cases, macro_dice  # noqa: E402
from reproducibility.checkpoint_displacement import displacement, load_body  # noqa: E402
from reproducibility.run_experiment import EVAL_TRAINERS, load_manifest  # noqa: E402


REQUIRED_ENV = ("RESULTS_FOLDER", "EVALUATION_FOLDER", "PAPER_INIT_CHECKPOINT", "PAPER_BASELINE_UMD_EVAL")


def run_root(base: str, exp: dict, manifest: dict) -> Path:
    defaults = manifest["defaults"]
    joined = "_".join(manifest["datasets"][str(task)]["name"] for task in exp["tasks"])
    trainer = f"{EVAL_TRAINERS[exp['method']]}__{exp['plan']}"
    return Path(base) / "nnUNet_ext" / defaults["network"] / joined / joined / trainer / "Generic_UNet" / "SEQ"


def evaluation_file(exp: dict, manifest: dict, task: int) -> Path:
    name = manifest["datasets"][str(task)]["name"]
    fold = manifest["defaults"]["fold"]
    return (run_root(os.environ["EVALUATION_FOLDER"], exp, manifest)
            / f"head_{name}" / f"fold_{fold}" / name / "val_metrics_eval.json")


def checkpoint_file(exp: dict, manifest: dict) -> Path:
    defaults = manifest["defaults"]
    return (run_root(os.environ["RESULTS_FOLDER"], exp, manifest)
            / f"fold_{defaults['fold']}" / f"{defaults['checkpoint']}.model")


def outer_pair(blocks: list[str]) -> list[str]:
    """Return the outermost encoder/decoder pair of a nested region (e.g. E4, D4)."""
    encoders = sorted(int(b[1:]) for b in blocks if b.startswith("E"))
    decoders = sorted(int(b[1:]) for b in blocks if b.startswith("D"))
    pair = []
    if encoders:
        pair.append(f"E{encoders[0]}")
    if decoders:
        pair.append(f"D{decoders[0]}")
    return pair


def experiment_metrics(exp, manifest, initial, baseline_old_task):
    ckpt = checkpoint_file(exp, manifest)
    old_eval = evaluation_file(exp, manifest, exp["tasks"][0])
    new_eval = evaluation_file(exp, manifest, exp["tasks"][-1])
    missing = [str(p) for p in (ckpt, old_eval, new_eval) if not p.exists()]
    if missing:
        print(f"warning: skipping {exp['id']}; missing {missing[0]}", file=sys.stderr)
        return None

    blocks = list(exp.get("blocks", []))
    final = load_body(ckpt)
    overall = displacement(initial, final, set(blocks))
    total_sq = overall["l2_delta"] ** 2
    shares = {}
    for block in blocks:
        block_sq = displacement(initial, final, {block})["l2_delta"] ** 2
        shares[block] = 100.0 * block_sq / total_sq if total_sq > 0 else 0.0

    return {
        "experiment": exp["id"],
        "seed": exp.get("run_seed", 12345),
        "base_lr": exp.get("base_lr", ""),
        "lr_allocation": "scale-aware" if exp.get("gradient_lr_control") else "shared",
        "trainable_parameters": overall["parameters"],
        "rms_delta": overall["rms_delta"],
        "relative_l2_percent": overall["relative_l2_percent"],
        "l2_delta": overall["l2_delta"],
        "old_task_forgetting": baseline_old_task - macro_dice(load_cases(old_eval)[1]),
        "new_task_dice": macro_dice(load_cases(new_eval)[1]),
        "outer_pair_share_percent": sum(shares.get(b, 0.0) for b in outer_pair(blocks)) if blocks else 0.0,
        "bottleneck_share_percent": shares.get("B", 0.0),
        "block_shares_percent": ";".join(f"{b}={v:.1f}" for b, v in shares.items()),
    }


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else float("nan"))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell(mean, sd, digits):
    return f"{mean:.{digits}f}" if np.isnan(sd) else f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", choices=("Q1", "Q2", "Q3"), action="append",
                        help="question(s) to summarize; default: all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/questions"))
    args = parser.parse_args()

    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing_env:
        raise SystemExit("Missing environment variables: " + ", ".join(missing_env)
                         + ". Source reproducibility/config.example.env first.")

    manifest = load_manifest()
    experiments = {exp["id"]: exp for exp in manifest["experiments"]}
    initial = load_body(Path(os.environ["PAPER_INIT_CHECKPOINT"]) / f"{manifest['defaults']['checkpoint']}.model")
    baseline_old_task = macro_dice(load_cases(Path(os.environ["PAPER_BASELINE_UMD_EVAL"]))[1])
    cache = {}

    for question in args.question or list(manifest["questions"]):
        spec = manifest["questions"][question]
        print(f"\n## {question}: {spec['title']}")
        for table, table_rows in spec["tables"].items():
            per_run, summary = [], []
            for row in table_rows:
                runs = []
                for experiment_id in row["experiments"]:
                    if experiment_id not in cache:
                        cache[experiment_id] = experiment_metrics(
                            experiments[experiment_id], manifest, initial, baseline_old_task)
                    if cache[experiment_id] is not None:
                        runs.append({"question": question, "table": table, "row": row["label"],
                                     **cache[experiment_id]})
                per_run.extend(runs)
                if not runs:
                    continue
                entry = {"question": question, "table": table, "row": row["label"], "n_seeds": len(runs),
                         "seeds": ";".join(str(r["seed"]) for r in runs)}
                for metric in ("old_task_forgetting", "new_task_dice", "rms_delta",
                               "outer_pair_share_percent", "bottleneck_share_percent"):
                    entry[f"{metric}_mean"], entry[f"{metric}_sd"] = mean_sd([r[metric] for r in runs])
                summary.append(entry)

            write_csv(args.output_dir / f"{question}_{table}_per_seed.csv", per_run)
            write_csv(args.output_dir / f"{question}_{table}_summary.csv", summary)

            print(f"\n### {table}\n")
            print("| Row | n | Forgetting | Current-task DSC | D_RMS | Outer-pair share (%) | Bottleneck share (%) |")
            print("|---|---|---|---|---|---|---|")
            for e in summary:
                print(f"| {e['row']} | {e['n_seeds']} "
                      f"| {cell(e['old_task_forgetting_mean'], e['old_task_forgetting_sd'], 3)} "
                      f"| {cell(e['new_task_dice_mean'], e['new_task_dice_sd'], 3)} "
                      f"| {cell(e['rms_delta_mean'], e['rms_delta_sd'], 5)} "
                      f"| {e['outer_pair_share_percent_mean']:.1f} | {e['bottleneck_share_percent_mean']:.1f} |")
    print(f"\nCSV files written to {args.output_dir}")


if __name__ == "__main__":
    main()
