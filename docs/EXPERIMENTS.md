# Experiment guide

## Registry-first workflow

Every paper configuration is defined once in
`reproducibility/experiments.json`. The runner validates experiment IDs, task
IDs, methods, and agreement between CLI indices and human-readable blocks
before rendering commands.

```bash
python reproducibility/run_experiment.py --list
python reproducibility/run_experiment.py --group fixed_lr_depth
python reproducibility/run_experiment.py --question Q2
```

Rendering is safe and does not start a process. Add `--execute` only to the
specific stage you intend to run. With `--group` or `--question`, executed runs
train and evaluate one after another, which suits a single GPU.

## Research questions

The `questions` section of the registry maps each paper table row to its
experiments:

- `Q1` (adaptation depth): `progressive`, `matched_rms`, and `mri_to_mri`;
- `Q2` (learning-rate allocation): `shared_vs_scale_aware`;
- `Q3` (smaller learning rate): `uniform_lr_control`.

`--question` selects every experiment of a question once, even when a run
appears in several rows.

## Plans aliases

Each experiment has its own plans identifier so that results land in separate
output folders. Before training or evaluating, the runner checks every task's
alias: a missing alias is created as a copy of that task's base plans
(`nnUNetPlansv2.1_plans_3D.pkl`), and an alias that differs from it stops the
run. A dry run only reports aliases it would create.

## Common initialization

Controlled two-task experiments must start from the same UMD checkpoint. Set:

```bash
export PAPER_INIT_CHECKPOINT=/absolute/path/to/the/common/fold_0/checkpoint_directory
```

The path is a trainer output directory, not the `.model` file itself. Keep its
checksum in run metadata.

## Fixed-LR progressive sweep

```bash
python reproducibility/run_experiment.py --group fixed_lr_depth --stage train
```

The group includes frozen-body, bottleneck-only, nested `E/D` regions, and an
`E0/D0`-only diagnostic. The diagnostic is not part of the nested progression.

## Scale-aware controls

```bash
python reproducibility/run_experiment.py --group scale_control --stage train
```

The gradient-normalized controls use 20 gradient-only batches, bottleneck as
the reference block, and the same base LR schedule; each seed is calibrated
separately. The broad
RMS-targeted configuration instead uses base LR 0.008; final RMS is measured
after training and is not constrained online.

## Cross-transition replication

```bash
python reproducibility/run_experiment.py --group cross_transition
```

This changes the successor task from ECPC-IDS to UT-EndoMRI but retains UMD as
the first task. It is an additional transition, not a randomized or reversed
task-order experiment.

## Uniform learning-rate control

```bash
python reproducibility/run_experiment.py --group uniform_lr_control
```

These runs train `R4` with a single base learning rate of 0.005 or 0.0025
(seed 12345) and are compared with the shared-LR (0.01) and scale-aware `R4`
runs in `Q3`.

## Additional seeds

Shared-LR and scale-aware configurations are repeated with adaptation seeds
23456 and 34567 (groups `seed_replicates` and `scale_control`). The data split
and initialization checkpoint remain fixed; only adaptation stochasticity
changes. The full-backbone shared-LR runs used for seed comparisons
(`ecpc_full_controlled_seed*`) use the same controlled trainer as the
restricted regions; `ecpc_full` follows the standard sequential trainer and is
kept as a boundary case.

## Evaluation

```bash
python reproducibility/run_experiment.py EXPERIMENT_ID --stage evaluate
python reproducibility/run_experiment.py EXPERIMENT_ID --stage evaluate --execute
```

The runner evaluates every task/head registered for the sequence. Always
inspect the rendered `trained_on`, `use_model`, `use_head`, and `evaluate_on`
arguments before execution.

## Run record

For each completed job, archive:

- experiment ID and full rendered command;
- Git commit and dirty status;
- environment export and GPU information;
- start/end timestamps;
- task split and seed;
- initialization and final checkpoint checksums;
- calibration JSON when applicable;
- evaluation JSON files.

Large artifacts belong in controlled object storage, not Git.
