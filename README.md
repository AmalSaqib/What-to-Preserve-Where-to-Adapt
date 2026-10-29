# What to Preserve, Where to Adapt

[![CI](https://github.com/AmalSaqib/What-to-Preserve-Where-to-Adapt/actions/workflows/ci.yml/badge.svg)](https://github.com/AmalSaqib/What-to-Preserve-Where-to-Adapt/actions/workflows/ci.yml)

Code for our study of catastrophic forgetting in continual gynecological image
segmentation. A 3D nnU-Net is first trained on uterine MRI (UMD) and then
adapted to tumor segmentation in PET/CT (ECPC-IDS), while different parts of
the encoder–decoder backbone are allowed to change. The study answers three
questions:

| | Question | Short answer |
|---|---|---|
| **Q1** | **Adaptation depth.** What happens as more of the network becomes trainable? | Adapting near the bottleneck preserves the previous task; forgetting rises sharply once the trainable region reaches the E4/D4 blocks, even at approximately matched average parameter movement. |
| **Q2** | **Learning-rate allocation.** Can scale-aware block-specific rates reduce forgetting at a given depth? | Yes, while part of the backbone stays frozen: over three seeds, forgetting drops from 0.363 to 0.045 at R4 and from 0.624 to 0.199 at R3, for about one Dice point. |
| **Q3** | **Smaller learning rate.** Is the benefit explained by simply using a smaller learning rate? | Not entirely: at similar average movement, a uniformly reduced rate forgets about four times more than scale-aware rates, because it concentrates movement in the outer blocks instead of the bottleneck. |

The implementation extends [Lifelong nnU-Net](https://github.com/MECLabTUDA/Lifelong-nnUNet)
(nnU-Net v1) with explicit encoder/bottleneck/decoder freezing, static
block-specific learning rates calibrated from initial relative gradient norms,
and a registry that defines every experiment in the paper.

The repository contains **no medical images, patient identifiers, model
checkpoints, or per-patient predictions**.

## Notation

The backbone has encoder blocks `E0`–`E5`, a bottleneck `B`, and decoder blocks
`D5`–`D0` (`E0`/`D0` are closest to the input/output). Nested trainable regions
grow outward from the bottleneck:

```text
B,   R5 = {E5, B, D5},   R4 = {E4..E5, B, D5..D4},   ...,   R0 = full backbone
```

During Task-2 adaptation only the selected blocks and the new task's head are
trained; everything else, including the Task-1 head, stays frozen.

## Setup

The recorded experiments use Python 3.9, nnU-Net v1, and one NVIDIA Quadro RTX
6000 (24 GB). Install the PyTorch build that matches your CUDA version, then:

```bash
conda env create -f environment.yml
conda activate depth-aware-cl
source reproducibility/config.example.env   # set PAPER_DATA_ROOT first to use existing data
```

Data are the public UMD, ECPC-IDS, and UT-EndoMRI datasets, converted to the
nnU-Net v1 raw-data layout with two input channels (see
[`docs/DATA.md`](docs/DATA.md)). Validate, then preprocess:

```bash
python reproducibility/preprocess.py            # read-only check
python reproducibility/preprocess.py --execute  # nnU-Net planning and preprocessing
```

All controlled runs start from the same UMD checkpoint, given by
`PAPER_INIT_CHECKPOINT`; its held-out UMD evaluation (`PAPER_BASELINE_UMD_EVAL`)
is the reference for forgetting. Both are set in
[`reproducibility/config.example.env`](reproducibility/config.example.env).

## Reproducing the three questions

Every configuration is defined once in
[`reproducibility/experiments.json`](reproducibility/experiments.json). Its
`questions` section lists which runs make up each table row. The runner prints
commands without executing them unless `--execute` is given:

```bash
python reproducibility/run_experiment.py --question Q1             # inspect commands
python reproducibility/run_experiment.py --question Q1 --execute   # train + evaluate, one run at a time
```

| Question | Table | Runs | Seeds |
|---|---|---|---|
| Q1 | `progressive` | head only, B, R5 … R1, R0 (shared LR 0.01) | 12345 |
| Q1 | `matched_rms` | B and R4 at LR 0.01, R1 at LR 0.008 (approximately equal final RMS) | 3 |
| Q1 | `mri_to_mri` | B, R4, R1 for UMD → UT-EndoMRI | 12345 |
| Q2 | `shared_vs_scale_aware` | R5, R4, R3, R0 with shared and scale-aware rates | 3 |
| Q3 | `uniform_lr_control` | R4 with shared LR 0.01, uniform 0.005 and 0.0025, and scale-aware rates | 12345 |

Runs shared between questions are trained once. The three adaptation seeds are
12345, 23456, and 34567; the data split and the initial checkpoint are fixed.

After the runs finish, rebuild the tables:

```bash
python reproducibility/summarize_questions.py                  # all questions
python reproducibility/summarize_questions.py --question Q3    # one question
```

For every row the script reports UMD forgetting, current-task Dice, RMS
parameter displacement, and the share of squared displacement in the
outermost trainable encoder–decoder pair and in the bottleneck, as per-seed
values and mean ± sample SD. It writes CSV files to `outputs/questions/`.

The values reported in the paper are included as aggregate, patient-anonymous
reference files in [`reproducibility/reported/`](reproducibility/reported/).

## Other experiments in the registry

| Group | Contents |
|---|---|
| `single_task` | Independent UMD, ECPC-IDS, and UT-EndoMRI models |
| `fixed_lr_depth` | Frozen, bottleneck, nested regions, and an `E0/D0`-only diagnostic |
| `scale_control` | Scale-aware runs and the RMS-targeted broad run |
| `seed_replicates` | Additional shared-LR seeds |
| `uniform_lr_control` | Uniformly reduced learning rates for R4 |
| `cross_transition` | UMD → UT-EndoMRI |
| `continual_baselines` | Sequential, rehearsal, LwF, and EWC |

```bash
python reproducibility/run_experiment.py --list
python reproducibility/run_experiment.py --group continual_baselines
```

The joint MultiTalent baseline uses a separate nnU-Net v2 environment; see
[`reproducibility/MULTITALENT.md`](reproducibility/MULTITALENT.md). Further
analysis utilities (single-checkpoint displacement, case-bootstrap intervals,
and inference-time block ablation) are described in
[`reproducibility/README.md`](reproducibility/README.md).

## Scope and limitations

- One architecture family (3D nnU-Net) and a fixed task order with UMD first.
- Encoder and decoder blocks at the same resolution are always trained
  together, so conclusions concern resolution stages, not encoder versus
  decoder sides.
- Scale-aware calibration changes both the overall size and the depth-wise
  distribution of updates; Q3 tests the size component directly at R4.
- RMS matching is assessed at the final checkpoint and is not enforced during
  training.
- Seed standard deviations measure training variability; case-bootstrap
  intervals measure evaluation-cohort uncertainty. The two are not
  interchangeable.
- The UT-EndoMRI input sequence was chosen in an exploratory comparison that
  included the held-out cases, so that transition is supporting evidence only.

See [`docs/METHOD.md`](docs/METHOD.md), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md),
and [`docs/RESULTS.md`](docs/RESULTS.md) for details, and
[`CHANGELOG.md`](CHANGELOG.md) for corrections to earlier versions.

## Development checks

These need no data or GPU:

```bash
python -m unittest discover -s reproducibility/tests -v
python reproducibility/run_experiment.py --list > /dev/null
python -m compileall -q reproducibility
```

## Attribution and license

This repository is derived from Lifelong nnU-Net and includes its original
continual-learning implementations. Please cite both this study and the
upstream framework. The code is distributed under the
[Apache License 2.0](LICENSE).
