# Changelog

## ISBI 2027 release

### Added

- `questions` section in `reproducibility/experiments.json` mapping the three
  research questions (adaptation depth, learning-rate allocation, smaller
  learning rate) to the runs behind each table row, and `--question` in
  `run_experiment.py` to run them.
- `reproducibility/summarize_questions.py`, which rebuilds the Q1–Q3 tables,
  including block-level displacement shares and mean ± SD over seeds.
- Scale-aware runs for seeds 23456 and 34567 (R5, R4, R3, R0), shared-LR seeds
  for R5, R3, and R0, and controlled-trainer full-backbone shared-LR runs.
- Uniform learning-rate control for R4 (base LR 0.005 and 0.0025).
- Aggregate reference results in `reproducibility/reported/`.

### Fixed

- **Held-out ECPC-IDS evaluation of three configurations.** In earlier
  results, the PET/CT test images of head-only, bottleneck-only (B), and
  shared-LR R5 runs were preprocessed with UMD (MRI) plans, because their
  experiment-specific plans aliases had been copied from the wrong task.
  Training was unaffected, and UMD forgetting values were correct. With the
  correct preprocessing, ECPC-IDS Dice is 0.280 (was 0.007) for B and 0.606
  (was 0.067) for R5 at seed 12345; head-only remains 0.000. All other
  configurations were re-evaluated and are unchanged.
- To prevent this class of error, `run_experiment.py` now creates every plans
  alias as a copy of the evaluated task's own base plans and refuses to run if
  an existing alias differs, and held-out evaluation (`nnUNet_evaluate2`) now
  stops with an error if the evaluated task's plans are missing instead of
  falling back to the plans of the last trained task.

### Removed

- `reported_results.csv` and `reported_seed_summary.csv` (superseded by
  `reproducibility/reported/`).
- The per-group shell launchers and `prepare_seed_plan_aliases.sh`; use
  `run_experiment.py --group` or `--question`, which handle plans aliases.
