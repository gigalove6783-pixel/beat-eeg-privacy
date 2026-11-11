# Transmission ladder, yang2025 (calibration-day contrasts; exploratory)

Rendered from `results/audit/ladder_compare_yang2025_calibration_day.csv`, the table written by `beat-audit compare`; the tool checked cohort, days, condition, k, physical sides and the population hash of every report before merging.

- Contrasts: `cross-day-d1-d23` = day 1 (calibration) against days 2-3
- Population: 51 users on a common support of 30591 trials; every row of a condition and k shares one population hash.
- Comparison digest `3b6a843b57ef49bffcfc27b34d77b2170257c9fb33548398ee91ca04edf68c14`, equal for every rung of this table: it binds the ordered source-trial keys and layout (user, day, segment, order, target) of every model-seed input, equal to the plan's common support, the population and fold assignment of every cell, and the complete condition specification with the analysis settings; the script refuses a contrast set whose rungs differ.
- Attacker: open-set linkage of transmitted outputs only; comparators fitted on the development users of each fold and scored on the fold's unseen target users (feature rungs: cosine, LDA, rich-summary cosine and LDA, the strongest named post hoc; command rungs: cue-aware Bayes factor).
- Utility: later-day trials of the common support, per user (mean over model seeds 0-2), then mean over users with a 95% percentile bootstrap over users; task_ba = mean of the two class recalls of the argmax decision of the rung's decoder; rates over all trials (hold, move toward the cue, move away from it); a feature rung's receiver acts on the argmax of every trial (no hold).
- Seeds: model seeds 0,1,2; fold seed 20261111; attack seed 20261112 + 1009 x fold + 101 x model-seed position; matching permutations 20261112; bootstrap 20261113; per-user decoder CV seed 20261115; utility bootstrap 20261116.
- The CSV also carries the tool's coverage rows same-day (not evaluable: not requested on these days; see the contrasts above).

## cross-day-d1-d23, k = 20

| alternative | derived from | task BA | hold | wrong move | comparator | mean AUC [95% interval] | wording | data-processing guard |
|---|---|---|---|---|---|---|---|---|
| features_P0 |  | 0.555 | 0.000 | 0.445 | cosine_rich | 0.855 [0.802, 0.905] | linkage detected | not flagged |
| commands_shared_tau0.50 |  | 0.544 | 0.000 | 0.456 | bayes_factor | 0.533 [0.512, 0.556] | linkage detected | not evaluated: source rung unavailable |
| commands_shared_tau0.60 |  | 0.544 | 0.473 | 0.227 | bayes_factor | 0.551 [0.527, 0.575] | linkage detected | not evaluated: source rung unavailable |
| commands_per_user_tau0.50 | features_P0 | 0.545 | 0.000 | 0.455 | bayes_factor | 0.535 [0.509, 0.563] | linkage detected | not flagged |
| commands_per_user_tau0.60 | features_P0 | 0.545 | 0.035 | 0.438 | bayes_factor | 0.533 [0.509, 0.558] | linkage detected | not flagged |
| commands_per_user_tau0.80 | features_P0 | 0.545 | 0.118 | 0.397 | bayes_factor | 0.523 [0.502, 0.548] | linkage detected | not flagged |

Mean AUC by k (cross-day-d1-d23; blank = not evaluable):

| alternative | k = 1 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|---|
| features_P0 | 0.772 | 0.823 | 0.844 | 0.855 | 0.871 | 0.849 |
| commands_shared_tau0.50 | 0.500 | 0.519 | 0.519 | 0.533 | 0.546 | 0.546 |
| commands_shared_tau0.60 | 0.506 | 0.521 | 0.533 | 0.551 | 0.567 | 0.570 |
| commands_per_user_tau0.50 | 0.504 | 0.515 | 0.524 | 0.535 | 0.554 | 0.554 |
| commands_per_user_tau0.60 | 0.505 | 0.516 | 0.524 | 0.533 | 0.549 | 0.547 |
| commands_per_user_tau0.80 | 0.505 | 0.512 | 0.517 | 0.523 | 0.544 | 0.545 |

## Data-processing guard, every declared derivation

- `commands_per_user_tau0.50` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.60` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.80` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_shared_tau0.50` derives from `features_P2-RC_d8`, which is not a rung on these days, so the derivation cannot be declared: not evaluated: source rung unavailable
- `commands_shared_tau0.60` derives from `features_P2-RC_d8`, which is not a rung on these days, so the derivation cannot be declared: not evaluated: source rung unavailable

Command line (from the project root):

```
python -m beat_audit compare features_P0=results/audit/ladder/yang2025/calibration_day/features_P0/report.json commands_shared_tau0.50=results/audit/ladder/yang2025/calibration_day/commands_shared_tau0.50/report.json commands_shared_tau0.60=results/audit/ladder/yang2025/calibration_day/commands_shared_tau0.60/report.json commands_per_user_tau0.50=results/audit/ladder/yang2025/calibration_day/commands_per_user_tau0.50/report.json commands_per_user_tau0.60=results/audit/ladder/yang2025/calibration_day/commands_per_user_tau0.60/report.json commands_per_user_tau0.80=results/audit/ladder/yang2025/calibration_day/commands_per_user_tau0.80/report.json --utility results/audit/ladder_compare_yang2025_calibration_day_utility.json --out results/audit/ladder_compare_yang2025_calibration_day.csv --derived-from=commands_per_user_tau0.50=features_P0 --derived-from=commands_per_user_tau0.60=features_P0 --derived-from=commands_per_user_tau0.80=features_P0
```
