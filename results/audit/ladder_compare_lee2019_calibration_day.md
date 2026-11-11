# Transmission ladder, lee2019 (calibration-day contrasts; exploratory)

Rendered from `results/audit/ladder_compare_lee2019_calibration_day.csv`, the table written by `beat-audit compare`; the tool checked cohort, days, condition, k, physical sides and the population hash of every report before merging.

- Contrasts: `cross-day-session1-session2` = session 1 (calibration) against session 2
- Population: 54 users on a common support of 21600 trials; every row of a condition and k shares one population hash.
- Comparison digest `11057ec3a51068108f60a23804cc7704ee666ab9e5c851edd5159b7e38315783`, equal for every rung of this table: it binds the ordered source-trial keys and layout (user, day, segment, order, target) of every model-seed input, equal to the plan's common support, the population and fold assignment of every cell, and the complete condition specification with the analysis settings; the script refuses a contrast set whose rungs differ.
- Attacker: open-set linkage of transmitted outputs only; comparators fitted on the development users of each fold and scored on the fold's unseen target users (feature rungs: cosine, LDA, rich-summary cosine and LDA, the strongest named post hoc; command rungs: cue-aware Bayes factor).
- Utility: later-day trials of the common support, per user (mean over model seeds 0-2), then mean over users with a 95% percentile bootstrap over users; task_ba = mean of the two class recalls of the argmax decision of the rung's decoder; rates over all trials (hold, move toward the cue, move away from it); a feature rung's receiver acts on the argmax of every trial (no hold).
- Seeds: model seeds 0,1,2; fold seed 20261111; attack seed 20261112 + 1009 x fold + 101 x model-seed position; matching permutations 20261112; bootstrap 20261113; per-user decoder CV seed 20261115; utility bootstrap 20261116.
- The CSV also carries the tool's coverage rows same-day (not evaluable: not requested on these days; see the contrasts above).

## cross-day-session1-session2, k = 20

| alternative | derived from | task BA | hold | wrong move | comparator | mean AUC [95% interval] | wording | data-processing guard |
|---|---|---|---|---|---|---|---|---|
| features_P0 |  | 0.698 | 0.000 | 0.302 | cosine_rich | 0.954 [0.929, 0.974] | linkage detected | not flagged |
| features_P2-RC_d8 |  | 0.692 | 0.000 | 0.308 | cosine_rich | 0.688 [0.641, 0.731] | linkage detected | not flagged |
| commands_shared_tau0.50 | features_P2-RC_d8 | 0.692 | 0.000 | 0.308 | bayes_factor | 0.630 [0.588, 0.672] | linkage detected | not flagged |
| commands_shared_tau0.80 | features_P2-RC_d8 | 0.692 | 0.648 | 0.055 | bayes_factor | 0.677 [0.627, 0.722] | linkage detected | not flagged |
| commands_per_user_tau0.50 | features_P0 | 0.646 | 0.000 | 0.354 | bayes_factor | 0.618 [0.576, 0.660] | linkage detected | not flagged |
| commands_per_user_tau0.60 | features_P0 | 0.646 | 0.048 | 0.330 | bayes_factor | 0.613 [0.573, 0.655] | linkage detected | not flagged |
| commands_per_user_tau0.80 | features_P0 | 0.646 | 0.155 | 0.279 | bayes_factor | 0.612 [0.572, 0.654] | linkage detected | not flagged |

Mean AUC by k (cross-day-session1-session2; blank = not evaluable):

| alternative | k = 1 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|---|
| features_P0 | 0.877 | 0.931 | 0.946 | 0.954 | 0.962 | 0.967 |
| features_P2-RC_d8 | 0.515 | 0.636 | 0.671 | 0.688 | 0.723 | 0.759 |
| commands_shared_tau0.50 | 0.520 | 0.571 | 0.602 | 0.630 | 0.660 | 0.680 |
| commands_shared_tau0.80 | 0.537 | 0.619 | 0.653 | 0.677 | 0.707 | 0.745 |
| commands_per_user_tau0.50 | 0.518 | 0.560 | 0.592 | 0.618 | 0.643 | 0.665 |
| commands_per_user_tau0.60 | 0.519 | 0.552 | 0.582 | 0.613 | 0.641 | 0.664 |
| commands_per_user_tau0.80 | 0.518 | 0.553 | 0.583 | 0.612 | 0.641 | 0.655 |

## Data-processing guard, every declared derivation

- `commands_shared_tau0.50` declared derived from `features_P2-RC_d8`: flagged at cross-day-session1-session2 k = 1 (the feature attacker is too weak there; the feature rung's exposure is at least the command rung's)
- `commands_shared_tau0.80` declared derived from `features_P2-RC_d8`: flagged at cross-day-session1-session2 k = 1 (the feature attacker is too weak there; the feature rung's exposure is at least the command rung's)
- `commands_per_user_tau0.50` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.60` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.80` declared derived from `features_P0`: not flagged at any of 6 evaluable cells

Command line (from the project root):

```
python -m beat_audit compare features_P0=results/audit/ladder/lee2019/calibration_day/features_P0/report.json features_P2-RC_d8=results/audit/ladder/lee2019/calibration_day/features_P2-RC_d8/report.json commands_shared_tau0.50=results/audit/ladder/lee2019/calibration_day/commands_shared_tau0.50/report.json commands_shared_tau0.80=results/audit/ladder/lee2019/calibration_day/commands_shared_tau0.80/report.json commands_per_user_tau0.50=results/audit/ladder/lee2019/calibration_day/commands_per_user_tau0.50/report.json commands_per_user_tau0.60=results/audit/ladder/lee2019/calibration_day/commands_per_user_tau0.60/report.json commands_per_user_tau0.80=results/audit/ladder/lee2019/calibration_day/commands_per_user_tau0.80/report.json --utility results/audit/ladder_compare_lee2019_calibration_day_utility.json --out results/audit/ladder_compare_lee2019_calibration_day.csv --derived-from=commands_shared_tau0.50=features_P2-RC_d8 --derived-from=commands_shared_tau0.80=features_P2-RC_d8 --derived-from=commands_per_user_tau0.50=features_P0 --derived-from=commands_per_user_tau0.60=features_P0 --derived-from=commands_per_user_tau0.80=features_P0
```
