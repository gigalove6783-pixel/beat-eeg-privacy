# Transmission ladder, bnci2014_001 (principal contrasts; exploratory)

Rendered from `results/audit/ladder_compare_bnci2014_001.csv`, the table written by `beat-audit compare`; the tool checked cohort, days, condition, k, physical sides and the population hash of every report before merging.

- Contrasts: `same-day-sessionE` = runs 1-3 against runs 4-6 of session E
- Population: 9 users on a common support of 2346 trials; every row of a condition and k shares one population hash.
- Comparison digest `e82ccbfb717cce2ef91fac8ac470f21231d1854d8064c803b25ac4747fca1956`, equal for every rung of this table: it binds the ordered source-trial keys and layout (user, day, segment, order, target) of every model-seed input, equal to the plan's common support, the population and fold assignment of every cell, and the complete condition specification with the analysis settings; the script refuses a contrast set whose rungs differ.
- Attacker: open-set linkage of transmitted outputs only; comparators fitted on the development users of each fold and scored on the fold's unseen target users (feature rungs: cosine, LDA, rich-summary cosine and LDA, the strongest named post hoc; command rungs: cue-aware Bayes factor).
- Utility: later-day trials of the common support, per user (mean over model seeds 0-2), then mean over users with a 95% percentile bootstrap over users; task_ba = mean of the two class recalls of the argmax decision of the rung's decoder; rates over all trials (hold, move toward the cue, move away from it); a feature rung's receiver acts on the argmax of every trial (no hold).
- Seeds: model seeds 0,1,2; fold seed 20261111; attack seed 20261112 + 1009 x fold + 101 x model-seed position; matching permutations 20261112; bootstrap 20261113; per-user decoder CV seed 20261115; utility bootstrap 20261116.
- The CSV also carries the tool's coverage rows cross-day (not evaluable: not requested on these days; see the contrasts above).

## same-day-sessionE, k = 20

| alternative | derived from | task BA | hold | wrong move | comparator | mean AUC [95% interval] | wording | data-processing guard |
|---|---|---|---|---|---|---|---|---|
| features_P0 |  | 0.679 | 0.000 | 0.319 | cosine | 1.000 [1.000, 1.000] | linkage detected | not flagged |
| features_P2-RC_d8 |  | 0.718 | 0.000 | 0.281 | cosine_rich | 0.848 [0.705, 0.995] | linkage detected | not flagged |
| commands_shared_tau0.50 | features_P2-RC_d8 | 0.718 | 0.000 | 0.281 | bayes_factor | 0.740 [0.596, 0.919] | not detected: insufficient evidence, not absence | not flagged |
| commands_shared_tau0.80 | features_P2-RC_d8 | 0.718 | 0.602 | 0.061 | bayes_factor | 0.796 [0.603, 0.984] | not detected: insufficient evidence, not absence | not flagged |
| commands_per_user_tau0.50 | features_P0 | 0.759 | 0.000 | 0.242 | bayes_factor | 0.713 [0.494, 0.934] | not detected: insufficient evidence, not absence | not flagged |
| commands_per_user_tau0.60 | features_P0 | 0.759 | 0.070 | 0.210 | bayes_factor | 0.724 [0.532, 0.909] | not detected: insufficient evidence, not absence | not flagged |
| commands_per_user_tau0.80 | features_P0 | 0.759 | 0.223 | 0.154 | bayes_factor | 0.703 [0.498, 0.934] | not detected: insufficient evidence, not absence | not flagged |

Mean AUC by k (same-day-sessionE; blank = not evaluable):

| alternative | k = 1 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|---|
| features_P0 | 0.982 | 0.997 | 0.999 | 1.000 | 1.000 |  |
| features_P2-RC_d8 | 0.623 | 0.763 | 0.816 | 0.848 | 0.889 |  |
| commands_shared_tau0.50 | 0.555 | 0.651 | 0.691 | 0.740 | 0.843 |  |
| commands_shared_tau0.80 | 0.539 | 0.639 | 0.712 | 0.796 | 0.787 |  |
| commands_per_user_tau0.50 | 0.519 | 0.602 | 0.648 | 0.713 | 0.778 |  |
| commands_per_user_tau0.60 | 0.517 | 0.598 | 0.637 | 0.724 | 0.750 |  |
| commands_per_user_tau0.80 | 0.519 | 0.593 | 0.660 | 0.703 | 0.694 |  |

## Data-processing guard, every declared derivation

- `commands_shared_tau0.50` declared derived from `features_P2-RC_d8`: not flagged at any of 5 evaluable cells
- `commands_shared_tau0.80` declared derived from `features_P2-RC_d8`: not flagged at any of 5 evaluable cells
- `commands_per_user_tau0.50` declared derived from `features_P0`: not flagged at any of 5 evaluable cells
- `commands_per_user_tau0.60` declared derived from `features_P0`: not flagged at any of 5 evaluable cells
- `commands_per_user_tau0.80` declared derived from `features_P0`: not flagged at any of 5 evaluable cells

Command line (from the project root):

```
python -m beat_audit compare features_P0=results/audit/ladder/bnci2014_001/principal/features_P0/report.json features_P2-RC_d8=results/audit/ladder/bnci2014_001/principal/features_P2-RC_d8/report.json commands_shared_tau0.50=results/audit/ladder/bnci2014_001/principal/commands_shared_tau0.50/report.json commands_shared_tau0.80=results/audit/ladder/bnci2014_001/principal/commands_shared_tau0.80/report.json commands_per_user_tau0.50=results/audit/ladder/bnci2014_001/principal/commands_per_user_tau0.50/report.json commands_per_user_tau0.60=results/audit/ladder/bnci2014_001/principal/commands_per_user_tau0.60/report.json commands_per_user_tau0.80=results/audit/ladder/bnci2014_001/principal/commands_per_user_tau0.80/report.json --utility results/audit/ladder_compare_bnci2014_001_utility.json --out results/audit/ladder_compare_bnci2014_001.csv --derived-from=commands_shared_tau0.50=features_P2-RC_d8 --derived-from=commands_shared_tau0.80=features_P2-RC_d8 --derived-from=commands_per_user_tau0.50=features_P0 --derived-from=commands_per_user_tau0.60=features_P0 --derived-from=commands_per_user_tau0.80=features_P0
```
