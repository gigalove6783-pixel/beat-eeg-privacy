# Transmission ladder, yang2025 (principal contrasts; exploratory)

Rendered from `results/audit/ladder_compare_yang2025.csv`, the table written by `beat-audit compare`; the tool checked cohort, days, condition, k, physical sides and the population hash of every report before merging.

- Contrasts: `same-day-d2-d3` = trials 1-100 against 101-200 of day 2 and of day 3; `cross-day-d2-d3` = day 2 against day 3
- Population: 51 users on a common support of 20393 trials; every row of a condition and k shares one population hash.
- Comparison digest `4393bc82956a035a233ba519598fc59d1e54bfe5b72e3b8339293642798bc5c0`, equal for every rung of this table: it binds the ordered source-trial keys and layout (user, day, segment, order, target) of every model-seed input, equal to the plan's common support, the population and fold assignment of every cell, and the complete condition specification with the analysis settings; the script refuses a contrast set whose rungs differ.
- Attacker: open-set linkage of transmitted outputs only; comparators fitted on the development users of each fold and scored on the fold's unseen target users (feature rungs: cosine, LDA, rich-summary cosine and LDA, the strongest named post hoc; command rungs: cue-aware Bayes factor).
- Utility: later-day trials of the common support, per user (mean over model seeds 0-2), then mean over users with a 95% percentile bootstrap over users; task_ba = mean of the two class recalls of the argmax decision of the rung's decoder; rates over all trials (hold, move toward the cue, move away from it); a feature rung's receiver acts on the argmax of every trial (no hold).
- Seeds: model seeds 0,1,2; fold seed 20261111; attack seed 20261112 + 1009 x fold + 101 x model-seed position; matching permutations 20261112; bootstrap 20261113; per-user decoder CV seed 20261115; utility bootstrap 20261116.

## same-day-d2-d3, k = 20

| alternative | derived from | task BA | hold | wrong move | comparator | mean AUC [95% interval] | wording | data-processing guard |
|---|---|---|---|---|---|---|---|---|
| features_P0 |  | 0.555 | 0.000 | 0.445 | lda_cosine | 1.000 [1.000, 1.000] | linkage detected | not flagged |
| features_P2-RC_d8 |  | 0.544 | 0.000 | 0.456 | lda_cosine | 0.988 [0.979, 0.994] | linkage detected | not flagged |
| commands_shared_tau0.50 | features_P2-RC_d8 | 0.544 | 0.000 | 0.456 | bayes_factor | 0.827 [0.794, 0.859] | linkage detected | not flagged |
| commands_shared_tau0.60 | features_P2-RC_d8 | 0.544 | 0.473 | 0.227 | bayes_factor | 0.864 [0.829, 0.895] | linkage detected | not flagged |
| commands_per_user_tau0.50 | features_P0 | 0.545 | 0.000 | 0.455 | bayes_factor | 0.826 [0.789, 0.860] | linkage detected | not flagged |
| commands_per_user_tau0.60 | features_P0 | 0.545 | 0.035 | 0.438 | bayes_factor | 0.815 [0.777, 0.849] | linkage detected | not flagged |
| commands_per_user_tau0.80 | features_P0 | 0.545 | 0.118 | 0.397 | bayes_factor | 0.829 [0.790, 0.863] | linkage detected | not flagged |

Mean AUC by k (same-day-d2-d3; blank = not evaluable):

| alternative | k = 1 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|---|
| features_P0 | 0.990 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| features_P2-RC_d8 | 0.876 | 0.961 | 0.979 | 0.988 | 0.994 | 0.998 |
| commands_shared_tau0.50 | 0.587 | 0.723 | 0.777 | 0.827 | 0.878 | 0.919 |
| commands_shared_tau0.60 | 0.613 | 0.761 | 0.820 | 0.864 | 0.919 | 0.952 |
| commands_per_user_tau0.50 | 0.591 | 0.720 | 0.774 | 0.826 | 0.888 | 0.908 |
| commands_per_user_tau0.60 | 0.593 | 0.716 | 0.765 | 0.815 | 0.880 | 0.903 |
| commands_per_user_tau0.80 | 0.594 | 0.724 | 0.777 | 0.829 | 0.894 | 0.917 |

## cross-day-d2-d3, k = 20

| alternative | derived from | task BA | hold | wrong move | comparator | mean AUC [95% interval] | wording | data-processing guard |
|---|---|---|---|---|---|---|---|---|
| features_P0 |  | 0.555 | 0.000 | 0.445 | cosine_rich | 0.910 [0.864, 0.950] | linkage detected | not flagged |
| features_P2-RC_d8 |  | 0.544 | 0.000 | 0.456 | lda_cosine_rich | 0.835 [0.781, 0.880] | linkage detected | not flagged |
| commands_shared_tau0.50 | features_P2-RC_d8 | 0.544 | 0.000 | 0.456 | bayes_factor | 0.663 [0.590, 0.727] | linkage detected | not flagged |
| commands_shared_tau0.60 | features_P2-RC_d8 | 0.544 | 0.473 | 0.227 | bayes_factor | 0.696 [0.619, 0.759] | linkage detected | not flagged |
| commands_per_user_tau0.50 | features_P0 | 0.545 | 0.000 | 0.455 | bayes_factor | 0.634 [0.557, 0.707] | linkage detected | not flagged |
| commands_per_user_tau0.60 | features_P0 | 0.545 | 0.035 | 0.438 | bayes_factor | 0.627 [0.550, 0.701] | linkage detected | not flagged |
| commands_per_user_tau0.80 | features_P0 | 0.545 | 0.118 | 0.397 | bayes_factor | 0.635 [0.556, 0.710] | linkage detected | not flagged |

Mean AUC by k (cross-day-d2-d3; blank = not evaluable):

| alternative | k = 1 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|---|
| features_P0 | 0.835 | 0.881 | 0.899 | 0.910 | 0.924 | 0.922 |
| features_P2-RC_d8 | 0.753 | 0.812 | 0.822 | 0.835 | 0.866 | 0.878 |
| commands_shared_tau0.50 | 0.542 | 0.611 | 0.637 | 0.663 | 0.685 | 0.696 |
| commands_shared_tau0.60 | 0.573 | 0.648 | 0.677 | 0.696 | 0.715 | 0.723 |
| commands_per_user_tau0.50 | 0.535 | 0.594 | 0.622 | 0.634 | 0.655 | 0.657 |
| commands_per_user_tau0.60 | 0.535 | 0.588 | 0.616 | 0.627 | 0.651 | 0.655 |
| commands_per_user_tau0.80 | 0.535 | 0.592 | 0.619 | 0.635 | 0.654 | 0.661 |

## Data-processing guard, every declared derivation

- `commands_shared_tau0.50` declared derived from `features_P2-RC_d8`: not flagged at any of 12 evaluable cells
- `commands_shared_tau0.60` declared derived from `features_P2-RC_d8`: not flagged at any of 12 evaluable cells
- `commands_per_user_tau0.50` declared derived from `features_P0`: not flagged at any of 12 evaluable cells
- `commands_per_user_tau0.60` declared derived from `features_P0`: not flagged at any of 12 evaluable cells
- `commands_per_user_tau0.80` declared derived from `features_P0`: not flagged at any of 12 evaluable cells

Command line (from the project root):

```
python -m beat_audit compare features_P0=results/audit/ladder/yang2025/principal/features_P0/report.json features_P2-RC_d8=results/audit/ladder/yang2025/principal/features_P2-RC_d8/report.json commands_shared_tau0.50=results/audit/ladder/yang2025/principal/commands_shared_tau0.50/report.json commands_shared_tau0.60=results/audit/ladder/yang2025/principal/commands_shared_tau0.60/report.json commands_per_user_tau0.50=results/audit/ladder/yang2025/principal/commands_per_user_tau0.50/report.json commands_per_user_tau0.60=results/audit/ladder/yang2025/principal/commands_per_user_tau0.60/report.json commands_per_user_tau0.80=results/audit/ladder/yang2025/principal/commands_per_user_tau0.80/report.json --utility results/audit/ladder_compare_yang2025_utility.json --out results/audit/ladder_compare_yang2025.csv --derived-from=commands_shared_tau0.50=features_P2-RC_d8 --derived-from=commands_shared_tau0.60=features_P2-RC_d8 --derived-from=commands_per_user_tau0.50=features_P0 --derived-from=commands_per_user_tau0.60=features_P0 --derived-from=commands_per_user_tau0.80=features_P0
```
