# Transmission ladder, lee2019 (principal contrasts; exploratory)

Rendered from `results/audit/ladder_compare_lee2019.csv`, the table written by `beat-audit compare`; the tool checked cohort, days, condition, k, physical sides and the population hash of every report before merging.

- Contrasts: `same-day-session2` = offline against online phase of session 2
- Population: 54 users on a common support of 21600 trials; every row of a condition and k shares one population hash.
- Comparison digest `01c2e89a909b51e746066d706c9cfd05f330e437c90c02c182cd6c96e90c4f65`, equal for every rung of this table: it binds the ordered source-trial keys and layout (user, day, segment, order, target) of every model-seed input, equal to the plan's common support, the population and fold assignment of every cell, and the complete condition specification with the analysis settings; the script refuses a contrast set whose rungs differ.
- Attacker: open-set linkage of transmitted outputs only; comparators fitted on the development users of each fold and scored on the fold's unseen target users (feature rungs: cosine, LDA, rich-summary cosine and LDA, the strongest named post hoc; command rungs: cue-aware Bayes factor).
- Utility: later-day trials of the common support, per user (mean over model seeds 0-2), then mean over users with a 95% percentile bootstrap over users; task_ba = mean of the two class recalls of the argmax decision of the rung's decoder; rates over all trials (hold, move toward the cue, move away from it); a feature rung's receiver acts on the argmax of every trial (no hold).
- Seeds: model seeds 0,1,2; fold seed 20261111; attack seed 20261112 + 1009 x fold + 101 x model-seed position; matching permutations 20261112; bootstrap 20261113; per-user decoder CV seed 20261115; utility bootstrap 20261116.
- The CSV also carries the tool's coverage rows cross-day (not evaluable: not requested on these days; see the contrasts above).

## same-day-session2, k = 20

| alternative | derived from | task BA | hold | wrong move | comparator | mean AUC [95% interval] | wording | data-processing guard |
|---|---|---|---|---|---|---|---|---|
| features_P0 |  | 0.698 | 0.000 | 0.302 | lda_cosine | 1.000 [1.000, 1.000] | linkage detected | not flagged |
| features_P2-RC_d8 |  | 0.692 | 0.000 | 0.308 | lda_cosine_rich | 0.961 [0.941, 0.977] | linkage detected | not flagged |
| commands_shared_tau0.50 | features_P2-RC_d8 | 0.692 | 0.000 | 0.308 | bayes_factor | 0.789 [0.740, 0.832] | linkage detected | not flagged |
| commands_shared_tau0.80 | features_P2-RC_d8 | 0.692 | 0.648 | 0.055 | bayes_factor | 0.832 [0.784, 0.874] | linkage detected | not flagged |
| commands_per_user_tau0.50 | features_P0 | 0.646 | 0.000 | 0.354 | bayes_factor | 0.811 [0.768, 0.852] | linkage detected | not flagged |
| commands_per_user_tau0.60 | features_P0 | 0.646 | 0.048 | 0.330 | bayes_factor | 0.805 [0.761, 0.845] | linkage detected | not flagged |
| commands_per_user_tau0.80 | features_P0 | 0.646 | 0.155 | 0.279 | bayes_factor | 0.825 [0.782, 0.863] | linkage detected | not flagged |

Mean AUC by k (same-day-session2; blank = not evaluable):

| alternative | k = 1 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 |
|---|---|---|---|---|---|---|
| features_P0 | 0.993 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| features_P2-RC_d8 | 0.743 | 0.887 | 0.933 | 0.961 | 0.979 | 0.983 |
| commands_shared_tau0.50 | 0.556 | 0.683 | 0.742 | 0.789 | 0.843 | 0.876 |
| commands_shared_tau0.80 | 0.579 | 0.729 | 0.784 | 0.832 | 0.885 | 0.918 |
| commands_per_user_tau0.50 | 0.564 | 0.703 | 0.759 | 0.811 | 0.864 | 0.903 |
| commands_per_user_tau0.60 | 0.563 | 0.699 | 0.751 | 0.805 | 0.865 | 0.905 |
| commands_per_user_tau0.80 | 0.563 | 0.704 | 0.761 | 0.825 | 0.878 | 0.913 |

## Data-processing guard, every declared derivation

- `commands_shared_tau0.50` declared derived from `features_P2-RC_d8`: not flagged at any of 6 evaluable cells
- `commands_shared_tau0.80` declared derived from `features_P2-RC_d8`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.50` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.60` declared derived from `features_P0`: not flagged at any of 6 evaluable cells
- `commands_per_user_tau0.80` declared derived from `features_P0`: not flagged at any of 6 evaluable cells

Command line (from the project root):

```
python -m beat_audit compare features_P0=results/audit/ladder/lee2019/principal/features_P0/report.json features_P2-RC_d8=results/audit/ladder/lee2019/principal/features_P2-RC_d8/report.json commands_shared_tau0.50=results/audit/ladder/lee2019/principal/commands_shared_tau0.50/report.json commands_shared_tau0.80=results/audit/ladder/lee2019/principal/commands_shared_tau0.80/report.json commands_per_user_tau0.50=results/audit/ladder/lee2019/principal/commands_per_user_tau0.50/report.json commands_per_user_tau0.60=results/audit/ladder/lee2019/principal/commands_per_user_tau0.60/report.json commands_per_user_tau0.80=results/audit/ladder/lee2019/principal/commands_per_user_tau0.80/report.json --utility results/audit/ladder_compare_lee2019_utility.json --out results/audit/ladder_compare_lee2019.csv --derived-from=commands_shared_tau0.50=features_P2-RC_d8 --derived-from=commands_shared_tau0.80=features_P2-RC_d8 --derived-from=commands_per_user_tau0.50=features_P0 --derived-from=commands_per_user_tau0.60=features_P0 --derived-from=commands_per_user_tau0.80=features_P0
```
