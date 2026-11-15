# Cue-blind attacker beside the cue-aware one

**Tier: exploratory and post hoc.** This experiment was decided after every other result was
known, to answer one question: every reported linkage gave the attacker each trial's task cue
(the `target` column); how much linkage does an attacker achieve who sees only the transmitted
outputs? Nothing here is a confirmatory result, and no verdict is drawn from the differences.

## What was run

- **Same tool and settings as the recorded runs:** `beat_audit` source `73ce3bec1b47…`, 10,000
  permutations, 2,000 bootstrap replicates, the recorded fold seed, seed, model seeds,
  populations, folds and windows. The only change is that the `target` array is removed from
  each input file, together with the cue declaration that names its levels. Every other array is
  copied unchanged. Without `target`, the rich window summary is the mean and log variance
  (instead of mean, the two cue means and log variance), and the Bayes factor counts commands in
  one stratum instead of one per cue. The mean comparators (`cosine`, `lda_cosine`) never read
  `target`, so their cue-blind values equal the cue-aware ones by construction.
- **Same support:** each cue-blind cell has the recorded cell's users, folds, analysed rows and
  sides; the run refuses otherwise. The population hash in the table holds for both columns.
- **Reproduction first:** for every condition the cue-aware cell was rerun with `target` present.
  AUC, interval, p, partner AUCs and per-user AUCs equal the recorded report exactly in all 11
  conditions (column `reproduction`). The cue-aware values in the table are read from the
  recorded reports.
- **Restricted through the ordinary API:** one condition and one k per run. The cue-blind run's
  own Holm family therefore has one test; `blind_wording_in_recorded_family` places its p in the
  recorded family instead, through the Holm bounds p ≤ p_Holm ≤ m·p; it certifies the wording
  and is not an exact adjusted p-value of a fully rerun cue-blind k family.
- **Paired difference:** cue-aware minus cue-blind mean per-user AUC, with the tool's subject
  bootstrap of the paired partner-AUC differences (2,000 replicates). It is an estimate, not a
  test.

## Table (k as stated; strongest recorded comparator, plus any other comparator that differs)

AUC is the mean per-user AUC with its 95% interval. Every value is "linkage detected" (p ≈ 1e-4)
except where stated.

| Condition | k | Comparator | Cue-aware | Cue-blind | Aware − blind |
|---|---|---|---|---|---|
| 1. V1: Yang2025 full features, day 1 vs days 2–3 | 20 | cosine_rich | .855 [.802, .905] | .852 [.806, .896] | +.002 [−.011, +.015] |
| | | lda_cosine_rich | .816 [.747, .880] | .842 [.776, .898] | −.026 [−.042, −.013] |
| 2. V2: Yang2025 P2-RC commands τ .60, day 1 vs days 2–3 | 50 | Bayes factor | .567 [.540, .596] | .554 [.526, .582] | +.013 [+.004, +.023] |
| 3. Yang2025 8-d features, day 2 vs day 3 | 20 | lda_cosine_rich | .835 [.781, .880] | .834 [.778, .881] | +.001 [−.006, +.009] |
| | | cosine_rich | .821 [.764, .868] | .792 [.739, .840] | +.029 [+.014, +.044] |
| 3. Yang2025 shared commands τ .50, day 2 vs day 3 | 20 | Bayes factor | .663 [.590, .727] | .650 [.577, .715] | +.013 [+.000, +.029] |
| 3. Yang2025 shared commands τ .60, day 2 vs day 3 | 20 | Bayes factor | .696 [.619, .759] | .685 [.608, .750] | +.011 [+.003, +.020] |
| 4. Lee2019 full features, session 1 vs 2 | 20 | cosine_rich | .954 [.929, .974] | .952 [.929, .971] | +.003 [−.006, +.010] |
| 4. Lee2019 8-d features, session 1 vs 2 | 20 | cosine_rich | .688 [.641, .731] | .681 [.636, .721] | +.007 [−.003, +.017] |
| 4. Lee2019 shared commands τ .50, session 1 vs 2 | 20 | Bayes factor | .630 [.588, .672] | .507 [.494, .519], p = .11, not detected | +.123 [+.082, +.163] |
| 4. Lee2019 shared commands τ .80, session 1 vs 2 | 20 | Bayes factor | .677 [.627, .722] | .670 [.619, .715] | +.008 [−.001, +.016] |
| V1 same-day: Yang2025 full features, halves of day 2 and day 3 | 20 | lda_cosine | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 (by construction) |
| | | cosine_rich | .999 [.997, 1.000] | .995 [.991, .998] | +.004 [+.001, +.007] |
| V2 same-day: Yang2025 P2-RC commands τ .60, halves of day 2 and day 3 | 50 | Bayes factor | .919 [.891, .942] | .910 [.879, .935] | +.010 [+.002, +.020] |

All rows, comparators, p-values, Holm adjustments, wording, users (51 on Yang2025, 54 on
Lee2019) and the input and source hashes are in `summary.csv`.

## Reading

- **Withholding the cue removes little linkage in most conditions.** The cue-blind attacker
  reaches within about .03 of the cue-aware AUC in every condition but one. Where the strongest
  comparator changes, the cue-blind attacker's best value is close to the cue-aware best (V1: .852
  against .855).
- **The exception is Lee2019's shared commands at τ .50:** the AUC falls from .630 to .507 and the
  cue-blind attacker does not detect linkage (insufficient evidence, not absence). At τ .50 these
  commands have no hold symbol (the streams contain left and right only), so without the cue
  each window of 20 outputs supplies the Bayes factor with its left/right counts alone. Why the
  value is near chance here was not tested. At τ .80 the stream also has holds and links without
  the cue (.670); that hold-related proportions account for this is an interpretation, not a
  tested mechanism.
- **On Yang2025 the same-τ commands still link cue-blind (.650 day 2 vs day 3).** Stable
  per-user command proportions across days would explain this; it was not tested either.
- **"Commands link less than the features they are decoded from" also holds cue-blind** on
  Yang2025 day 2 vs day 3: commands .650 and .685, 8-d features .834 at best. This is a
  descriptive comparison of separate intervals, not a paired test across rungs.
- **Lee2019's session 1 is the calibration session,** where features and shared commands are
  in-sample or cross-fitted; those rows are not comparable with Yang2025's later-day rows.

## What this does not cover

- **An attacker who infers the cue.** Commands are decoded estimates of the cue, and a cue
  classifier fitted on development users could label each window or trial. This attacker was
  not evaluated, and its performance is not bounded by the two measured attacks (in V1 the
  cue-blind `lda_cosine_rich` already exceeds its cue-aware counterpart, .842 against .816).
- Timing, sequence models or side information; other k, other comparators, or other conditions
  and cohorts.
- Any multiplicity control across conditions; the comparator named "strongest" was chosen on the
  same users.
- Any statement about protection of users: linkage below detection here is insufficient evidence,
  not absence.

## Files

- `summary.csv`: one row per condition and comparator.
- `<condition>/run.json`: inputs, withheld-input hashes, reproduction outcome, run times and
  support checks.
- `<condition>/{cue_aware_reproduction,cue_blind}/report.slim.json` (0.34–0.52 MB each, written by
  `scripts/audit_slim_reports.py`), `report.md` and the figures. The full `report.json` files and
  the cue-withheld inputs under `inputs/` are not tracked.

Reproduce (needs the recorded inputs under `artifacts/`):

```
python scripts/audit_cue_blind.py run V1 V2
python scripts/audit_cue_blind.py run --reproduce
python scripts/audit_cue_blind.py table
```
