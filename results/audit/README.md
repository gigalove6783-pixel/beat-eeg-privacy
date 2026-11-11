# Audit results: how to read this folder

Outputs of `beat_audit` on BEAT's three cohorts (`docs/plan_v3.md`). The measurement is open-set
linkage: an attacker who never enrolled the target users decides whether two windows of
transmitted outputs come from the same person. The statistic is the mean per-user AUC (0.5 is
chance). It is not an identification rate.

## Tiers

| Folder | Cohort | Tier | Verdicts |
|---|---|---|---|
| `bnci2014_001/` | BCI IV 2a, 9 users | *exploratory* | none |
| `lee2019/` | Lee2019, 54 users | *exploratory*; the tool was developed on it | none |
| `yang2025/` | Yang2025, 51 users | *pre-specified (non-blind)*: fixed at `freeze-5` (commit `55cd01e`) before computation, run once | V1 and V2 only (`yang2025/decision.json`) |

Yang2025 was already analysed in study M, whose identity results were known when v3 was designed.
The batch is therefore not blind and not a confirmatory replication.

## Verdicts and estimates

- Only two numbers carry a verdict: V1 and V2 in `yang2025/decision.json`, decided by a Holm
  family of two at α = 0.05. The paired same-day minus cross-day contrast in the same file is a
  pre-specified estimate without a verdict.
- Every other cell of every cohort is an estimate. The tool prints "linkage detected" in
  `report.md` whenever a cell's permutation p-value passes Holm's correction over the k grid of
  its comparator and condition. In the Yang2025 reports this wording appears on 179 of 180 rows;
  for the 177 rows that are not V1 or V2 it is the tool's per-cell convention, not a
  pre-specified verdict.
- Two Holm families exist and give different adjusted p-values for the same cell. `report.md`
  and the column `holm_p_over_k` of the summaries adjust over the k grid. `decision.json` adjusts
  over V1 and V2. Example: V1 has 0.0005 over k and 0.0002 in the decision.

## Labels the standalone reports do not carry

The `report.md` files are the tool's generic output and do not name the tier or the day-1 status
of a stream. For Yang2025:

- Full features (`features_P0`): the tangent reference was fitted on day 1, as in study M's
  claim M1. Cells that involve day 1 are in-sample in that respect.
- Commands: day-1 commands come from study M's cross-fitted day-1 logits, days 2–3 from the
  final models.
- 8-d features (`features_P2-RC_d8`): days 2–3 only, because day 1 trained the encoder.
- The attacker never fits on a fold's target users. The BCI's own encoder, recentring and
  decoder were fitted with these users' calibration data, so the result is about users unseen
  by the attacker, not users unseen by the BCI.

For 2a and Lee2019 the same notes are in the `day1_note` column of `summary.csv`.

## Files

- `summary.csv`: 2a and Lee2019 (development). `yang2025/summary.csv`: the frozen batch. The two
  are kept apart on purpose; plan check A4 names one summary file and is met by these two.
- `yang2025/decision.json`, `yang2025/cells.json`: verdicts and the status of all 180 cells.
- `freeze/`: the bundle (source zip, hashes, environment, folds, seeds, predictions, job
  manifest). `attempts.jsonl`: every execution with its kind.
- `parity.json`: the closed-set module against study L's registered values.
  `validation.json`: synthetic validation records. `runtime_projection.json`,
  `v1_selection.json`, `export_manifest.json`, `export_checks.json`, `utility.csv`: inputs,
  gate and selection records.
- `report_index.json`: size and SHA-256 of each full `report.json`. Those files hold the
  component-level evidence (up to 208 MB each) and are not tracked in Git.

## Independent audit of the frozen batch (2026-11-12)

An independent reviewer recomputed, with separate code, V1, V2 and the contrast from the stored
evidence and from the exported inputs: all values and both decisions agree (largest difference
2.2 × 10⁻¹⁶), no target user of a fold contributes to anything fitted for that fold, all 180
expected cells are present once, and the executed source, configuration and inputs equal the
bundle. The freeze commit and tag precede the batch by 21 seconds in the Git history and the
log; these timestamps, the absence of unlogged runs and the push order are self-attested. The
audit's findings on wording and labels are what this file addresses.
