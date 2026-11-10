# Pre-registration M: confirmatory cross-day replication on Yang2025 (WBCIC-SHU)

Status: **sealed (seal 4)** after two independent pre-registration reviews; the seal commit, the
annotated tag `seal-4` and its RFC 3161 timestamp are recorded in `seals/manifest.yaml` and
`results/exp_m/seal4_timestamp/`. Recording days 2–3 were not downloaded before the seal. The
`lead_status: draft` label in `configs/exp_m.yaml` is a stale label in a hashed file and has no
effect on the analysis. Lane K is frozen after its re-review; every value
that uses K's code was recomputed by the post-freeze rerun (section 11) and
`thresholds_status` is `frozen`. The confirmatory claims, τ*, the gates and the power
table do not use K's EQ. This file and `configs/exp_m.yaml` supersede `configs/exp_k.yaml`'s
declared decision rule for this study.
Plan: `.mixdog/plan_v2.md` revision 4, lane M. Config: `configs/exp_m.yaml`. Code:
`beat/yang2025.py`, `beat/exp_m.py`, `scripts/fetch_yang2025.py`.

## 1. Purpose

Replicate BEAT's identity-exposure findings on 51 users that no BEAT analysis has used, with
**cross-day** exposure as the primary estimand: a decoder calibrated on recording day 1 is
deployed on days 2 and 3, and an attacker who observes full features or the decoded command
stream tries to re-identify users across days.

## 2. Data and access

- Source (the only one): Yang et al., WBCIC-SHU Motor Imagery Dataset, figshare article
  22671172, version 5, file 51001884 (`WBCIC_SHU Motor Imagery dataset.zip`, 65,630,556,299
  bytes, MD5 e5384a58ac51b0d0c78a28a500185479, storage ETag
  `"6613e0a0b9c4139eff285d45dd9f9739-251"`), CC BY 4.0. Two-class paradigm (left vs right
  hand grasping), 51 users (`sourcedata/2C dataset/sub-001..sub-051`), three recording days
  (`ses-01..03`), 200 trials per day.
- Raw route: `sourcedata/2C dataset/sub-NNN/ses-0D/eeg/{data,evt}.bdf`. The processed `.mat`
  derivatives (Pz reference, FIR 0.5–40 Hz, epoched after cue end) are not used.
- Access before seal 4: an explicit allowlist (`beat.yang2025.allowlist`) of the 102 day-1
  members and 12 top-level metadata files. The range reader refuses every other member before
  any request (unit test over all 204 day-2/3 paths, `tests/test_yang2025.py`). Besides member
  bytes, lane M's reader requested only the archive's end records and central directory
  (member names, sizes and offsets of all days). Every byte-range request of lane M's reader
  is logged with UTC time, member, range, figshare version and ETag in
  `results/exp_m/download_log.jsonl`; this log is self-attested.
- Figshare API metadata calls (not in the request log): each resolution of the signed storage
  URL (`beat.yang2025.resolve_source`) read the article metadata from
  `https://api.figshare.com/v2/articles/22671172` (version, file id, name, size, MD5 and
  download URL; no archive bytes) and the download URL's redirect without following it —
  at the start of each fetch run and after each of the 53 HTTP 403 responses (expired
  signed URL) in the request log. The Lead's ad hoc script below resolved the file the same
  way through the API.
- **Access before stage M0 (Lead account).** Before stage M0, the Lead made unlogged HTTP
  range requests to the same figshare file with an ad hoc Python script (urllib Range
  requests behind zipfile): on 2026-11-10 at about 14:20–14:40 UTC it read the
  end-of-central-directory records and the central directory twice to list member names and
  sizes (read-ahead 1 MiB from the central-directory start, so no member data); on
  2026-11-11 at about 15:50–16:10 UTC (times reconstructed from the session record, not
  logged) it read the top-level member task-motorimagery_channels.tsv (read-ahead up to
  64 KiB, within the trailing top-level metadata members and the central directory). Member
  names and sizes of all days were therefore seen; no day-2/3 member bytes were requested.
  These requests cannot be evidenced from the request log.
- SHA-256 of every day-1 raw member and metadata file (recorded before each raw file was
  deleted) and of every reduced day-1 file `data/processed/yang2025/day1_subNNN.npz`
  (recorded when it was written, also in `results/exp_m/day1_events.jsonl`):
  `checksums/yang2025_day1.sha256` (165 entries; a seal-4 file).
- Days 2–3 are requested only after seal 4, by `scripts/fetch_yang2025.py --days 2 3
  --seal-commit HASH`, which first runs `beat.exp_m.verify_seal` (section 9) and
  `beat.yang2025.post_seal_allowlist`: annotated tag `seal-4` naming HASH, and
  `C:\Program Files\Git\usr\bin\openssl.exe ts -verify -digest D -in seal4.tsr -CAfile
  cacert.pem -untrusted tsa.crt` exiting 0 with "Verification: OK", where D = SHA-256 of
  the 40-character seal commit id in ASCII without a newline (the RFC 3161 request's
  message imprint; only this hash is sent to the TSA) and the three files lie in
  `results/exp_m/seal4_timestamp/` under these names. The token must carry a SHA-256
  imprint; each verification (command, output, OpenSSL version, token time) is appended to
  `results/exp_m/seal4_timestamp/guard_verification.jsonl`. Day-2/3 byte-range requests are
  logged to `results/exp_m/download_log_day23.jsonl` (the sealed day-1 log is not appended
  to); raw-member and reduced-file hashes go to `checksums/yang2025_day23.sha256`.
- Disclosure: the dataset paper (Yang et al., Scientific Data 2025,
  doi:10.1038/s41597-025-04826-y) and its published decoding statistics were read before
  this pre-registration; no BEAT analysis of days 2–3 exists.
- Disclosure: before τ*, the claims and the gates were frozen, **day-1 identity results
  were inspected across τ values (0.5, 0.8, then the τ grid) and across statistics** (top-1
  N-way and within-group recall, percentile rank, four attackers, k = 1–50) by the
  development runs and by an advisor consulted by the Lead; the earlier draft (τ = 0.8,
  top-1 M3, M4 bound) was replaced in light of them. **L's cross-day rank figures (Lee2019)
  were computed during design.** The τ rule itself uses no identity outcome (section 4).

## 3. Events, timing and reduction (confirmed on day 1)

- `evt.bdf` is a BDF+ annotation file (one `BDF Annotations` signal, 206 one-second records).
  Codes: **1 = left hand, 2 = right hand** (task-motorimagery_events.json), one per trial at
  the **trial onset = start of the 1.5 s video cue**; **9** = boundary between the five
  40-trial runs (4 per day); **7** and **8** = one each before the first trial (pre-task
  markers, unused). Observed on day 1 (`results/exp_m/day1_events.jsonl`): 49 users have
  200 trials (100 left, 100 right) in five runs of 40; sub-023 and sub-026 have 199 (one
  run-1 trial carries code 255 instead of a class code and is not a trial; 99 left, 100
  right). Onset-to-onset interval within runs 7.996–8.007 s (per-user median 8.000–8.007 s);
  recording length 2142–3092 s; 10,198 trials in total, all finite.
- Alignment: onsets count from the start of `data.bdf` (1000 Hz, 64 signals, 59 EEG; header
  start times agree within the 1 s header resolution: offset 0 s for 34 users, −1 s for 17).
  Evidence that code onset = video onset: in 36 of 51 users the largest trial-averaged
  occipital (O1, Oz, O2) 1–20 Hz deflection within −2..3 s lies 0.10–0.41 s after the code
  (median 0.14 s; a visual evoked response to the video); in the other 15 the largest
  deflection lies elsewhere in the window (diagnostic only, zero-phase, not part of the
  analysis signal). The fixed 8.0 s trial spacing and the published paradigm (1.5 s cue,
  4 s imagery, rest) agree with this alignment.
- Analysis window: **2.5–5.5 s after the trial code = 1–4 s after imagery onset** (L's
  window).
- Reduction (`beat.yang2025.reduce_recording`, mirroring `scripts/prepare_lee2019.py`):
  19 channels FC5, FC3, FC1, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CP2,
  CP4, CP6 (L's 20 motor channels without CPz, which this montage does not have);
  recorded reference kept (`EEGReference: REF`, as in L, no re-referencing); causal 4th-order
  Butterworth 8–30 Hz at 1000 Hz on the continuous recording (initial conditions from the
  first sample); decimation by 4 to 250 Hz; 750 samples per trial; a trial is non-finite if
  any raw sample is non-finite from 1 s before its window to its end, or if its window
  extends past the end of the recording (its signals are then NaN). Covariances: OAS,
  190-d tangent vectors.
- **Days 2–3: deviating or malformed recordings** (the day-1 rules, applied without abort;
  `beat.yang2025.reduce_recording`, `load_trials`, `scripts/fetch_yang2025.py`):
  - trial windows past the end of the recording → non-finite trials (counted as
    `past_end_trials`);
  - a header start-time offset between `data.bdf` and `evt.bdf` of any size → onsets still
    count from the start of `data.bdf`, as on day 1; the offset is logged
    (`event_start_offset_s`, `offset_within_header_resolution` = |offset| ≤ 1 s), never
    corrected;
  - a run count other than 5 → no effect (run numbers are used only for day-1
    cross-fitting); counts per run are logged;
  - more than 200 class-coded trials → all are kept in recorded order (no trial is dropped
    for count reasons; `extra_trials` = n − 200 is logged); the same-day halves are trials
    1–100 and 101–n;
  - a member absent from the archive (`status: absent`), or a recording that
    `reduce_recording` cannot reduce — unreadable header or annotations, missing motor
    channel or annotation signal, sampling rate other than 1000 Hz, no class-coded trial
    (`status: unparseable`, with the error) → no reduced file; that user-day is missing;
  - a reduced file missing at the final → that user-day is missing.
  Every user-day gets one line in `results/exp_m/day2_3_events.jsonl` (status, trial, run,
  class, retained, past-end and extra-trial counts, offset, visual-evoked latency, codes,
  reduced-file hash); missing user-days are also listed in `decision.json`
  (`missing_user_days`) and `final_exclusions.csv` (total 0). A missing user-day enters the
  claims through the missing-cell rule (section 8).

## 4. Systems (fitted on day 1 only)

- Decoders, settings unchanged from Experiment L (`configs/exp_l.yaml` training):
  P0 (logistic regression on the common tangent space), P2-RC d = 8 (recentred task-only
  bottleneck), P3 λ = 1 (gradient-reversal, common family, d = 8). Model seeds 0, 1, 2.
- Development (pre-seal): leave-one-run-out cross-fitting over the five day-1 runs; epoch
  budgets by `exp_b.choose_epochs` over 40-epoch pilots (families `common_d8`, `rc_d8`);
  temperature = NLL-optimal on the cross-fitted day-1 logits; command rule of Experiment C
  (temperature-scaled maximum probability below τ → hold). Day-1 command streams are
  cross-fitted (out-of-fold); deployment streams on days 2–3 come from the final models
  fitted on all retained day-1 trials with the sealed budgets.
- **τ rule** (uses no identity outcome): τ* = the value on the grid 0.50, 0.51, …, 0.90
  whose day-1 cross-fitted P2-RC d = 8 hold rate (share of held trials, mean over model
  seeds) is closest to **0.720** — L's cross-fitted P2-RC d = 8 hold rate at its sealed
  τ = 0.8 (0.7201, `results/exp_k/stream_summary.csv`, lee2019, cal_oof); ties → smaller τ.
  Result: **τ* = 0.60** (hold 0.7231; 0.59 → 0.6761, 0.61 → 0.7649;
  `results/exp_m/dev_tau_grid.csv`). τ = 0.5 (no hold) is reported as a secondary operating
  point; τ = 0.8 is not used.
- Primary command-stream system: **P2-RC d = 8 at τ* = 0.60**. Development values
  (`results/exp_m/dev_decoders.json`, `dev_task.csv`, `dev_streams.csv`): 10,198 day-1
  trials, all retained; budgets common_d8 = 12 epochs, rc_d8 = 1 epoch; cross-fitted
  balanced accuracy (mean over users and seeds) P0 0.578, P2-RC d = 8 0.582, P3 λ = 1 0.565;
  temperatures P0 1.41, P2-RC 1.01–1.10, P3 1.82–1.91. P2-RC at τ* = 0.60: hold 72.3 %,
  correct 18.5 %, wrong 9.2 %, net progress 0.094, weighted utility 0.048; at τ = 0.5:
  hold 0 %, net progress 0.165.
- Utility weights: weighted utility = P(correct) − w · P(wrong) with w = τ/(1 − τ)
  (Experiment K; w = 1.5 at τ* = 0.60, w = 1 at τ = 0.5, where it equals net progress).
- Groups for EQ and within-group statistics: K's rule — users sorted by prior-weighted
  calibration net progress, consecutive chunks of g = 3; leftovers join the last group.
  Calibration = the enrolment stream: on day 1 (gates) the enrolment half; in the final
  **the whole day-1 cross-fitted stream** (`calibration_groups(day1, …)` in
  `beat.exp_m.run_final`). The final's groups are therefore formed from exactly the rows
  the cross-day attacker enrols on (and from all five blocks of EQ's cross-fitting, so a
  block's group membership also used that block), never from days 2–3; the same day-1
  groups are used in all three scenarios, including dep_to_dep and same_day_halves, whose
  enrolment rows are not day 1. Grouping is not cross-fitted: an attacker with the group
  revealed knows the deployed grouping, which a deployment would form from calibration
  data.
- EQ: Experiment K's equalized decoding, variant **confusion**, objective
  P(correct) − w · P(wrong), fitted by K's LP on smoothed (+0.5) day-1 confusion counts.
  The day-1 enrolment stream is post-processed with **cross-fitted** matrices (5 contiguous
  blocks per user, each block by matrices fitted on the other blocks); days 2–3 with the
  matrices fitted on all of day 1. Post-processing seeds 0, 1, 2.
- **Recalibration only** (utility baseline, `recal`): the same LP with every user as its own
  group (`fit_eq` with `own_groups`), i.e. each user's commands remapped to that user's
  utility optimum without equalization; fitted and cross-fitted on day 1 exactly as EQ.
  EQ's loss against the undefended τ rule mixes this recalibration gain with the cost of
  equalization and can be negative, so the **privacy cost** of a defence is reported as
  recalibration-only utility − defended utility (per user, same test streams), next to the
  loss against the undefended stream.
- Randomized response: k-ary RR over the 3 outputs (output kept with probability 1 − q,
  otherwise uniform), (a) q = 0.5 on the undefended stream and (b) **RR-matched: RR(q)
  applied after recalibration only**, with q chosen so that its day-1 weighted utility
  equals EQ's day-1 weighted utility: q = (U_r − U_eq) / (U_r − (1 − w)/3), U_r and U_eq =
  mean (over users and model seeds) cross-fitted day-1 weighted utility of recalibration
  only and EQ (exact for RR; clipped to [0, 1]; defined whenever U_r exceeds (1 − w)/3,
  the utility of uniformly random outputs: −0.167 at τ* = 0.60, 0 at τ = 0.5).
  Day-1 values with frozen K code (`selection.json` `rr_matched`; means over 51 users, 95 %
  t-intervals):

  | τ | w | U undefended | U recalibration | U EQ | privacy cost (U_r − U_eq) | EQ loss vs undefended | **q (RR-matched)** |
  |---|---|---|---|---|---|---|---|
  | 0.60 (τ*) | 1.5 | 0.048 | 0.056 | 0.041 | 0.015 (−0.002 to 0.032) | 0.007 (−0.015 to 0.028) | **0.0668** |
  | 0.50 | 1.0 | 0.165 | 0.183 | 0.165 | 0.018 (0.004 to 0.033) | −0.0003 (−0.029 to 0.029) | **0.1001** |

## 5. Attackers and scenarios

- Command-stream attackers (K): Bayes Dirichlet–categorical target-blind and target-aware,
  first-order Markov blind and aware (α = 0.5), each scoring k consecutive commands of one
  user within one day (non-overlapping test windows; all k of a comparison score the same
  rows: those inside the largest k's windows).
- Full-feature attackers (L, fixed): logistic regression, RBF SVM, MLP on the unstandardized
  190-d P0 tangent vectors (day-1 Riemannian reference); for M2 on the 8-d P3
  representation.
- Scenarios, always reported separately (never a maximum over scenarios):
  **cal_to_dep** (primary) — enrol on day 1 (cross-fitted stream / day-1 features), test on
  days 2–3 (command streams: the two days pooled per user, windows within a day;
  full features: recall averaged over the two days); **dep_to_dep** — enrol on day 2, test
  on day 3 and the reverse (averaged); **same_day_halves** — enrol on trials 1–100 of day 2
  or 3, test on trials 101–n of the same day and the reverse (averaged; estimates only).
- Statistics per user, averaged within user over model seeds (and post-processing seeds),
  then over the 51 users: top-1 recall (N-way: argmax over all 51 users; within-group:
  argmax over the user's own group, group revealed; within-group excess = recall −
  1/|own group|) and the **percentile rank** of the true user (share of the other 50
  enrolled users whose score is below the true user's, ties counting one half; 0.5 =
  chance), averaged over the user's test windows.

## 6. Claims and decision rules

`T(v, a)`: one-sided one-sample t-test p-value that the mean of the 51 participant values
`v` exceeds `a` (50 df). Holm step-down at familywise **α = 0.05** over the claims whose
day-1 gate passed (gates in section 7). Claim-level p-values are computed exactly as in
`beat/exp_m.run_final` (`m1_test`, `m2_test`, `m3_test`). **Family: M1, M2, M3** (both
gates passed).

- **M1 — cross-day identification from full features.** cal_to_dep only; r_j = per-user
  recall of L attacker j (3 attackers). `p1 = min(1, 3 · min_j T(r_j, 10/51))`, i.e. the
  strongest attacker's mean recall exceeds 10/51 (Bonferroni over 3). The P0
  representation is the model with seed 0 (P0 is deterministic), and each attacker is
  fitted once, with seed 0, on the retained day-1 vectors (`beat.exp_m.m1_vectors`); a
  user's r_j is the mean of its day-2 and day-3 recall (each over that day's finite
  trials), so a user with no finite trial on one test day has r_j = NaN and is a missing
  cell (section 8). dep_to_dep is reported separately with the same statistic as an
  estimate. A low-risk replication estimate (within-day development values are far above
  10/N), reported as such.
- **M2 — independent attacker vs co-trained head (confirmatory; gate passed).** P3 λ = 1,
  cal_to_dep: b_j = per-user recall of L attacker j trained on the day-1 8-d P3
  representation, h = per-user recall of the co-trained identity head; gap_j = b_j − h
  (each the mean of the day-2 and day-3 recall, then over model seeds 0–2; a missing test
  day makes the user a missing cell as in M1). Only attackers whose **observed** mean gap
  is ≥ 0.10 are tested; the inferential test is **gap > 0**, one-sided, Bonferroni over the
  3 attackers: `p2 = min(1, 3 · min_{j: mean gap_j ≥ 0.10} T(gap_j, 0))`, p2 = 1 if no
  attacker reaches 0.10. A supported M2 therefore means "some independent attacker's gap
  is positive (inferential) and at least 0.10 (observed)"; it is never reported as a gap
  significantly ≥ 10 pp. Runtime (`dev_decoders.json` `p3_runtime`): one full day-1 P3
  fit 6.1 s (12 epochs, 10,198 trials, one torch thread); half-day fit 4.2 s.
- **M3 — command-stream identity grows with observation length (confirmatory; gate
  passed).** P2-RC d = 8 at τ* = 0.60, target-aware Bayes categorical attacker only,
  enrol on day 1, test on days 2–3 pooled, **K* = 50**, k_base = 1. ρ_K = per-user mean
  percentile rank at k = K (same test rows at both k).
  `p3 = max{ T(ρ_50, 0.5), T(ρ_50 − ρ_1, 0) }` (one-sample and paired one-sided t-tests).
  Sensitivity (not decisive): label-permutation p — the enrolled identities are permuted
  (10,000 permutations; the same permutation at both k), p = max over the two statistics of
  (1 + #{permuted ≥ observed}) / 10,001.

### Estimates (no pass/fail; 95 % t-intervals over the 51 users)

Reported for τ* = 0.60 and τ = 0.5, each scenario separately (cal_to_dep, dep_to_dep,
same_day_halves), every Bayes attacker, k = 20 and 50 (`final_estimates_*.csv`):
- **EQ (confusion, g = 3)**: within-group excess undefended and defended and their
  difference (undefended − defended); N-way top-1 recall under EQ and undefended; task
  statistics on the scenario's test streams — weighted utility (w = τ/(1 − τ)), net
  progress, wrong-move rate and hold rate, undefended and defended, their loss
  (undefended − defended) and the **privacy cost** of weighted utility and net progress
  (recalibration only − defended).
- **Recalibration only** (section 4): the same identification and task statistics (its
  identification shows what recalibration alone leaves exposed).
- **Randomized response** at q = 0.5 (on the undefended stream) and RR-matched (RR after
  recalibration, section 4): the same statistics, including the privacy cost.
- **M3 secondary estimates** (undefended P2-RC, both τ, all scenarios): ρ and top-1 N-way
  recall for all four attackers (blind and Markov included) along the k curve 1, 5, 10, 20,
  50; top-1 compared with 1/51 and 2/51 (one-sided t-test p-values, descriptive).
- M1 for dep_to_dep (statistic as M1) and, if a gate had failed, the demoted claim.

## 7. Gates and triviality checks (day-1 development data only, recorded before the seal)

Day 1 has no cross-day pair, so gates and checks use the within-day split (trials 1–100 vs
101–200, both directions, means over model seeds), the same decoders and attackers, and
groups from the enrolment half. Each claim's criterion applied to a null baseline must fail
(p > 0.05, unadjusted). Source: `results/exp_m/selection.json` (`gates`),
`results/exp_m/dev_identification_summary.csv`.

Gates (set by the Lead; day-1 results had been inspected, see the disclosure in section 2):
- **M2** is confirmatory only if its day-1 within-day gap (best attacker − head) ≥ 0.25 and
  its permuted-label null fails; otherwise descriptive.
- **M3** is confirmatory only if τ* lies in 0.58–0.62, the day-1 within-day rank at k = 50
  has t ≥ 4 (vs 0.5) and the user-independent null fails; otherwise M3 becomes an estimate
  and the family is M1 + M2.

| Claim | Gate / real day-1 result | Null baseline (must fail) | Outcome |
|---|---|---|---|
| M1 | recall logistic 0.989, SVM 0.981, MLP 0.971 (threshold 0.196); p = 2.7e-59 | identities permuted at enrolment: 0.018 / 0.011 / 0.020, p = 1.0 — **fails** | confirmatory |
| M2 | best gap 0.717 (MLP 0.769 − head 0.052; logistic 0.683, SVM 0.675) ≥ 0.25; p = 1.4e-33 | battery trained on permuted identities: best gap −0.036, p = 1.0 — **fails** | **gate passed → confirmatory** |
| M3 | τ* = 0.60 ∈ [0.58, 0.62]; ρ_50 = 0.674 (95 % CI 0.620–0.727), t = 6.52 ≥ 4; ρ_1 = 0.546, paired t = 5.05; p = 3.1e-6; permutation p = 1.0e-4 | user-independent stream (commands redrawn from the pooled P(output \| target)): ρ_50 = 0.506, p = 0.40 — **fails** | **gate passed → confirmatory** |

Secondary day-1 within-day figures (`dev_identification_summary.csv`, ρ_50 / top-1 at
k = 50): τ* = 0.60 — cat_blind 0.629 / 0.051, cat_aware 0.674 / 0.069, markov_blind
0.649 / 0.046, markov_aware 0.675 / 0.069 (2/51 = 0.039); τ = 0.5 — 0.422 / 0.018,
0.547 / 0.051, 0.481 / 0.038, 0.586 / 0.061. At τ = 0.5 the target-blind attackers rank
the true user below chance on day 1 (cat_blind t = −2.8 at k = 50, −5.0 at k = 1). The
independent pre-registration review traced this to P2-RC's per-user recentring under
leave-one-run-out cross-fitting: across users, the share of one output (left vs right) in
the first half of day 1 correlates −0.61 with its share in the second half for P2-RC
cross-fitted streams, against +0.21 for P0. The effect is confined to comparisons in
which both enrolment and test streams are cross-fitted within one day (the day-1 gates
and checks); it does not bias the day 1 → days 2–3 scenario, whose test streams come from
one model fitted on all of day 1.

**Power of M3** (rank-based; `results/exp_m/dev_m3_power.csv`): days 2–3 retain a share r
of the day-1 within-day excess (both vs 0.5 and vs k = 1), with the day-1 between-user SD
(0.190 and 0.180), 51 users resampled with replacement, 4,000 draws; power = P(p3 ≤ level).
The day-1 SD is conservative for the pooled test (8 windows of 50 per user instead of 2).

| Retention r | Excess vs 0.5 | Excess vs k = 1 | Power at α = 0.05 | Power at α/3 |
|---|---|---|---|---|
| 0.35 | 0.061 | 0.045 | 0.47 | 0.29 |
| 0.55 | 0.096 | 0.070 | 0.85 | 0.69 |
| 0.70 | 0.122 | 0.089 | 0.96 | 0.90 |

## 8. Exclusions, missing data and failures

- Trials: non-finite trials (guard above) are excluded from fitting and from streams; no
  other trial rejection. Participants are never excluded; the identity set stays 51 users.
- Missing participant cells (a user without a value in any vector a claim uses, e.g. no
  complete K*-window on the test days): up to **2 users** receive the chance value (rank
  0.5; recall 1/51; for M2 battery and head both 1/51, i.e. gap 0) in every vector of that
  claim; with more than 2 such users the claim is not evaluable (p = 1; the Holm family is
  not reduced). Counts are reported per claim.
- Missing user-days (section 3) are not imputed at the trial level: a user without a
  day-2 or day-3 recording keeps the other test day in pooled command-stream statistics
  (M3: missing only without any complete K*-window on days 2–3) and is a missing cell in
  M1 and M2 (two-day recall mean undefined); every claim reports its count.
- Computational failure or non-convergence: fixed-budget predictions are retained and
  reported; no reselection.
- Runtime: final ≤ 2 h on CPU (plan limit); exceeding it is reported, not used to change the
  analysis.
- **Crash and rerun policy.** Fetch (days 2–3): a transfer failure aborts the run; the same
  command is rerun, which skips user-days that already have a status line or reduced file
  and re-verifies any raw file left on disk against its recorded hash; every request,
  including repeated ones, is logged. A recording found absent or unparseable is not
  retried. Final: every attempt is appended to `results/exp_m/final_attempts.jsonl`
  (UTC start, failure with the error, completion). An attempt that fails before writing
  `decision.json` is rerun with the identical sealed command, code and data (same seeds);
  outputs of the failed attempt are overwritten. Once `decision.json` exists the final
  refuses to run again. A failure that needs a code or configuration change is a deviation
  (section 10); the change is made in a new commit, logged and reported with the reason.

## 9. Seal-4 contents and final-stage guard

Seal-4 files: this document, `configs/exp_m.yaml`, `configs/exp_k.yaml`,
`beat/{exp_m, yang2025, exp_k, common}.py`, `scripts/fetch_yang2025.py`,
`results/exp_m/selection.json`, `checksums/yang2025_day1.sha256`.
`selection.json` records `input_sha256`: the SHA-256 of every file in exp_m's static import
closure (`beat/__init__.py`, `attacks.py`, `common.py`, `config.py`, `data.py`, `exp_b.py`,
`exp_c.py`, `exp_g.py`, `exp_k.py`, `exp_l.py`, `exp_m.py`, `features.py`, `lee2019.py`,
`models.py`, `splits.py`, `stats.py`, `verify.py`, `yang2025.py`), of both configs and of the
fetch script, as they were when `streams` wrote it.

`python -m beat.exp_m final --seal-commit HASH --result-dir results/exp_m --artifact-dir
artifacts/exp_m` refuses unless:
- annotated tag `seal-4` names HASH and HASH is an ancestor of HEAD;
- the supplied config equals the sealed one and `thresholds_status` is `frozen`;
- every seal-4 file and every `beat` module in exp_m's import closure is byte-equal to the
  seal commit;
- every `input_sha256` entry of the sealed selection equals the SHA-256 of that file's
  bytes in the seal commit (a selection written with other code or configs, e.g. pre-freeze
  K code, fails), and the current files match it again before the analysis
  (`check_selection`);
- package versions equal those in `selection.json`; the config's hash, `tau_star` and the
  M2/M3 statuses equal the sealed day-1 selection and its gates; the cross-fitted day-1
  logits match their sealed SHA-256;
- `seals/manifest.yaml` has a `seal-4` entry naming HASH and `beat.verify.check_seals()`
  reports no error (no unsanctioned change to any sealed path of seals 1–4; working tree
  byte-equal to HEAD on every sealed path);
- the post-seal timestamp guard (section 2) verifies the RFC 3161 token for HASH;
- every day-1 reduced file has the SHA-256 recorded in `checksums/yang2025_day1.sha256`,
  and every day-2/3 reduced file present has the SHA-256 recorded in
  `checksums/yang2025_day23.sha256` when it was written, with both raw members of each
  present user-day recorded (`check_data`; absent day-2/3 files are missing user-days);
- output directories are given explicitly and equal the sealed paths, and no completed
  `decision.json` exists (section 8).

## 10. Deviation policy

Any change after seal 4 is logged in `docs/decision_log.md` with reason, date and affected
claims, and the original sealed analysis is reported alongside. No threshold, candidate set,
condition or τ changes after the seal. Outcomes are reported whatever they are.

## 11. Post-freeze rerun (done)

The final stage refuses to run unless `thresholds_status` is `frozen` in the sealed
commit. Only quantities computed with K's EQ depended on K's freeze: the day-1 utilities and
the RR-matched q (section 4) and the EQ estimates (section 6, computed in the final). τ*,
the gates, the confirmatory statistics and the power table use K's stream primitives (Bayes
scores, windows, groups) but not its EQ. Steps performed:

1. K frozen after re-review: `beat/exp_k.py`, `beat/common.py`, `configs/exp_k.yaml`
   (seal-4 files and in exp_m's import closure).
2. `python -m beat.exp_m streams` with the frozen K code (decoders not refitted — the stage
   checks the decoder configuration hash; it re-derived τ* = 0.60 and the gates, unchanged).
3. RR-matched q values in section 4 updated from that `selection.json`;
   `thresholds_status: frozen`; `streams` re-run so that `results/exp_m/selection.json`
   carries the frozen configuration hash and the input hashes (`input_sha256`). No file in
   `input_sha256` may change before the seal commit; a change requires another `streams`
   run.

## 12. Day-1 identity results inspected before freezing

Recorded so that the inspection disclosed in section 2 can be traced. All are day-1,
within-day (trials 1–100 vs 101–200), cross-fitted P2-RC d = 8 streams unless stated.
- First `streams` runs (τ = 0.5 and 0.8; M3 draft with top-1 N-way recall at K* = 20; M4
  draft with the within-group bound; four Bayes attackers, k = 1–50), first with groups
  formed on both halves, then with groups from the enrolment half. Their tables
  (`results/exp_m/dev_identification.csv`, `dev_m4_bound.csv` and the τ = 0.5/0.8 version
  of `dev_identification_summary.csv`) were overwritten or deleted on 2026-11-11 when the
  current rule replaced them; their decision-relevant values, from the previous draft of
  this document: hold rate 99.3 % at τ = 0.8; M3 draft at τ = 0.8, K* = 20 — strongest
  top-1 0.033 (k = 1: 0.021; 2/51 = 0.039), p = 1.0; user-independent null strongest
  0.022, p = 1.0; M4 draft at τ = 0.8, g = 3, k = 20 — undefended strongest within-group
  excess 0.008, bound p = 0.011 at m = 0.05 (bound failed only for m ≤ 0.03 at g = 3 and
  m ≤ 0.05 at g = 2 or 5), EQ preview strongest excess 0.022, loss 0.8 pp; τ = 0.5 —
  top-1 N-way recall at k = 20 up to 0.057 (Markov target-aware).
- Current rule (τ* = 0.60 and τ = 0.5): `results/exp_m/dev_identification_summary.csv`
  (rank and top-1 for four attackers at k = 1, 5, 10, 20, 50, within-group excess at
  g = 3), `selection.json` (`gates`, `within_day_identification`, `m3_power`).
- The τ grid (`dev_tau_grid.csv`, 0.50–0.90) holds hold rates only; lane M computed no
  identity statistic at other grid values. Identity results examined by the Lead's
  advisor and L's cross-day rank figures (Lee2019) are not lane-M files; the Lead records
  their location in `docs/decision_log.md`.
