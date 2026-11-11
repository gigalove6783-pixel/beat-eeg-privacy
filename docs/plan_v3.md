# BEAT v3 plan: from an audit study to an audit tool

Status file for v3. Lead owns this file; stage results are appended under "State". Deviations go to
`docs/decision_log.md` (append-only), not here.

## 1. Task contract

**Request (2026-11-11).** Assess BEAT critically and make it useful in practice, drawing on methods
and literature from other fields and testing them; the result must work as a portfolio and in the
2026-11-16 talk.

**User decisions.** Audience: the 11/16 results talk and graduate-school / lab applications.
Direction: audit tool first, then a working BCI. Talk: revise to v3 by the talk gate
(2026-11-14 18:00 KST).

**Diagnosis that v3 answers** (evidence in README.md and `results/`):

1. Risk is not stated in units a builder can use. Every number is closed-set identification among
   enrolled users (9, 54, 51). Linking two recordings of a person the attacker never enrolled, and
   scaling beyond the cohort, were not measured.
2. The audit procedure exists only as experiment code (`beat/exp_k.py`, `beat/exp_m.py`); nobody
   can run it on their own BCI outputs.
3. The decoder of study M is weak (BA 0.544–0.555), and EQ raises the hold rate to 0.83, so the
   utility side of the trade-off does not describe a usable BCI.
4. The first screen of the README is caveats; purpose and use are not visible.

**Outcome.** (a) `beat_audit`: a dataset-agnostic library and CLI that takes what a BCI transmits
(features or commands with user and day labels) and writes a standard leakage report. (b) The tool
validated on synthetic ground truth, against BEAT's sealed numbers, and run on the three cohorts,
the third under a freeze. (c) A "what to transmit" ladder with per-user decoders (exploratory).
(d) README, report, explainer, demo and a v3 talk deck built around the tool.

**Non-goals.** No new dataset download. No edit to any sealed file (`seals/manifest.yaml` scope).
No change to any v1/v2 verdict, tier label or registered number. No claim of anonymity or safety.
No push, publication or release unless the user asks.

## 2. Methods imported from other fields

Each method enters the tool only with a validation test (§5). References are checked with
`scripts/check_references.py` before they are cited in any document.

| Field | Method | Use in the tool |
|---|---|---|
| Biometric performance testing (ISO/IEC 19795-1) | Open-set verification: mated / non-mated comparison scores, FMR, FNMR, EER, DET | Linkage of two windows of outputs from users the attacker never enrolled |
| Biometric template protection (ISO/IEC 24745; Gomez-Barrero et al. 2018) | Unlinkability measure D_sys in [0, 1] from mated / non-mated score densities | One-number linkability of a transmitted stream |
| Speaker anonymisation (VoicePrivacy evaluation plans) | Attackers graded by knowledge; a weak attacker overstates protection | Attacker grid: enrolment (none / reference day / deployment day) × day relation. Stated as an analogy, not as the VoicePrivacy protocol |
| Forensic likelihood ratios (Brümmer & du Preez 2006; Ramos & Gonzalez-Rodriguez 2013) | Calibrated LR, Cllr and Cllr_min | Evidence per comparison in bits; calibration loss across days |
| Sequential analysis (Wald 1945) | SPRT with nominal error rates | Number of commands until a link decision; realised versus nominal error |
| Large-population biometrics (Grother & Phillips 2004) | Rank-1 rate at gallery size N under the independence model, P_N = ∫ F_nonmated(s)^(N−1) dF_mated(s) | Checked against measured 1:N rates of the same comparator on held-out galleries; larger N shown only as an assumption-dependent scenario |
| Bayesian two-sample test (Dirichlet-multinomial Bayes factor) | Same-source versus different-source evidence for two count vectors | Training-free comparator for command streams |
| Regulation | UNESCO Recommendation on the Ethics of Neurotechnology (2025), Colorado HB24-1058, California SB 1223, GDPR Recital 26, Korean pseudonymisation guideline | Report section that maps each metric to the "reasonably likely means" question; no legal conclusion |

Stretch, not gated: empirical lower bound on a claimed ε of a randomising defence (Nasr et al.
2021) validated against the analytic ε of randomized response.

## 3. Tool specification (`beat_audit/`)

**Constraints.** New top-level package; imports numpy, scipy, scikit-learn, pandas, matplotlib and
PyYAML only (no torch, no `beat.exp_*`). Small scoring routines are re-implemented instead of
imported from `beat.exp_k` (which loads torch through `beat.exp_c`); a parity test compares them
with `beat.exp_k.bayes_scores` on fixed inputs.

**Input** (`.npz` or `.csv`; `beat_audit.io`):

| Field | Type | Meaning |
|---|---|---|
| `outputs` | float [N, d] or int [N] | Transmitted features, or command codes 0 … C−1 |
| `user` | [N] | Pseudonymous user label |
| `day` | [N] | Recording day or session label |
| `segment` | [N], optional | Disjoint parts of one day (runs, phases); default: halves by order |
| `order` | [N], optional | Position in time within user and day; default: file order |
| `target` | [N], optional | Cue or intended class, if the attacker may see it |

**Protocol.**

- Command codes are 0 … C−1 with C declared in the input (default: largest code + 1). BEAT's
  stored codes −1 = left, 0 = hold, +1 = right are mapped {−1: 1, 0: 0, +1: 2} at export, C = 3.
- Segments are disjoint parts of one day; default: halves by order. Cohorts: 2a runs 1–3 and
  4–6; Lee2019 offline and online phase; Yang2025 trials 1–100 and 101–200 of each day.
- Windows: k consecutive outputs of one user inside one segment, non-overlapping,
  k ∈ {1, 5, 10, 20, 50, 100}. Eligibility, one rule for every analysis: a user enters a cell
  (stream, condition, k) only with at least one window in every segment the condition needs;
  users without it are excluded from that cell and listed; a cell with more than 10% of its
  users excluded is *not evaluable*.
- User folds: G = min(5, ⌊U/3⌋) user-disjoint folds from a seeded permutation of the sorted user
  labels (9 users: 3 folds of 3; 51 and 54 users: 5 folds of 10–11). Fewer than 6 users is not
  evaluable. Every comparator, prior, calibration map and threshold is fitted on a fold's
  development users only, and both users of every scored pair are target users of that fold.
  This supports a claim about users unseen by the attacker's comparator, not about users unseen
  by the BCI: BEAT's encoders and recentring were fitted with these users' calibration data.
- Comparisons are directional: a reference window of user u against a probe window. Mated: the
  probe is from u. Non-mated: the probe is from another target user of the fold. *Same-day*:
  reference and probe come from different segments of one day. *Cross-day*: from different days.
  Each condition has two physical sides, A and B (same-day: the first and the second segment of
  each day; cross-day: the earlier day and the later day or days). Direction A→B takes
  references from A and probes from B, direction B→A the reverse; both are scored and averaged
  within user.
- Bounds: per target user and condition, every mated comparison up to 200 and at most 50
  non-mated comparisons per partner, drawn with a fixed seed. The learned comparator is trained
  on at most 20,000 development pairs.
- Feature comparators, all reported, the strongest named: cosine on the standardised window
  summary; LDA + within-class covariance normalisation + cosine; pairwise logistic regression on
  absolute differences of window means. They use no data of target users. Two window summaries
  (plan change of 2026-11-11, see State): *mean*, the mean output; *rich*, the mean, the mean
  per cue when `target` is given (a cue absent from the window uses the window mean) and the
  log of the per-dimension variance (variance floored; k ≥ 5 only). Cosine and LDA run on both
  summaries (`cosine`, `lda_cosine`, `cosine_rich`, `lda_cosine_rich`); at k = 1 the rich
  summary equals the mean and is skipped.
- Command comparator: the same-source versus different-source Bayes factor of two count vectors
  x (reference) and y (probe) under a shared Dirichlet prior,
  BF = B(α + x + y) · B(α) / (B(α + x) · B(α + y)), B the multivariate beta function. The prior is
  α = α0 · π, always positive: π = (pooled development counts + 0.5) / (total + 0.5 · C) over the
  declared alphabet, and α0 maximises the Dirichlet-multinomial marginal likelihood of the
  development users' windows over 41 log-spaced values in [0.1, 1000]; an optimum at either end
  is accepted and flagged. With `target`, counts and prior are kept per cue and the log factors
  are added; a cue absent from a window adds 0. The same-source hypothesis assumes one command
  distribution in both windows, so a drift-aware attacker may find more than this comparator.
- Closed-set identification with enrolment (BEAT's v1/v2 measurement: logistic regression,
  RBF-SVM, MLP on features; enrolled Dirichlet-categorical predictive likelihood on commands) is
  kept as a second module on the same windows.

**Metrics and inference.** No inference treats pairs as independent.

- Statistic: AUC_{u,v}, the probability that a mated score of reference user u exceeds a
  non-mated score of u against partner v (ties count 0.5). AUC_u is its mean over the partners
  of u's fold, then over both directions, days and model seeds, all with equal weights; the
  reported statistic is the mean over users, each user weight 1. This is the open-set form of
  M3's rank.
- Test of linkage (mean AUC > 0.5), primary and only deciding test: a matching permutation test.
  Within each fold one matching π is drawn per draw: the side-B blocks are reassigned to users
  as whole blocks (all side-B segments or days of a user move together; the same π for every
  model seed). Direction A→B then treats A_i and B_π(i) as mated, and direction B→A is derived
  from the same matching (B_π(i) as reference, A_i as its mated probe); side A is never
  reassigned separately. The comparators stay as fitted and the statistic is recomputed; 10,000
  draws, or all matchings when fewer exist; p = (1 + #{draws ≥ observed}) / (1 + draws).
  Assumption: with no identity information, side-B blocks are exchangeable among the target
  users of a fold.
- Paired contrast (same-day minus cross-day): d_u = AUC_u(same-day) − AUC_u(cross-day), reported
  as the mean over users with its interval. It is an estimate; no verdict rests on it.
- Intervals: subject bootstrap, 2,000 replicates, 95% percentile. Target users are drawn with
  replacement within each fold. Each drawn copy of user u contributes AUC_u recomputed as the
  mean of AUC_{u,v} over the other distinct users v of the replicate, each weighted by the
  number of its copies; copies of one user are never paired with each other. A fold with fewer
  than two distinct users in a replicate is left out of that replicate, and a replicate with no
  fold left is redrawn. Intervals are conditional on the fitted comparators; with folds of three
  users (2a) they are printed as unreliable.
- Pooled-score metrics (EER, TPR at FPR 1% and 10%, D_sys, Cllr) pool the target scores of a
  cell over folds, directions, days and seeds, weighted so that every target user carries the
  same total weight among mated and among non-mated scores. EER and TPR at fixed FPR are
  descriptive. Deployable operating point: the threshold at the development users' EER, applied
  unchanged to target users, with realised FMR and FNMR.
- Calibration: a one-dimensional logistic regression of mated against non-mated on the
  development-fold score, classes weighted to equal totals. Its logit ℓ is the *estimated* log
  likelihood ratio (natural log), valid only as far as the logistic form and the development
  users transfer. Cllr = ½ [E_mated log2(1 + e^(−ℓ)) + E_nonmated log2(1 + e^(ℓ))] on the target
  scores. Cllr_min replaces ℓ by weighted isotonic (PAV) calibration fitted on the target scores
  themselves; it is an oracle diagnostic and feeds nothing back into the attacker.
- D_sys (Gomez-Barrero et al. 2018): D(s) = 2ωLR(s) / (1 + ωLR(s)) − 1 where ωLR(s) > 1, else 0,
  ω = 1; D_sys is the integral of D(s) against the mated score density. LR(s) is the ratio of
  the weighted histograms on 50 equal-width bins over the pooled score range (one bin per value
  when a score takes at most 50 values). A bin with mated but no non-mated mass has D = 1; a bin
  without mated mass contributes nothing. It is a linkability index, not an identification rate.
- Curve over k, with the smallest k at which linkage is detected (Holm over the k grid).
- Commands only, sequential attacker: reference counts x from the reference segment or day (up
  to 100 commands); probe commands y_1 … y_n in time order. Same-source hypothesis: y_t follows
  the Dirichlet-categorical predictive given α + x + counts(y_<t). Different-source hypothesis:
  the predictive given α + counts(y_<t). The cumulative log ratio equals log BF(x, y_1..n), so
  each command is counted once. Decide "link" at log((1 − β)/α), "no link" at log(β/(1 − α)),
  α = β = 0.05, truncated at the probe length. Reported: share decided, median commands to
  decision, realised FMR (non-mated streams decided "link" over all non-mated streams) and FNMR
  (mated decided "no link" over all mated), undecided streams separately. Under the generative
  model both error rates are bounded by α/(1 − β) and β/(1 − α) (Ville's inequality); on real
  streams, with drift and dependent commands, they are operating characteristics, not guarantees.
- Gallery module. Ties are broken uniformly at random, in the prediction and in the measurement.
  Prediction for a mated score s: Σ_{j=0}^{N−1} C(N−1, j) a^(N−1−j) b^j / (j + 1), with
  a = P(non-mated < s) and b = P(non-mated = s), averaged over mated scores with the pooled-score
  weights, per fold comparator and then over folds. Measurement: the 1:N rank-1 rate of the same
  comparator, windows and users on held-out fold galleries, N = 2 … fold size. The check is a
  diagnostic tolerance, not a test: if any |predicted − measured| exceeds 0.05, the report prints
  "independence model not supported" and no larger N. Otherwise larger N is printed up to one
  tenth of the number of non-mated scores, labelled as a sensitivity scenario under the
  independence assumption, never as a population risk estimate.
- Report wording per condition and k (conventions, never "safe"): *linkage detected* with the
  smallest k, or *not detected: insufficient evidence, not absence*, with the interval. A
  detection in any condition is never overridden by a non-detection elsewhere.

**Guards from BEAT's findings.** The tool takes transmitted outputs only and computes its own
independent attacks; it has no input for attacker scores supplied by the builder, so a co-trained
head's accuracy cannot enter a report. It requires day labels, always prints same-day beside
cross-day, warns below 20 users and flags a mean AUC below 0.5. Data-processing guard in
`compare`: commands are a function of the features they are decoded from, so the best feature
attacker cannot do worse than the best command attacker. When the builder declares a command
rung as derived from a feature rung and the command rung's mean AUC is the higher one at the
same contrast and k, the table says that the feature attacker is too weak and that the feature
rung's exposure is at least the command rung's.

**Outputs.** `report.json` (all numbers), `report.md` (tables, plain-language reading,
limitations), PNG figures (curve over k, DET, score histograms, gallery extrapolation).
CLI: `beat-audit run <input> --out <dir> [--kind features|commands] [--seconds-per-output s]`.

## 4. Stages, owners and gates

| Stage | Work | Owner (files) | Gate |
|---|---|---|---|
| S1 | Library, CLI, synthetic validation, example input | core implementer: `beat_audit/**`, `tests/test_audit*.py`, `examples/**`, `pyproject.toml`, `requirements.txt` | §5 A1–A2 |
| S2 | Cohort adapters: export every rung of 2a, Lee2019 and Yang2025 day 1 to the input schema; parity run; development runs on 2a and Lee2019 | cohort implementer: `scripts/audit_cohorts.py`, `configs/audit.yaml`, `results/audit/**`, `artifacts/audit/**` | §5 A3 |
| S3 | Freeze (tool, config, job manifest, predictions V1 and V2), then one frozen Yang2025 batch including days 2–3; independent recomputation | cohort implementer; reviewer | §5 A4–A5 |
| S4 | Per-user decoders and the transmission ladder (exploratory) | cohort implementer: `scripts/audit_ladder.py`, `results/audit/ladder*`; frozen files are not edited | §5 A10 |
| S5 | README, report §6.11, related work §8, explainer, numbers registry, demo panel | writer: `README.md`, `docs/report.md`, `docs/related_work.md`, `docs/explainer_ko.md`, `docs/numbers.yaml`; demo owner: `demo/**`, `scripts/export_demo_data.py` | §5 A6–A7 |
| S6 | Talk deck v3 (Korean, notes) | deck owner: `presentation/BEAT_v3_talk.*` | §5 A8 |
| S7 | Cross-document review, CI checks | reviewer; Lead | §5 A6, A9 |

**Schedule (KST).** S1 and the S2 exports run in parallel (the input schema is the interface) and
end by 11/12 12:00. Development runs, runtime gate and freeze: 11/12 24:00. S3 run and audit:
cut-off 11/13 12:00, when the Lead chooses the full or the fallback deck. The deck skeleton
(problem, tool, method, 2a/Lee2019 exploratory numbers) and the README draft start when the
Lee2019 development run is complete and do not wait for S3. If the Lee2019 development run is not complete by 11/13 00:00, the
deck is built from the tool, its synthetic validation and the v2 results alone. S5 and S6 end by
11/14 12:00; 11/14 12:00–18:00 is reserved for S7. S4 belongs to the outcome but not to the talk
gate: it ends by 11/15 12:00 within 2 h of compute, its numbers enter the deck only if audited
by 11/13 18:00, and the task is not complete until S4 is done.

**Runtime gate (before the freeze).** A pilot on one Lee2019 fold projects the time and peak
memory of the whole job manifest (all cohorts, rungs, seeds, conditions and k). Limits: 60 min
and 8 GB per stream (30 min until the plan change of 2026-11-11), 3 h for the Yang2025 batch, 4 threads per job (BLAS and OpenMP thread
variables set), at most two jobs at once. If the projection exceeds a limit, reductions are
applied in this order and logged: drop the pairwise-logistic comparator; drop the gallery
module; drop k = 1; 2,000 permutations instead of 10,000. If the limit is still exceeded, the
stage stops, the blocker is reported and the fallback deck applies.

**Rungs exported per cohort.** Full tangent features (P0), recentred 8-d bottleneck (P2-RC d = 8),
commands of P2-RC at the cohort's τ values.

- 2a and Lee2019: features from `artifacts/exp_b/final/` and `artifacts/exp_l/`; commands at
  τ = 0.5 and 0.8 from `artifacts/exp_k/command_records.csv.gz`, calibration day cross-fitted
  (`cal_oof`), deployment day `dep`. Days: 2a T/E; Lee2019 sessions 1/2.
- Yang2025, τ = 0.60 and 0.5. P0 features on all three days (tangent reference fitted on day 1,
  as in M1). P2-RC 8-d features on days 2–3 only, because day 1 is the encoder's training day.
  Commands: day 1 from M's cross-fitted logits (`artifacts/exp_m/oof_*`), days 2–3 from the
  stored final models (`artifacts/exp_m/final/`) applied to covariances recomputed from
  `data/processed/yang2025/`. Sealed code is imported unedited; `beat.exp_m.run_final` is not
  called, because it retrains and writes results.

**Freeze (S3).** Yang2025 was analysed in study M and its results shaped this project, so the run
is not blind and is not a confirmatory replication. Its tier is *pre-specified (non-blind)*:
fixed before this computation, run once. Exports of all three cohorts are produced before the
freeze; `beat_audit` is not run on any Yang2025 export before it (`results/audit/attempts.jsonl`
logs every tool run; the log is self-attested). The freeze bundle `results/audit/freeze/` holds:

- a zip of the source bytes (`beat_audit/`, `scripts/audit_cohorts.py`, `configs/audit.yaml`,
  this plan) and their SHA-256;
- `pip freeze` output;
- SHA-256 of every exported input, of the 153 Yang2025 archives, of the nine final models and of
  the day-1 cross-fitted files;
- the Yang2025 fold assignment, every sampling seed, and the comparator and k named for each
  prediction from the Lee2019 development run;
- `jobs.json`, the manifest of every expected cell: cohort, rung, model seed, condition, day
  pair and direction, k, comparator.

The freeze is a Git commit with tag `freeze-5`, pushed before the Yang2025 batch.

**Attempts.** `results/audit/attempts.jsonl` gives every execution a kind: `development` (2a,
Lee2019, synthetic), `frozen_batch` (one command that runs every Yang2025 cell of `jobs.json`),
`technical_rerun` (the identical batch after a failure, same bundle hashes), `recomputation`
(independent audit) and `exploratory_s4`. The first completed `frozen_batch` or
`technical_rerun` is the pre-specified result. A run after a code or config change is a
deviation, is reported as exploratory and never replaces it. Each cell ends as `ok` (with its
excluded users), `not_evaluable` (with the reason) or `failed`; any `failed` cell makes the
batch a technical failure.

Predictions on Yang2025, a Holm family of two at α = 0.05. A prediction whose cell is not
evaluable stays in the family with p = 1.

- V1. Full features, cross-day (day 1 against days 2–3), pre-named comparator and k: mean
  per-user AUC exceeds 0.5 (matching permutation test).
- V2. Commands of P2-RC at τ* = 0.60, k = 50, cue-aware Bayes factor, cross-day (cross-fitted
  day 1 against days 2–3): mean per-user AUC exceeds 0.5 (matching permutation test).

Pre-specified estimate without a verdict: the paired contrast of full features, same-day (halves
of day 2 and of day 3) minus cross-day (day 1 against days 2–3), same comparator and k as V1.
Everything else from the Yang2025 batch is an estimate without a verdict. All 2a and Lee2019
audit numbers are *exploratory*.

**S4 protocol (exploratory).** Per-user decoder: standardised logistic regression (C = 1) on the
user's own calibration-day P0 features (2a session T, Lee2019 session 1, Yang2025 day 1),
evaluated on the later day or days; the scaler and the model are fitted on calibration-day trials
only. Commands by a maximum-probability threshold τ ∈ {0.5, 0.6, 0.8}. Utility over all
later-day trials of a user: task BA (mean of the two class recalls of the argmax decision), and
hold rate, correct-move rate and wrong-move rate with all trials as denominator.

One contrast per cohort for every rung, so that rungs differ only in what is transmitted, and
only days that no encoder or decoder was fitted on:

- Yang2025: same-day = halves of day 2 and of day 3; cross-day = day 2 against day 3.
- Lee2019: same-day = offline against online phase of session 2. 2a: same-day = runs 1–3
  against 4–6 of session E. No out-of-sample cross-day contrast exists for these two cohorts.

One population per cohort: before any ladder row is scored, the users eligible in every rung at
the row's condition and k form the common population. Every rung is scored on that population
with one fold assignment, and each report records a hash of its user list, masks and folds.

`ladder.csv` has, per cohort, the rungs full features, 8-d bottleneck, shared-decoder commands
and per-user-decoder commands, each with its utility figures and the mean AUC with interval at
k = 20, plus the cohort, days, condition, τ, seeds, k and population hash of the row. Contrasts
that involve a calibration day go to separate, labelled columns; there the newly fitted per-user
decoders are cross-fitted in 5 stratified folds with a fixed seed, and the shared decoder keeps
the cross-fitting of its study. No usefulness threshold is claimed, and "no preferable alternative demonstrated" is an
admissible result.

## 5. Acceptance checks

- A1. In a fresh virtual environment, from a working directory outside the repository, the
  installed `beat-audit run` on the command example and on the feature example writes
  `report.json`, `report.md` and figures. `import beat_audit` leaves `torch` out of
  `sys.modules`. Invalid inputs (missing day labels, one day only, fewer than 6 users, a k with
  no window, non-integer commands) end with a message that names the field and the fix.
- A2. Synthetic validation, two lanes. CI lane in `tests/test_audit*.py`: deterministic seeds, at
  most 5 min in total. Full lane: `python -m beat_audit.validate` writes
  `results/audit/validation.json` in at most 60 min. Both state their Monte-Carlo tolerance
  (binomial 99% bound for the number of simulations used).
  - Analytic unit tests on score distributions: equal-variance Gaussian scores give
    EER = Φ(−d′/2); D_sys is 0 for identical and 1 for disjoint distributions and matches
    numerical integration of its definition for Gaussian scores; the Bayes factor equals
    brute-force integration for small counts, is symmetric in x and y, and equals the ratio of
    the posterior predictive of y given x to its prior predictive; the calibrated output is a
    log likelihood ratio on data with known likelihood ratios; P_N matches simulation under
    independent scores.
  - Whole-procedure tests (fitting, folds, pairing, seed averaging, inference) at cohort shapes
    9 × 2 days, 54 × 2 days and 51 × 3 days, 1,000 simulations each in the full lane. With no
    identity information the permutation test rejects at or below nominal. At 54 × 2 and 51 × 3,
    with heterogeneous users, the interval of the mean AUC under a planted effect and the
    interval of the paired contrast under strong linkage with zero contrast each cover the
    large-sample value in 90–99% of simulations; at 9 × 2 coverage is reported, not gated.
  - Sequential attacker: under the generative model both realised error rates stay below the
    Ville bounds. Under a planted drift (each user's day-2 command probabilities are an equal
    mixture of their own and an independent population draw) the mated log Bayes factor falls
    and FNMR rises against the no-drift case; under Markov-dependent commands the realised rates
    are reported without an assertion.
  - Edge cases: folds of three users, all scores tied (gallery prediction 1/N), a command or a
    cue that never occurs, α0 at a grid end, a user without a window, a cell above 10% exclusions.
- A3. Parity of the closed-set module with BEAT, on the original masks, directions, seeds and
  attacker settings: Lee2019 full features with the logistic attacker give 0.994 within-day and
  0.520 cross-day, each within 0.01; enrolled command likelihoods equal
  `beat.exp_k.bayes_scores` on fixed inputs. Recorded in `results/audit/parity.json`. Parity
  validates the closed-set module only.
- A4. `results/audit/{bnci2014_001,lee2019,yang2025}/**/report.json`, `results/audit/summary.csv`,
  `results/audit/yang2025/decision.json` (V1, V2). Every cell of `jobs.json` is reconciled with
  a status in the results; the hashes of the executed source, config and inputs recorded in the
  batch equal the freeze bundle; `results/audit/attempts.jsonl` shows no execution on Yang2025
  before the freeze and names the attempt that produced the result.
- A5. Independent review with its own code: every statistic that drives V1 and V2 and their
  decisions; on one feature stream and one command stream per cohort, the defining calculation
  of each imported method (per-user AUC, EER, D_sys, Cllr, Bayes factor, sequential decision,
  P_N unless the runtime gate dropped the gallery module), equal within 1e-6 or within the
  stated Monte-Carlo tolerance; freeze compliance. PASS.
- A6. `ruff check .`, `python -m beat.verify` and `python -m pytest -m "not data"` pass. New
  numbers in README, report, explainer and the deck's text source are registered in
  `docs/numbers.yaml`; a reviewer checks that no new number is unregistered and that the PPTX
  and PDF carry the numbers of the text source. No v1/v2 verdict, tier or registered value
  changes (diff of `docs/numbers.yaml` shows additions and moved snippets only).
- A7. README first screen: what the tool is, a three-line quickstart, one risk table in outputs
  observed (and minutes where the protocol gives a trial period), tier labels including
  *pre-specified (non-blind)*; the study remains below, unchanged in content.
- A8. `presentation/BEAT_v3_talk.pptx` and `.pdf` in Korean, at most 13 main slides plus backups,
  speaker notes with a time budget per slide that sums to the v2 talk's length; rendered pages
  checked; the demo page with the audit panel loads and its controls respond. A spoken rehearsal
  is the presenter's and is listed as open.
- A9. Decision-log entries for the v3 scope, the freeze, the run, the audit and any deviation.
- A10. Builder-facing case, reproducible from the README: `beat-audit compare` merges the reports
  of the transmission alternatives of one cohort with their utility figures into one table
  (task BA, hold rate, wrong-move rate, same-day and cross-day linkage under the open-set
  attacker), and the README states which choice the table supports, for which attacker, and what
  it does not show. `compare` refuses reports whose cohort, days, condition, k or population hash differ;
  a test feeds it two reports with equal metadata and different analysed users and expects the
  refusal. Produced
  for Lee2019 and Yang2025 from `results/audit/ladder.csv`; a reviewer follows the README case
  from a clean checkout and confirms the table and its reading.

## 6. Limits and stop conditions

- Talk gate 2026-11-14 18:00 KST. If S3 is not audited by its cut-off (11/13 12:00), the deck
  presents the tool and the 2a/Lee2019 exploratory numbers only and says that the Yang2025 run is
  pending.
- At most two failed repair cycles per stage, then one discriminating check or advice, then the
  blocker is reported.
- One frozen Yang2025 batch (§4 Attempts). An identical rerun after a technical failure is
  allowed and logged; a run after a code or config change is a deviation and is exploratory.
- 8-thread CPU, 31 GB RAM, 30 GB free disk: at most two compute jobs at once, 4 threads each.

## 7. Open decisions

- Resolved 2026-11-11: the user authorised local commits, the `freeze-5` tag and pushes to the
  private origin. Commits are made per stage; the freeze is a commit with tag `freeze-5`, pushed
  before the Yang2025 run.

## State

- 2026-11-11: plan drafted; procedure hf-gen-plan, review round 0 of 3.
- 2026-11-11: review round 1 NEEDS_CHANGES (8 blockers). Accepted and repaired: fold and segment
  rules; matching permutation and subject-bootstrap inference; gallery module restricted to a
  checked scenario; explicit Bayes factor and sequential attacker; score-level versus
  whole-procedure validation; freeze bundle and non-blind wording; schedule cut-offs and runtime
  gate; builder-facing acceptance A10 and S4 protocol.
- 2026-11-11: review round 2 NEEDS_CHANGES (9 blockers). Accepted and repaired: the paired
  contrast is an estimate, so the verdict family is V1 and V2 (former V3); subject-bootstrap
  rules; positive Dirichlet prior and code map; estimated log LR, Cllr formula, pooled-score
  weights and D_sys bins; gallery tie rule; one eligibility rule and a job manifest; attempt
  kinds; one contrast per cohort in the ladder; guard narrowed to what the input can enforce;
  whole-manifest runtime projection with a terminal stop. 
- 2026-11-11: review round 3 NEEDS_CHANGES, two blockers, every other criterion without
  objection. Both corrections applied as the reviewer stated them: one matching per permutation
  draw with the reverse direction derived from it (the reviewer's simulation gave 10.55% false
  rejections for independent per-direction reassignment against 5.20% for one matching); one
  common population, fold assignment and population hash per cohort ladder, enforced by
  `compare`. Review cap reached; no open blocker. The plan is accepted for implementation; the
  A2 null test exercises the bidirectional permutation as specified.
- Procedure from here: hf-rlcr on this plan. Stage S1 and S2 exports start 2026-11-11.
- 2026-11-11: S2 exports committed (16fc648) after review and two repairs. S1 reviewed:
  numerical core confirmed, four protocol and reporting blockers in repair (repair round 1).
- 2026-11-11: Lee2019 and 2a development run (exploratory, tool before repair). Two plan changes
  by the Lead, both before any Yang2025 execution:
  1. Feature comparators gain the *rich* window summary and `compare` gains the data-processing
     guard. Reason: on Lee2019 cross-day at k = 20 the 8-d bottleneck gave mean AUC 0.542 with
     the mean-based comparators while the commands decoded from it gave 0.630–0.677 with the
     cue-aware Bayes factor. Commands are a function of those features, so the feature attacker
     was too weak; reporting 0.542 as the bottleneck's exposure would repeat the weak-attacker
     error that BEAT warns against. The change strengthens the attacker; no criterion is relaxed.
  2. The per-stream runtime limit is 60 min instead of 30. Reason: the 30-min limit dropped the
     pairwise-logistic comparator, the gallery module and k = 1 although the whole Yang2025 batch
     was projected at 1.15 h against its 3 h limit; the batch limit is what protects the talk
     gate. Parity, pilot, development runs, V1 selection and the job manifest are rerun on the
     repaired tool; V1's comparator is named again by the same rule over all feature comparators.
