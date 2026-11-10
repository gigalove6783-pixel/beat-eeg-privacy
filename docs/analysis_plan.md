# BEAT analysis plan

Version 1, adapted 2026-11-10 from `BEAT 프로젝트 기획안.pdf` (2026-10-09) for a
results presentation on 2026-11-16. This file, `configs/` and the code are
committed before any session-E evaluation; the commit hash is recorded in
`docs/decision_log.md` ("E seal"). Session-E results never drive re-selection of
models, λ, τ, thresholds or architecture. Failures judged by these rules are
reported as results.

## Question

Can identity cues in transmitted motor-imagery (MI) EEG representations be
reduced beyond per-subject Riemannian recentering while limiting the loss in
left/right MI decoding to 5 percentage points (pp)? How much identity leaks
from command-only transmission as the number of observed commands grows?

## Data and preprocessing

- BNCI2014-001 / BCI Competition IV 2a: 9 subjects, sessions T and E recorded
  on different days, 22 EEG channels (3 EOG excluded), 250 Hz. Files are the
  BNCI `.mat` release fetched by `scripts/download_data.py`.
- Left vs right hand trials only; 6 MI runs × 24 trials per session before
  exclusion.
- Analysis window: 1–4 s after cue onset (cue at 2 s after trial start).
- Band-pass 8–30 Hz, 4th-order Butterworth, causal (forward-only), applied per
  run to the continuous signal. The same channels, filter and window are used
  in every experiment.
- Exclusion in experiments A and B: official artifact flag or any non-finite
  sample. Experiment C replays every E trial; a trial that is non-finite or
  fails a rejection rule fitted on that subject's T trials becomes a hold.

## Leakage rules

- Every fitted step (standardization, CSP, covariance reference, recentering
  reference, encoder, classifiers, temperature, rejection rule, selection of
  epochs/λ/τ) is fit on session-T data of the current training split only.
  Fit entry points raise when given E trials; tests in `tests/` enforce this.
- Recentering reference of subject s: Riemannian mean of s's T covariances in
  the training split. Tangent spaces use a fixed reference (`tsupdate=False`).
- Identity attackers are outside the defended system and are trained on the
  outputs they would collect (see experiment B); attackers tune their own
  hyperparameters only on their own training data.

## Experiment A: decoders × calibration amount (per subject, H1)

- B1 band power: Welch PSD (1 s segments, 50% overlap), log mean power in
  8–13 and 13–30 Hz per channel (44-d); standardization + L2 logistic
  regression, C = 1.
- B2 CSP: 6 spatial filters, log-variance; shrinkage LDA (automatic shrinkage).
- B3 Riemannian tangent space: OAS covariance, tangent space at the training
  Riemannian mean (253-d); standardization + L2 logistic regression, C = 1.
- Calibration: every combination of 1 (6), 2 (15), 3 (20) and 6 (1) T runs;
  each combination is fit with the same trials for B1–B3 and evaluated on all
  retained E trials of the same subject.
- Metrics: balanced accuracy (BA) and Cohen's κ; a subject's value at a
  calibration level is the mean over combinations; inter-subject spread is
  reported.
- Primary test (H1): 1-run B3 − B2, paired over 9 subjects, exact one-sided
  sign-flip permutation test (2^9 = 512 sign patterns).

## Experiment B: shared-representation identity suppression (H2)

One model pooled over the 9 subjects, trained on T and evaluated on E.

| Condition | Representation sent out | Training |
|---|---|---|
| P0 | 253-d tangent vector, common reference | linear task classifier |
| P0-RC | 253-d tangent vector after per-subject recentering | linear task classifier |
| P2 / P2-RC | 8-d bottleneck of MLP encoder 253→64→8 | task loss only (λ = 0) |
| P3 / P3-RC | 8-d bottleneck, same encoder | task loss + identity adversary through gradient reversal |

- Identity adversary: 9-way MLP with one hidden layer on the 8-d bottleneck.
  λ ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}, ramped from 0 to the target with the
  schedule λ·(2/(1+exp(−10p)) − 1), p = training progress.
- P2/P3 and their RC variants share initialization, optimizer and training
  budget; 5 seeds each.
- Development: 6-fold cross-validation over T runs (one run held out per fold).
  The epoch budget is chosen from the fold-mean validation task BA of P2
  (resp. P2-RC) and reused for every λ of P3 (resp. P3-RC). In each fold the
  attackers are trained and tested on a subject-stratified 2-fold split of the
  validation-run outputs.
- Operating point per family: among λ whose fold- and seed-mean task BA is at
  most 5 pp below P2 (resp. P2-RC), choose the lowest strongest-attacker
  identity BA; ties go to higher task BA. No eligible λ = "no operating point"
  failure.
- Final: retrain on all T with the chosen budget and λ, then freeze.
- Attackers, retrained for every representation: multinomial logistic
  regression, RBF SVM (C and γ tuned by inner CV on the attacker's training
  data), and a small MLP distinct from the adversary head. Chance = 1/9.
- Scenarios: main = post-deployment collection (train on frozen outputs of E
  runs 1–3, test on runs 4–6, and the reverse; averaged); auxiliary = train on
  T outputs, test on E. The auxiliary scenario alone never establishes
  protection.
- Strongest attacker = maximum identity BA over attackers and both scenarios.
- Utility (non-inferiority): P3 − P2 E task BA paired over subjects
  (seed-mean per subject); the one-sided 95% lower bound (paired t, n = 9)
  must exceed −5 pp. The exact sign-flip p-value is reported alongside.
- Protection: strongest-attacker identity BA of P3 at least 5 pp below P2, and
  the one-sided 95% lower bound of the paired per-subject identity-recall
  difference (P2 − P3) above 0. The adversary's contribution is accepted only
  if the same holds for P3-RC vs P2-RC.
- Not accepted as evidence: a low score of the training-time identity head.

## Experiment C: replayed virtual one-joint arm (H3)

- Models: the six frozen final models of experiment B.
- One command per trial from the 1–4 s window. Left → −5°, right → +5°;
  temperature-scaled maximum probability below τ → hold (no motion).
  τ ∈ {0.5 (no-hold control), 0.6, 0.7, 0.8}; temperature fitted on T
  cross-validation outputs. Joint range ±90°.
- Episodes: E trials of each run in recorded order, non-overlapping blocks of
  20. Labels only generate the target trajectory and score it; they never
  influence commands or holds.
- Metrics: mean absolute angle error to the target trajectory, end-point
  reach rate (|error| ≤ 10°), wrong-direction moves, out-of-range commands,
  hold rate, effective move count, cumulative unnecessary movement; per-trial
  processing time (filter, inference, control) median and 95th percentile.
- Transmission modes, same model/trials/τ/controller: feature mode sends the
  representation; command mode sends only left/right/hold. Identical
  predictions must produce identical trajectories (checked).
- Observation-length attack: k ∈ {1, 5, 10, 20} consecutive outputs of one
  subject. Feature attacker uses the mean and standard deviation of the k
  representations; command attacker uses hold rate, left/right bias, run
  lengths and transition frequencies. Same scenarios as experiment B.
  Training groups slide with stride 1; test groups do not overlap.
- Report: identity BA vs k for both modes, and the smallest k reaching twice
  chance (2/9 ≈ 22.2%).

## Statistics and reporting

- MI decoding: per-subject values, 9-subject mean and median, paired
  differences, confusion matrices; exact sign-flip permutation tests.
- Identity: BA and macro F1 per scenario/attacker/seed; confidence intervals
  by resampling runs within subject.
- If the unprotected baseline P0 is already at chance for identity, judgment of
  protection is withheld.
- README states what is not shown: 9-person closed set, recorded replay rather
  than closed-loop control, attack failure is not anonymization, unobserved
  sensitive attributes and device compromise are out of scope.

## Deviations from the 2026-10-09 proposal

1. Causal filtering in all experiments; no separate non-causal analysis.
2. Data from the TU Graz host of the same BNCI files (BNCI site unreachable).
3. Dropped: mixed-effects logistic regression, chronological calibration
   conditions, continuous-command auxiliary (2 s window, 0.5 s step),
   unseen-user and external-dataset extensions.
4. Observation-length attack training groups use stride-1 windows.
5. SVM attacker hyperparameters are tuned within the attacker's training data
   instead of T cross-validation.
6. Schedule: results by 2026-11-15; interactive demo, report and explainer
   follow the presentation.

## Follow-up experiments D and E (plan v2, 2026-11-10)

Status: **exploratory.** D and E were designed after the session-E results of
experiments A–C and reuse the same session E, which already shaped their
motivation. Their settings, code and T-only selections are committed before
their own E runs (seal 2, recorded in `docs/decision_log.md`). A confirmatory
replication on untouched data (Lee2019 / OpenBMI, 54 subjects, two days) is
planned separately once data access is verified.

Motivation: experiment B found that a gradient-reversal identity adversary is
fooled (co-trained head identity BA 0.026 at λ = 1) while independent attackers
reach 0.95. Published EEG identity-removal results on the same dataset report
near-zero identity accuracy measured with a co-trained identifier (e.g.
ID-RemovalNet, IJCAI 2025: 0.04% for EEGNet on BCI IV 2a). Two objections must
be answered before generalizing: only plain gradient reversal on Riemannian
features was tested, and the bottleneck dimension (8) was arbitrary.

### Experiment D: is the H2 failure specific to plain gradient reversal?

- Same data, families (common, RC), inputs (253-d tangent vectors), encoder
  (253→64→d), attackers, scenarios, seeds and statistics as experiment B.
- D1 bottleneck sweep: task-only encoders with d ∈ {1, 2, 4, 8, 16, 32}; the
  epoch budget per d and family is chosen as in B (fold-mean T-CV task BA).
- D2 adversarial variants at d = 8, λ ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}:
  - GRL-CE: experiment B's P3 / P3-RC (reference).
  - MaxEnt: the adversary minimizes identity cross-entropy; the encoder pushes
    the adversary posterior toward the uniform distribution instead of
    maximizing the adversary's loss (Roy & Boddeti, CVPR 2019).
  - Ensemble-refresh: 5 adversary heads trained in parallel; the encoder is
    penalized through their mean; one head is re-initialized each epoch in
    rotation.
  - To avoid an under-trained adversary, epoch budget and λ are chosen jointly
    in T-CV from epochs {5, 10, 20, 40} by B's operating-point rule (task BA at
    most 5 pp below the d = 8 task-only encoder of the same family; lowest
    strongest-attacker identity BA).
- Reported: E task BA, strongest independent-attacker identity BA (main and
  auxiliary), co-trained head identity BA as a diagnostic.
- Questions: (i) does any variant pass B's protection rule against the d = 8
  task-only encoder of its family? (ii) frontier dominance: does any
  adversarial operating point have lower strongest-attacker identity BA than
  every task-only bottleneck with equal or higher task BA?

### Experiment E: co-trained identifier vs independent attacker in a deep model

- EEGNet-8,2 (Lawhern et al., 2018), pooled over the 9 subjects, trained on all
  retained T trials; same trials, channels, window and causal 8–30 Hz filter,
  decimated to 125 Hz.
- Alignment: none, or per-subject Euclidean alignment with the subject's T
  reference (He & Wu, 2019), the deep-model analogue of recentering.
- Identity identifier: two fully connected layers on the EEGNet feature,
  attached through gradient reversal; λ ∈ {0, 0.03, 0.1, 0.3, 1, 3}; 3 seeds;
  fixed training budget from `configs/exp_e.yaml`; every λ is reported
  (descriptive, no selection on E).
- Reported per λ: E task BA; identity accuracy of the co-trained identifier on
  E (the metric used by co-trained evaluations); strongest independent-attacker
  identity BA on the frozen feature (main and auxiliary scenarios).
- Question: does near-zero or below-chance co-trained identifier accuracy
  coincide with high independent-attacker identity BA in a deep model?
- This is not a reproduction of ID-RemovalNet; it tests the evaluation metric
  on a common backbone that such methods build on.

### Experiment G (exploratory on BCI IV 2a): a reference-session-fitted defense

Muhaimin & Nagarajan (arXiv 2610.15626) fit a four-stage Riemannian transform
on session 1 and evaluate classifiers trained on defended session 1 against
defended session 2. Experiment G re-implements the transform from the paper,
reproduces that protocol, and re-evaluates the same defended data with
classifiers trained on labelled deployment-session trials (first half of
session 2, first n per participant for the enrolment curve; tested on the
second half; both directions). Results are threat-model dependent and do not
show the paper's own results to be wrong.

## Confirmatory replication on Lee2019 / OpenBMI (plan v3, 2026-11-10)

Status: **confirmatory.** Session 2 of Lee2019 has not been analysed. This
section, the code and every session-1-only selection are committed before any
session-2 analysis (seal 3, recorded in `docs/decision_log.md`). This is a
prospective replication of exploratory findings: λ = 1, d = 8, n = 5 and the
20 pp / 2 × chance / 10 × chance thresholds were fixed before any Lee2019
outcome analysis, motivated by the BCI IV 2a results. The thresholds are
predeclared operational criteria, not established privacy standards.

Report every sealed C1–C4 result and component criterion, including
unsupported claims, failed benchmark gates, missing cells and computational
failures. Publish the complete attacker/scenario/seed/direction results rather
than only maxima. Before seal 3, freeze the analysis manifest, numerical
settings, runtime limits and failure-handling rules.

Report trial exclusions and usable group counts by participant, session and
phase. Form groups in recorded order without bridging excluded trials or phase
boundaries; discard incomplete terminal groups deterministically. If any of
the 54 identities lacks observations required for a claim, mark that claim not
evaluable rather than silently dropping the identity or changing chance.

Session-2 findings cannot change primary preprocessing, exclusions,
thresholds, attackers or stopping rules. Any amendment is dated and justified,
retains the sealed analysis, and labels the amended analysis exploratory. A
parameter-free change is not automatically outcome-independent.

### Data and splits

- 54 participants, 2 sessions on different days. Reduced by
  `scripts/prepare_lee2019.py`: 20 motor channels (FC5/3/1/2/4/6, C5/3/1/z/2/4/6,
  CP5/3/1/z/2/4/6), causal 8–30 Hz Butterworth at 1000 Hz, decimated to 250 Hz,
  1–4 s after the cue. Raw-file SHA-256 in `data/processed/lee2019/SHA256SUMS`.
- Each session has an offline phase and an online phase of 100 trials
  (50 left, 50 right). Blocks: offline trials 1–50 = block 1, 51–100 = block 2,
  online 1–50 = block 3, 51–100 = block 4. Exclusion: exclude trials whose
  reducer-generated `finite` flag is false, indicating a non-finite raw
  motor-channel sample within the analysis epoch or its preceding 1 s guard.
  Zero replacement before filtering does not restore eligibility. No official
  artifact flags exist. Labels are the cue labels (`y_dec`):
  every session-1 phase of all 54 participants has exactly 50 left and 50
  right trials, which online classifier decisions would not produce.
- Session 1 = T for every defended-side fitted component and model-selection
  decision. Leave-one-block-out CV holds out the corresponding 50-trial block
  across all participants; fitted components are recomputed using only that
  fold's training blocks. These blocks are analytical subdivisions of two
  phases, not independent recording runs.
- Session 2 = E for frozen-system evaluation. Deployment attackers may fit
  preprocessing, classifiers and the sealed inner-CV searches using only their
  designated labelled training phase. They never use their test phase for
  fitting or tuning.
- The primary deployment estimand equally averages offline→online and
  online→offline identity performance. Report both directions separately. This
  is within-day, cross-phase retrospective linkage; only offline→online
  respects acquisition chronology. Online feedback makes the phases
  non-interchangeable experimental conditions. The auxiliary scenario trains
  attackers on session-1 outputs and evaluates session 2.
- Identity chance = 1/54 (1.85%); 2 × chance = 3.70%.

### Methods carried over unchanged unless stated

Experiment B's conditions, encoder, ramp schedule, attackers, seeds, budget
rule and statistics (`configs/exp_b.yaml`), applied to the 210-d tangent space
of 20 channels; Experiment C's command rule (temperature from session-1 CV,
τ ∈ {0.5, 0.6, 0.7, 0.8}) and observation-length grouping
(k ∈ {1, 5, 10, 20}, groups within a phase); Experiment G's transform at the
paper's OpenBMI operating point (0.9, 0.60, 0.6, 0.2). Attacker settings whose
projected runtime is impractical are fixed from session-1 timing before seal 3
and recorded in the config.

### Confirmatory claims and decision rules

For each fixed attacker and scenario, first average recall within participant
over the specified seeds and, for the deployment scenario, equally over the two
directions. Identity BA is the mean of these 54 participant recalls.
"Strongest" means the maximum of these cohort-level BAs over the sealed
attacker/scenario set; never a maximum separately within each participant or
seed.

Let `T(v, a)` be the one-sided t-test p-value for the mean of the 54
participant values `v` exceeding `a` (53 degrees of freedom). These are
approximate, conditional analyses of the fitted systems and enrolled identity
set, not uncertainty over retraining or new cohorts. Let `j, l` index the
J = 6 combinations of logistic regression, RBF SVM and MLP with the deployment
and auxiliary scenarios; `r[P, j]` the participant-recall vector for
representation P; `h` the co-trained head's recall vector on the matched
evaluation outputs; `u` the participant task-BA difference P2-RC − P0;
`c[k, j]` the command-recall vector at observation length k; `g[j]` the K = 4
deployment attackers of C4. Claim-level p-values (Bonferroni products capped
at 1):

- `p1 = min(1, J · min_j T(r[P3, j] − h, 0))`
- `p2 = max{ T(u, −0.05), min(1, J · min_j max_l T(r[P0, j] − r[P2-RC, l], 0)) }`
- `p3 = min(1, J · min_j max{ T(c[20, j], 2/54), max_l T(c[20, j] − c[1, l], 0) })`
- `p4 = min(1, K · min_j T(g[j], 10/54))`

Set p1 or p2 to 1 if its observed strongest-BA difference is below 0.20; the
20 pp requirements concern observed effect sizes and do not establish a
population effect of at least 20 pp. Set p4 to 1 if C4's benchmark gate fails.
Apply Holm's step-down procedure at familywise α = 0.05 to the four
claim-level p-values; C1–C4 are decided by these tests, not by unadjusted
winner intervals. Nominal 95% intervals are descriptive. Missing required cells
make the affected claim not evaluable; they do not shrink the sealed family.

Descriptive bootstrap intervals resample complete participant-summary vectors
within each predeclared enrolled identity set, using 10,000 replicates and
seed 42. The set contains 54 participants except in the N-scaling analyses,
where it contains the fixed N-participant subset. Use identical
participant-index draws across paired summaries within the same identity set.
Resampling changes participant weights, not the fixed label space. Missing
required participant, seed or direction cells make the affected summary not
evaluable; they never reduce the predeclared identity set. Intervals are
conditional on the fitted systems; blocks, overlapping windows, directions and
training seeds are not independent replicates.

- **C1 (co-trained evaluation understates exposure).** For the gradient-reversal
  model P3 (common family, d = 8, λ = 1 fixed before Lee2019 outcome analysis,
  motivated by the BCI IV 2a findings), the strongest independent classifier's
  identity BA exceeds the co-trained head's identity BA on the same evaluation
  outputs by at least 20 pp, and p1 passes the Holm procedure.
- **C2 (a recentred task-only bottleneck reduces measured exposure relative to
  full tangent features).** P2-RC (recentred, task-only bottleneck, d = 8)
  versus P0 (full tangent vector): strongest identity BA at least 20 pp lower,
  task BA non-inferior by the −5 pp margin, both through p2 under Holm. The
  d = 1 variant is reported as secondary (no decision).
- **C3 (longer command observations increase measured identity exposure).** For
  P2-RC command-only transmission at τ = 0.5, test jointly that the strongest
  identity BA at k = 20 exceeds 2/54 and exceeds the strongest BA at k = 1,
  using p3 under Holm. Both observation lengths use the same eligible
  test-trial support.
- **C4 (five deployment labels permit residual identification under the
  reduced preprocessing pipeline).** Fit the Experiment G transform on session 1
  at (0.9, 0.60, 0.6, 0.2), then freeze it. Evaluate paper_ts_svm,
  beat_logistic, beat_rbf_svm and beat_mlp, each trained using only the first
  five retained, chronologically ordered defended trials per participant in
  the training phase; all attacker-fitted preprocessing uses only those
  enrolment trials. Evaluate k = 1 on the opposite phase and average the two
  directions equally. C4 requires p4 to pass the Holm procedure.
  **Benchmark gate:** under S1→S2 training and testing on the same reduced
  pipeline, the fixed paper-style tangent-space SVM must have defended accuracy
  A with |A − 0.024| ≤ 0.05. This intentionally coarse range (0–7.4%) is a
  benchmark-consistency gate, not an equivalence test or evidence of faithful
  reproduction; the causal 1–4 s reduced epochs differ from the paper's
  preprocessing, so this is an adapted replication. Report undefended and
  defended reference-session accuracies, every deployment attacker result and
  the benchmark gap regardless of the gate. If the gate fails, C4 is not
  evaluable and p4 = 1 stays in Holm. Neither preprocessing nor the gate may
  change after session-2 evaluation. Conclusions concern these attackers, this
  pipeline and this labelled-enrolment threat model, not universal protection
  or a refutation of the paper's stated threat model.

### Reported without a decision

Confirmatory cells (C1–C4) use all five seeds. To keep the final run feasible
on one laptop CPU, the following descriptive analyses use seeds 0–2 and are
restricted as listed (fixed before seal 3 from session-1 timing):

- Enrolment curves, n ∈ {1, 2, 5, 10, 20, all}, for the Experiment G transform
  (defended and undefended) and for P0-RC.
- Bottleneck dimension d ∈ {1, 8} for P2 and P2-RC.
- Co-trained head versus independent classifiers for P3 (common family) at
  λ ∈ {0.1, 1, 3}.
- Observation-length curves, k ∈ {1, 5, 10, 20}, for P0, P2-RC and P3
  (λ = 1): features, and commands at τ ∈ {0.5, 0.8}.
- Identification versus the number of enrolled participants,
  N ∈ {9, 18, 27, 54} (participant subsets drawn with seed 42), for P0 and
  P2-RC in the deployment scenario.
- Control metrics for P0, P2-RC and P3 (λ = 1) at τ ∈ {0.5, 0.6, 0.7, 0.8}.

Attacker settings (session-1 timing on 5,400 × 210-d training sets): the RBF
SVM uses fixed C = 1 and gamma = "scale" instead of B's grid (identity BA
0.9704 versus 0.9706 for the full grid on session 1, 12 s versus 70 s per
fit); the MLP limits are max_iter = 500 and max_fun = 10,000 (converged in
18 iterations).

### Not pursued

A per-command randomized-response release with a formal identification bound
was considered and set aside for this phase: meaningful formal guarantees over
tens of commands require privacy budgets that make the commands nearly random,
so it would not change the practical conclusions.
