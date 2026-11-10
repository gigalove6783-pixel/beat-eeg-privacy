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
