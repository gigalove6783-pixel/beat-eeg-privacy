# Transmitted-output audit (builders)

Install the repository with `python -m pip install .`. The `beat_audit` package
uses NumPy, SciPy, scikit-learn, pandas, Matplotlib and PyYAML only; importing it
does not import torch or the sealed BEAT experiments. The parent distribution
retains BEAT's existing dependencies.

From **any working directory**, with absolute paths:

```
beat-audit run "C:/dev/BEAT PROJECT/examples/commands.csv" --out commands-report --k 1 5
beat-audit run "C:/dev/BEAT PROJECT/examples/features.csv" --out features-report --k 1 5
```

Replace the repository path (including spaces) with your actual, quoted path.
Both examples are synthetic; regenerate with `python -m examples.generate` in
the repository, or `python examples/generate.py` after installation.
CSV commands use `outputs`; features use `outputs_0`, `outputs_1`, etc.
Every row requires `user` and `day`. Optional: `segment`, `order`, `target`;
`C` declares the command alphabet, including never-observed commands.
NPZ uses the same names, with `outputs` a vector or matrix and scalar `C`.
Use Unicode labels, never pickle/object arrays.
Default chronology is inferred only from numeric or ISO-date labels. For opaque
labels, use `--day-order T E` or `AuditConfig(day_order=("T", "E"))`. The order is
recorded in the report. Explicit Condition pairs remain authoritative and do not
require an unused global chronology.

`beat_audit.audit_stream` is the Python entry point. See its docstring and
`Condition`, `Segment`, `Population`, `AuditConfig` in `beat_audit.api`.
Input mappings group model seeds. Explicit conditions and a common Population
are recommended for cohort work. Each cell records exclusions, status, its
actual user/mask/fold hash, and all estimates. Source, configuration and input
SHA-256 hashes are in the report. Unexpected computation errors remain failed.

`beat-audit compare baseline=.../report.json alternative=.../report.json
--utility utility.json --out comparison.csv` refuses population/condition/day/k
mismatches. Utility JSON maps the alternative names to `task_ba`, `hold_rate`,
`wrong_move_rate`. These are builder claims, never attack scores.
Compatibility includes canonical resolved A/B selectors, not the whole config.
Legacy reports without recoverable selectors must be rerun. Comparison rows
retain cell-level detection and name detecting comparators separately from the
selected comparator's statistics and wording. Missing same-day/cross-day analyses
are explicit `not_evaluable` rows, also recorded as `unavailable_analyses` without
altering the requested/scored `cells` manifest used by cohort adapters.
Declare a deterministic derivation with `--derived-from commands=features`, or
`compare_reports(reports, utilities, derived_from={"commands": "features"})`.
If the command AUC exceeds the feature AUC at the same condition/k, both rows
flag the weak feature attacker and the command-derived lower bound on exposure.
Measured AUCs and detection decisions are never rewritten.

The default feature attacks also include `cosine_rich` and `lda_cosine_rich`.
Their window summaries concatenate the mean, cue means (when target is supplied,
absent cues use the window mean), and log population variance for k >= 5
(fixed floor 1e-12, ddof=0). Rich attacks are skipped at k=1. All fitted transforms
and calibration use development users only. Pair-logistic and the closed-set
module retain mean features, and original comparator calculations are unchanged.

`beat_audit.audit_timing(inputs, *, folds=(0,), conditions=None, config=None,
population=None, kind=None)` runs selected target folds with all original
development users and fold-specific seeds. Restrict conditions/k/comparators
using the ordinary API. It returns overall runtime/peak RSS and per-comparator
`timing_seconds`, with `purpose="timing_only"`; `compare` refuses these partial
diagnostics. Fold positions are zero-based in sorted fold-ID order.

Implementation choices where the plan is silent:

- Day labels use numeric or ISO-date chronology, or an explicit day order.
  Default same-day sides split the sorted segment labels into two groups.
- Comparison budgets (200 mated, 50 per nonmated partner) are divided as evenly
  as possible across days, directions and model seeds. All candidate pairings
  use the same role-specific sampling rule, including permuted mates.
  Feature pair materialisation is streamed in chunks of at most 4,096 pairs;
  only scalar scores are retained across chunks.
- Concentration is shared across cue strata; each cue has its own positive
  development-count base measure. The first maximum wins a grid tie.
- LDA is within-class Ledoit-Wolf covariance whitening followed by between-class
  SVD. Both centering within user and shrinkage estimation use development users
  only. Reports record the estimator, shrinkage coefficient, isotropic target
  and condition number. The retained `within_covariance_ridge` field is now the
  estimated isotropic term (shrinkage times target), not a fixed eigenvalue floor.
  Exactly zero within-class noise uses identity whitening; nonpositive numerical
  eigenvalues otherwise have zero inverse in the PSD pseudoinverse. Pair-logistic
  uses C=1, balanced pairs, and at most 20,000 training pairs. Zero-vector cosine is 0.
- Equal class weights are also equal by user, partner and direction/day/seed
  when fitting calibration and deriving development thresholds. EER/TPR
  interpolate the ROC; a deployed threshold uses the nearest actual score cut.
- Closed-set feature defaults: standardised logistic C=1, RBF-SVM C=1 and
  gamma=scale, MLP (100 hidden units, 500 iterations). Public `identify` and
  `enrolled_log_likelihood` live in `beat_audit.closed_set`.
- Gallery measurement uses a common probe and one sampled reference per
  candidate; uniform ties and gallery-subset draws are integrated exactly.
  Diagnostic gallery rows use the same per-partner bound. Predictions are
  comparator-local (including model seed), then averaged; larger-N scenarios
  require every fold to pass.
- Bootstrap contrasts require identical analysed users and folds. No silently
  intersected population or zero-filled missing effect is reported.
- Detected comparators are labelled separately with Holm over k; the named
  strongest comparator is a ranking observed on target users (post hoc), not an
  attacker selected on development users. A frozen study must pre-name its
  comparator and k.
- Reduced inference budgets are explicitly labelled nonstandard. The validation
  module reports genuine failures; it never relaxes coverage targets to pass.

Run `python -m beat_audit.validate` for the full synthetic lane, or
`python -m pytest tests/test_audit*.py` (expand this glob in PowerShell) for CI.
Append representative 51 x 3 multi-window nulls with
`python -m beat_audit.validate --lane representative --simulations 16 --threads 4`.
This keeps the existing validation records and their original source hash,
adding a separate lane for the five feature attacks (210 raw dimensions, k=20)
and cue-aware commands (k=50), three model seeds and active comparison bounds.
Every case records its binomial 99% tolerance and all simulation outcomes.
The stronger decision-setting null extension is
`python -m beat_audit.validate --lane decision-null --simulations 200 --threads 4`.
It appends cosine-rich, LDA-rich (k=20) and cue-aware Bayes-factor (k=50) cases
without replacing earlier results. The default remains 10,000 permutations;
an explicitly reduced `--permutations 2000` is recorded rather than represented
as the full inference budget. The bootstrap still uses 2,000 replicates.
No report means anonymous or safe. The empirical-epsilon stretch module is absent.
