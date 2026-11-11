# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-d2-d3 | 1 | bayes_factor | linkage detected | 0.5909 | [0.5694, 0.6124] | 0.00059994 |
| cross-day-d2-d3 | 1 | bayes_factor | linkage detected | 0.5354 | [0.5063, 0.5645] | 0.0031997 |
| same-day-d2-d3 | 5 | bayes_factor | linkage detected | 0.7198 | [0.6802, 0.7550] | 0.00059994 |
| cross-day-d2-d3 | 5 | bayes_factor | linkage detected | 0.5938 | [0.5312, 0.6542] | 0.0009999 |
| same-day-d2-d3 | 10 | bayes_factor | linkage detected | 0.7742 | [0.7349, 0.8098] | 0.00059994 |
| cross-day-d2-d3 | 10 | bayes_factor | linkage detected | 0.6215 | [0.5497, 0.6884] | 0.00059994 |
| same-day-d2-d3 | 20 | bayes_factor | linkage detected | 0.8264 | [0.7888, 0.8601] | 0.00059994 |
| cross-day-d2-d3 | 20 | bayes_factor | linkage detected | 0.6337 | [0.5569, 0.7074] | 0.00059994 |
| same-day-d2-d3 | 50 | bayes_factor | linkage detected | 0.8882 | [0.8549, 0.9163] | 0.00059994 |
| cross-day-d2-d3 | 50 | bayes_factor | linkage detected | 0.6549 | [0.5715, 0.7351] | 0.00059994 |
| same-day-d2-d3 | 100 | bayes_factor | linkage detected | 0.9079 | [0.8673, 0.9411] | 0.00059994 |
| cross-day-d2-d3 | 100 | bayes_factor | linkage detected | 0.6574 | [0.5725, 0.7399] | 0.00059994 |

- same-day-d2-d3, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=100: excluded users ['24', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=100: excluded users ['24', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day-d2-d3': 1, 'cross-day-d2-d3': 1}

Any detected condition remains detected regardless of another condition's result.

## Limitations and reasonably likely means

- Linkage tests the association of physical A/B blocks, conditional on fitted attacks.
- EER and fixed-FPR TPR are descriptive; deployed thresholds use development users.
- Estimated log LR depends on logistic calibration and transfer to new users/days.
- Cllr_min is a target-fitted oracle diagnostic, never feedback to the attacker.
- D_sys is linkability, not identification probability.
- Gallery extrapolation is an independence sensitivity scenario, not population risk.
- Sequential error guarantees apply only to its generative hypotheses; drift and dependence can violate them.
- Comparator training uses no target users, but a BCI encoder may have seen them.
- These metrics describe technical means an attacker might use; they do not decide whether means are reasonably likely under GDPR Recital 26 or any other law.
- Not detected means insufficient evidence, not absence; never a safety claim.

## Guards

- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['2', '3']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `887aa0626c05580dfe37fb701d302714a12661440e31d9d8a4d6457584a0a9ff`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
