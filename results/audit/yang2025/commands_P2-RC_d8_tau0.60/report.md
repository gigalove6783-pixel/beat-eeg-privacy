# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day | 1 | bayes_factor | linkage detected | 0.5065 | [0.4976, 0.5155] | 0.0070993 |
| cross-day-d2-d3 | 1 | bayes_factor | linkage detected | 0.5730 | [0.5410, 0.6040] | 0.00059994 |
| same-day | 1 | bayes_factor | linkage detected | 0.5746 | [0.5552, 0.5935] | 0.00059994 |
| same-day-d2-d3 | 1 | bayes_factor | linkage detected | 0.6127 | [0.5878, 0.6389] | 0.00059994 |
| cross-day | 5 | bayes_factor | linkage detected | 0.5214 | [0.5022, 0.5400] | 0.00059994 |
| cross-day-d2-d3 | 5 | bayes_factor | linkage detected | 0.6476 | [0.5845, 0.7060] | 0.00059994 |
| same-day | 5 | bayes_factor | linkage detected | 0.6910 | [0.6640, 0.7180] | 0.00059994 |
| same-day-d2-d3 | 5 | bayes_factor | linkage detected | 0.7607 | [0.7202, 0.7977] | 0.00059994 |
| cross-day | 10 | bayes_factor | linkage detected | 0.5334 | [0.5114, 0.5561] | 0.00059994 |
| cross-day-d2-d3 | 10 | bayes_factor | linkage detected | 0.6766 | [0.6053, 0.7385] | 0.00059994 |
| same-day | 10 | bayes_factor | linkage detected | 0.7355 | [0.7065, 0.7621] | 0.00059994 |
| same-day-d2-d3 | 10 | bayes_factor | linkage detected | 0.8199 | [0.7834, 0.8526] | 0.00059994 |
| cross-day | 20 | bayes_factor | linkage detected | 0.5513 | [0.5272, 0.5752] | 0.00059994 |
| cross-day-d2-d3 | 20 | bayes_factor | linkage detected | 0.6956 | [0.6190, 0.7592] | 0.00059994 |
| same-day | 20 | bayes_factor | linkage detected | 0.7774 | [0.7471, 0.8046] | 0.00059994 |
| same-day-d2-d3 | 20 | bayes_factor | linkage detected | 0.8638 | [0.8288, 0.8948] | 0.00059994 |
| cross-day | 50 | bayes_factor | linkage detected | 0.5671 | [0.5397, 0.5960] | 0.00059994 |
| cross-day-d2-d3 | 50 | bayes_factor | linkage detected | 0.7149 | [0.6327, 0.7836] | 0.00059994 |
| same-day | 50 | bayes_factor | linkage detected | 0.8291 | [0.7998, 0.8548] | 0.00059994 |
| same-day-d2-d3 | 50 | bayes_factor | linkage detected | 0.9194 | [0.8909, 0.9419] | 0.00059994 |
| cross-day | 100 | bayes_factor | linkage detected | 0.5744 | [0.5376, 0.6124] | 0.00059994 |
| cross-day-d2-d3 | 100 | bayes_factor | linkage detected | 0.7271 | [0.6405, 0.8074] | 0.00059994 |
| same-day | 100 | bayes_factor | linkage detected | 0.8569 | [0.8193, 0.8917] | 0.00059994 |
| same-day-d2-d3 | 100 | bayes_factor | linkage detected | 0.9488 | [0.9251, 0.9683] | 0.00059994 |

- cross-day, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=100: excluded users ['23', '24', '26', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d2-d3, k=100: excluded users ['24', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=100: excluded users ['23', '24', '26', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-d2-d3, k=100: excluded users ['24', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'cross-day': 1, 'cross-day-d2-d3': 1, 'same-day': 1, 'same-day-d2-d3': 1}

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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2', '3']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `ba1a8197aa690737bd7ef05f7fe36ceb72356e8034a11e59d334cc6bf3248c3d`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
