# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.4996 | [0.4920, 0.5072] | 0.68183 |
| cross-day-d2-d3 | 1 | bayes_factor | linkage detected | 0.5416 | [0.5168, 0.5678] | 0.00059994 |
| same-day | 1 | bayes_factor | linkage detected | 0.5596 | [0.5464, 0.5749] | 0.00059994 |
| same-day-d2-d3 | 1 | bayes_factor | linkage detected | 0.5874 | [0.5683, 0.6093] | 0.00059994 |
| cross-day | 5 | bayes_factor | linkage detected | 0.5192 | [0.5051, 0.5330] | 0.0011999 |
| cross-day-d2-d3 | 5 | bayes_factor | linkage detected | 0.6111 | [0.5549, 0.6628] | 0.00059994 |
| same-day | 5 | bayes_factor | linkage detected | 0.6541 | [0.6328, 0.6761] | 0.00059994 |
| same-day-d2-d3 | 5 | bayes_factor | linkage detected | 0.7234 | [0.6904, 0.7566] | 0.00059994 |
| cross-day | 10 | bayes_factor | linkage detected | 0.5192 | [0.5029, 0.5370] | 0.0053995 |
| cross-day-d2-d3 | 10 | bayes_factor | linkage detected | 0.6371 | [0.5706, 0.6954] | 0.00059994 |
| same-day | 10 | bayes_factor | linkage detected | 0.6951 | [0.6706, 0.7189] | 0.00059994 |
| same-day-d2-d3 | 10 | bayes_factor | linkage detected | 0.7766 | [0.7420, 0.8090] | 0.00059994 |
| cross-day | 20 | bayes_factor | linkage detected | 0.5330 | [0.5116, 0.5563] | 0.0009999 |
| cross-day-d2-d3 | 20 | bayes_factor | linkage detected | 0.6630 | [0.5901, 0.7267] | 0.00059994 |
| same-day | 20 | bayes_factor | linkage detected | 0.7385 | [0.7128, 0.7626] | 0.00059994 |
| same-day-d2-d3 | 20 | bayes_factor | linkage detected | 0.8269 | [0.7938, 0.8588] | 0.00059994 |
| cross-day | 50 | bayes_factor | linkage detected | 0.5457 | [0.5220, 0.5711] | 0.00059994 |
| cross-day-d2-d3 | 50 | bayes_factor | linkage detected | 0.6850 | [0.6079, 0.7549] | 0.00059994 |
| same-day | 50 | bayes_factor | linkage detected | 0.7698 | [0.7419, 0.7967] | 0.00059994 |
| same-day-d2-d3 | 50 | bayes_factor | linkage detected | 0.8781 | [0.8455, 0.9054] | 0.00059994 |
| cross-day | 100 | bayes_factor | linkage detected | 0.5417 | [0.5103, 0.5755] | 0.0011999 |
| cross-day-d2-d3 | 100 | bayes_factor | linkage detected | 0.6978 | [0.6144, 0.7760] | 0.00059994 |
| same-day | 100 | bayes_factor | linkage detected | 0.7848 | [0.7474, 0.8233] | 0.00059994 |
| same-day-d2-d3 | 100 | bayes_factor | linkage detected | 0.9066 | [0.8795, 0.9302] | 0.00059994 |

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

{'cross-day': 5, 'cross-day-d2-d3': 1, 'same-day': 1, 'same-day-d2-d3': 1}

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

- cross-day/bayes_factor: mean AUC below 0.5
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
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
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2', '3']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `ba1a8197aa690737bd7ef05f7fe36ceb72356e8034a11e59d334cc6bf3248c3d`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
