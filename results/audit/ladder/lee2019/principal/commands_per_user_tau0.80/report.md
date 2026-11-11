# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-session2 | 1 | bayes_factor | linkage detected | 0.5629 | [0.5448, 0.5818] | 0.00059994 |
| same-day-session2 | 5 | bayes_factor | linkage detected | 0.7043 | [0.6662, 0.7436] | 0.00059994 |
| same-day-session2 | 10 | bayes_factor | linkage detected | 0.7607 | [0.7204, 0.8009] | 0.00059994 |
| same-day-session2 | 20 | bayes_factor | linkage detected | 0.8247 | [0.7818, 0.8630] | 0.00059994 |
| same-day-session2 | 50 | bayes_factor | linkage detected | 0.8781 | [0.8338, 0.9133] | 0.00059994 |
| same-day-session2 | 100 | bayes_factor | linkage detected | 0.9133 | [0.8583, 0.9551] | 0.00059994 |
| cross-day | 1 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 5 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 10 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 50 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 100 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-session2, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=100: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day-session2': 1}

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

- cross-day: not evaluated; supply explicit side conditions
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `e53f1efd132f903396dc6526003abbcf255a9e4feef4a42599f5f071f5c2b858`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
