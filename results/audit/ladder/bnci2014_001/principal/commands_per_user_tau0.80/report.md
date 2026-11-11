# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-sessionE | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.5189 | [0.4757, 0.5662] (unreliable: small folds) | 0.1106 |
| same-day-sessionE | 5 | bayes_factor | not detected: insufficient evidence, not absence | 0.5927 | [0.4819, 0.7292] (unreliable: small folds) | 0.2212 |
| same-day-sessionE | 10 | bayes_factor | not detected: insufficient evidence, not absence | 0.6600 | [0.5046, 0.8298] (unreliable: small folds) | 0.20737 |
| same-day-sessionE | 20 | bayes_factor | not detected: insufficient evidence, not absence | 0.7032 | [0.4985, 0.9336] (unreliable: small folds) | 0.20737 |
| same-day-sessionE | 50 | bayes_factor | not detected: insufficient evidence, not absence | 0.6944 | [0.4167, 1.0000] (unreliable: small folds) | 0.2212 |
| same-day-sessionE | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| cross-day | 1 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 5 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 10 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 50 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 100 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-sessionE, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day-sessionE': None}

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

- Below 20 users: estimates are unstable
- cross-day: not evaluated; supply explicit side conditions
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `748c3ae24abf5551ba9a7e375bda6e5c851000713b75de42dbe69aece701af74`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
