# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.5198 | [0.4861, 0.5657] (unreliable: small folds) | 0.18433 |
| cross-day | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.5124 | [0.4832, 0.5450] (unreliable: small folds) | 0.4977 |
| same-day | 5 | bayes_factor | not detected: insufficient evidence, not absence | 0.6105 | [0.5101, 0.7248] (unreliable: small folds) | 0.12442 |
| cross-day | 5 | bayes_factor | not detected: insufficient evidence, not absence | 0.5403 | [0.4550, 0.6313] (unreliable: small folds) | 0.88479 |
| same-day | 10 | bayes_factor | not detected: insufficient evidence, not absence | 0.6363 | [0.5158, 0.7682] (unreliable: small folds) | 0.082949 |
| cross-day | 10 | bayes_factor | not detected: insufficient evidence, not absence | 0.5726 | [0.4656, 0.7017] (unreliable: small folds) | 0.88479 |
| same-day | 20 | bayes_factor | not detected: insufficient evidence, not absence | 0.6715 | [0.5385, 0.8172] (unreliable: small folds) | 0.092166 |
| cross-day | 20 | bayes_factor | not detected: insufficient evidence, not absence | 0.5779 | [0.4271, 0.7449] (unreliable: small folds) | 0.88479 |
| same-day | 50 | bayes_factor | not detected: insufficient evidence, not absence | 0.7222 | [0.5185, 0.8819] (unreliable: small folds) | 0.092166 |
| cross-day | 50 | bayes_factor | not detected: insufficient evidence, not absence | 0.6343 | [0.4456, 0.8038] (unreliable: small folds) | 0.4977 |
| same-day | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| cross-day | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |

- same-day, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day': None, 'cross-day': None}

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
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `5e94b43843bbbabc08400bffad4449bb311f4afae4084974166af3b8056f24f9`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
