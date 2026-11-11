# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.5239 | [0.4881, 0.5632] (unreliable: small folds) | 0.21198 |
| cross-day | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.5389 | [0.4976, 0.5805] (unreliable: small folds) | 0.13825 |
| same-day | 5 | bayes_factor | not detected: insufficient evidence, not absence | 0.6200 | [0.5201, 0.7305] (unreliable: small folds) | 0.0553 |
| cross-day | 5 | bayes_factor | not detected: insufficient evidence, not absence | 0.5887 | [0.4933, 0.6882] (unreliable: small folds) | 0.16129 |
| same-day | 10 | bayes_factor | not detected: insufficient evidence, not absence | 0.6646 | [0.5524, 0.7841] (unreliable: small folds) | 0.0553 |
| cross-day | 10 | bayes_factor | not detected: insufficient evidence, not absence | 0.5891 | [0.4480, 0.7290] (unreliable: small folds) | 0.40092 |
| same-day | 20 | bayes_factor | not detected: insufficient evidence, not absence | 0.7088 | [0.5750, 0.8479] (unreliable: small folds) | 0.0553 |
| cross-day | 20 | bayes_factor | not detected: insufficient evidence, not absence | 0.6197 | [0.4552, 0.7711] (unreliable: small folds) | 0.31336 |
| same-day | 50 | bayes_factor | not detected: insufficient evidence, not absence | 0.7454 | [0.5972, 0.9028] (unreliable: small folds) | 0.0553 |
| cross-day | 50 | bayes_factor | not detected: insufficient evidence, not absence | 0.6181 | [0.4010, 0.7969] (unreliable: small folds) | 0.40092 |
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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `5e94b43843bbbabc08400bffad4449bb311f4afae4084974166af3b8056f24f9`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
