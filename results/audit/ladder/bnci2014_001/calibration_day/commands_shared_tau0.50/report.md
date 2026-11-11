# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-sessionT-sessionE | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.4845 | [0.4529, 0.5220] (unreliable: small folds) | 0.93088 |
| cross-day-sessionT-sessionE | 5 | bayes_factor | not detected: insufficient evidence, not absence | 0.5490 | [0.4567, 0.6389] (unreliable: small folds) | 0.92166 |
| cross-day-sessionT-sessionE | 10 | bayes_factor | not detected: insufficient evidence, not absence | 0.5698 | [0.4539, 0.6961] (unreliable: small folds) | 0.92166 |
| cross-day-sessionT-sessionE | 20 | bayes_factor | not detected: insufficient evidence, not absence | 0.5946 | [0.4356, 0.7454] (unreliable: small folds) | 0.92166 |
| cross-day-sessionT-sessionE | 50 | bayes_factor | not detected: insufficient evidence, not absence | 0.6192 | [0.4421, 0.7848] (unreliable: small folds) | 0.74654 |
| cross-day-sessionT-sessionE | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-sessionT-sessionE, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'cross-day-sessionT-sessionE': None}

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
- same-day: not evaluated; supply explicit side conditions
- cross-day-sessionT-sessionE/bayes_factor: mean AUC below 0.5
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `120f3a8359796b1b415c0a026c46cd0b7f998018e6c7168873ddd21e6d6538af`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
