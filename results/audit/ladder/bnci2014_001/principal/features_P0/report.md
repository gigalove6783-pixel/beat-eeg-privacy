# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-sessionE | 1 | cosine | not detected: insufficient evidence, not absence | 0.9822 | [0.9559, 0.9987] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.8190 | [0.6861, 0.9306] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 5 | cosine | not detected: insufficient evidence, not absence | 0.9961 | [0.9882, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.9346 | [0.8004, 0.9934] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 5 | cosine_rich | linkage detected | 0.9973 | [0.9919, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 5 | lda_cosine_rich | linkage detected | 0.9306 | [0.8199, 0.9906] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 10 | cosine | not detected: insufficient evidence, not absence | 0.9981 | [0.9944, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.9608 | [0.8596, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 10 | cosine_rich | linkage detected | 0.9986 | [0.9954, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 10 | lda_cosine_rich | linkage detected | 0.9635 | [0.8873, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 20 | cosine | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.9634 | [0.8238, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 20 | cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 20 | lda_cosine_rich | linkage detected | 0.9529 | [0.8561, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 50 | cosine | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.9167 | [0.7778, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 50 | cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 50 | lda_cosine_rich | linkage detected | 0.9722 | [0.8889, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| cross-day | 1 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 5 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 10 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 50 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 100 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-sessionE, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=20: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=50: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day-sessionE': 5}

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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `06c309c916265d5807556b7b70c57606d66357f5ccff92a20c304ad9f9e2d9f6`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
