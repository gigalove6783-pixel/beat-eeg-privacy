# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | cosine | not detected: insufficient evidence, not absence | 0.9838 | [0.9700, 0.9965] (unreliable: small folds) | 0.0553 |
| same-day | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.9162 | [0.8359, 0.9803] (unreliable: small folds) | 0.0553 |
| same-day | 1 | pair_logistic | not detected: insufficient evidence, not absence | 0.9690 | [0.9410, 0.9926] (unreliable: small folds) | 0.0553 |
| cross-day | 1 | cosine | not detected: insufficient evidence, not absence | 0.9552 | [0.8917, 0.9958] (unreliable: small folds) | 0.0553 |
| cross-day | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.8093 | [0.6876, 0.9388] (unreliable: small folds) | 0.0553 |
| cross-day | 1 | pair_logistic | not detected: insufficient evidence, not absence | 0.9098 | [0.8349, 0.9713] (unreliable: small folds) | 0.0553 |
| same-day | 5 | cosine | not detected: insufficient evidence, not absence | 0.9970 | [0.9911, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.9609 | [0.8561, 0.9979] (unreliable: small folds) | 0.0553 |
| same-day | 5 | cosine_rich | linkage detected | 0.9975 | [0.9926, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 5 | lda_cosine_rich | linkage detected | 0.9621 | [0.8628, 0.9974] (unreliable: small folds) | 0.046083 |
| same-day | 5 | pair_logistic | not detected: insufficient evidence, not absence | 0.9949 | [0.9847, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 5 | cosine | not detected: insufficient evidence, not absence | 0.9819 | [0.9393, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.8917 | [0.7825, 0.9851] (unreliable: small folds) | 0.0553 |
| cross-day | 5 | cosine_rich | linkage detected | 0.9863 | [0.9512, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day | 5 | lda_cosine_rich | linkage detected | 0.9188 | [0.8305, 0.9871] (unreliable: small folds) | 0.046083 |
| cross-day | 5 | pair_logistic | not detected: insufficient evidence, not absence | 0.9815 | [0.9457, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 10 | cosine | not detected: insufficient evidence, not absence | 0.9993 | [0.9979, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.9845 | [0.9419, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 10 | cosine_rich | linkage detected | 0.9995 | [0.9985, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 10 | lda_cosine_rich | linkage detected | 0.9834 | [0.9319, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 10 | pair_logistic | not detected: insufficient evidence, not absence | 0.9994 | [0.9983, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 10 | cosine | not detected: insufficient evidence, not absence | 0.9851 | [0.9367, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.9507 | [0.8807, 0.9992] (unreliable: small folds) | 0.0553 |
| cross-day | 10 | cosine_rich | linkage detected | 0.9888 | [0.9493, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day | 10 | lda_cosine_rich | linkage detected | 0.9482 | [0.8973, 0.9940] (unreliable: small folds) | 0.046083 |
| cross-day | 10 | pair_logistic | not detected: insufficient evidence, not absence | 0.9868 | [0.9468, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 20 | cosine | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.9666 | [0.8515, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 20 | cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 20 | lda_cosine_rich | linkage detected | 0.9777 | [0.8978, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 20 | pair_logistic | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 20 | cosine | not detected: insufficient evidence, not absence | 0.9843 | [0.9530, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.9108 | [0.7752, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 20 | cosine_rich | linkage detected | 0.9933 | [0.9800, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day | 20 | lda_cosine_rich | linkage detected | 0.9268 | [0.8084, 0.9977] (unreliable: small folds) | 0.046083 |
| cross-day | 20 | pair_logistic | not detected: insufficient evidence, not absence | 0.9827 | [0.9343, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 50 | cosine | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.9444 | [0.6667, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 50 | cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 50 | lda_cosine_rich | linkage detected | 0.9583 | [0.8333, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day | 50 | pair_logistic | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 50 | cosine | not detected: insufficient evidence, not absence | 0.9913 | [0.9653, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.8906 | [0.6736, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day | 50 | cosine_rich | linkage detected | 0.9983 | [0.9931, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day | 50 | lda_cosine_rich | linkage detected | 0.9045 | [0.6562, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day | 50 | pair_logistic | not detected: insufficient evidence, not absence | 1.0000 | [1.0000, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| cross-day | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |

- same-day, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator pair_logistic: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day': 5, 'cross-day': 5}

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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `da99faa92c3dc1107c4a09ddf5e1d0f06b033a5a3f423f00b70f9d773f583a74`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
