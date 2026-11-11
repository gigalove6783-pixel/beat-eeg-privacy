# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-sessionE | 1 | cosine | not detected: insufficient evidence, not absence | 0.5906 | [0.5307, 0.6675] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.6235 | [0.5637, 0.6815] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 5 | cosine | not detected: insufficient evidence, not absence | 0.7025 | [0.6007, 0.8210] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.7633 | [0.6581, 0.8789] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 5 | cosine_rich | linkage detected | 0.7517 | [0.6206, 0.8992] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 5 | lda_cosine_rich | linkage detected | 0.7340 | [0.6389, 0.8256] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 10 | cosine | not detected: insufficient evidence, not absence | 0.7710 | [0.6426, 0.8931] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.8159 | [0.7128, 0.9315] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 10 | cosine_rich | linkage detected | 0.8122 | [0.6678, 0.9766] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 10 | lda_cosine_rich | linkage detected | 0.7786 | [0.6763, 0.8705] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 20 | cosine | not detected: insufficient evidence, not absence | 0.8217 | [0.6914, 0.9348] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.8207 | [0.6767, 0.9707] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 20 | cosine_rich | linkage detected | 0.8478 | [0.7046, 0.9954] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 20 | lda_cosine_rich | linkage detected | 0.8219 | [0.6664, 0.9281] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 50 | cosine | not detected: insufficient evidence, not absence | 0.8519 | [0.7222, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.8056 | [0.6660, 1.0000] (unreliable: small folds) | 0.0553 |
| same-day-sessionE | 50 | cosine_rich | linkage detected | 0.8889 | [0.7361, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 50 | lda_cosine_rich | linkage detected | 0.8333 | [0.6250, 1.0000] (unreliable: small folds) | 0.046083 |
| same-day-sessionE | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| cross-day | 1 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 5 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 10 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 50 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 100 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-sessionE, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=5: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=10: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day-sessionE, k=50: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

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
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `06c309c916265d5807556b7b70c57606d66357f5ccff92a20c304ad9f9e2d9f6`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
