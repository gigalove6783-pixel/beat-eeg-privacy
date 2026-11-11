# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-sessionT-sessionE | 1 | cosine | not detected: insufficient evidence, not absence | 0.4920 | [0.4753, 0.5074] (unreliable: small folds) | 1 |
| cross-day-sessionT-sessionE | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.5010 | [0.4790, 0.5224] (unreliable: small folds) | 0.94009 |
| cross-day-sessionT-sessionE | 5 | cosine | not detected: insufficient evidence, not absence | 0.5069 | [0.4801, 0.5383] (unreliable: small folds) | 1 |
| cross-day-sessionT-sessionE | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.5136 | [0.4810, 0.5479] (unreliable: small folds) | 0.64516 |
| cross-day-sessionT-sessionE | 5 | cosine_rich | not detected: insufficient evidence, not absence | 0.5969 | [0.4835, 0.7071] (unreliable: small folds) | 0.32258 |
| cross-day-sessionT-sessionE | 5 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.5418 | [0.4736, 0.6244] (unreliable: small folds) | 0.59908 |
| cross-day-sessionT-sessionE | 10 | cosine | not detected: insufficient evidence, not absence | 0.4905 | [0.4555, 0.5296] (unreliable: small folds) | 1 |
| cross-day-sessionT-sessionE | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.5128 | [0.4479, 0.5674] (unreliable: small folds) | 0.94009 |
| cross-day-sessionT-sessionE | 10 | cosine_rich | not detected: insufficient evidence, not absence | 0.6221 | [0.4512, 0.7732] (unreliable: small folds) | 0.32258 |
| cross-day-sessionT-sessionE | 10 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.5859 | [0.4451, 0.7227] (unreliable: small folds) | 0.31797 |
| cross-day-sessionT-sessionE | 20 | cosine | not detected: insufficient evidence, not absence | 0.5108 | [0.4531, 0.5707] (unreliable: small folds) | 1 |
| cross-day-sessionT-sessionE | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.5365 | [0.4618, 0.6057] (unreliable: small folds) | 0.63594 |
| cross-day-sessionT-sessionE | 20 | cosine_rich | not detected: insufficient evidence, not absence | 0.6371 | [0.4528, 0.7974] (unreliable: small folds) | 0.32258 |
| cross-day-sessionT-sessionE | 20 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.6070 | [0.4519, 0.7591] (unreliable: small folds) | 0.29493 |
| cross-day-sessionT-sessionE | 50 | cosine | not detected: insufficient evidence, not absence | 0.4763 | [0.3727, 0.5885] (unreliable: small folds) | 1 |
| cross-day-sessionT-sessionE | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.5295 | [0.4074, 0.6580] (unreliable: small folds) | 0.94009 |
| cross-day-sessionT-sessionE | 50 | cosine_rich | not detected: insufficient evidence, not absence | 0.6360 | [0.3993, 0.8524] (unreliable: small folds) | 0.32258 |
| cross-day-sessionT-sessionE | 50 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.6505 | [0.4409, 0.8134] (unreliable: small folds) | 0.23041 |
| cross-day-sessionT-sessionE | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-sessionT-sessionE, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

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
- cross-day-sessionT-sessionE/cosine: mean AUC below 0.5
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- cross-day-sessionT-sessionE/cosine: mean AUC below 0.5
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- cross-day-sessionT-sessionE/cosine: mean AUC below 0.5
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `08a10bc6662a978440d40654cdf2f1d94c5679bb6a39c936cf953a0b44377bef`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
