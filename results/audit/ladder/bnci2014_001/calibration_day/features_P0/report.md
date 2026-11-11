# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-sessionT-sessionE | 1 | cosine | not detected: insufficient evidence, not absence | 0.9641 | [0.9083, 0.9944] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.8150 | [0.6763, 0.9537] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 5 | cosine | not detected: insufficient evidence, not absence | 0.9835 | [0.9513, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.8944 | [0.7886, 0.9835] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 5 | cosine_rich | linkage detected | 0.9876 | [0.9630, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 5 | lda_cosine_rich | linkage detected | 0.9138 | [0.8230, 0.9837] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 10 | cosine | not detected: insufficient evidence, not absence | 0.9832 | [0.9511, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.9466 | [0.8651, 0.9997] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 10 | cosine_rich | linkage detected | 0.9872 | [0.9599, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 10 | lda_cosine_rich | linkage detected | 0.9429 | [0.8803, 0.9976] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 20 | cosine | not detected: insufficient evidence, not absence | 0.9834 | [0.9414, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.9191 | [0.7689, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 20 | cosine_rich | linkage detected | 0.9921 | [0.9697, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 20 | lda_cosine_rich | linkage detected | 0.9224 | [0.7874, 0.9967] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 50 | cosine | not detected: insufficient evidence, not absence | 0.9913 | [0.9653, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.8906 | [0.6736, 1.0000] (unreliable: small folds) | 0.0553 |
| cross-day-sessionT-sessionE | 50 | cosine_rich | linkage detected | 0.9983 | [0.9931, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 50 | lda_cosine_rich | linkage detected | 0.9045 | [0.6562, 1.0000] (unreliable: small folds) | 0.046083 |
| cross-day-sessionT-sessionE | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-sessionT-sessionE, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-sessionT-sessionE, k=50: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'cross-day-sessionT-sessionE': 5}

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
- Gallery: independence model not supported
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
Configuration SHA-256: `08a10bc6662a978440d40654cdf2f1d94c5679bb6a39c936cf953a0b44377bef`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
