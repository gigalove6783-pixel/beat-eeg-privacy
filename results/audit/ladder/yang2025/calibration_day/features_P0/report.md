# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-d1-d23 | 1 | cosine | linkage detected | 0.7723 | [0.7128, 0.8284] | 0.00059994 |
| cross-day-d1-d23 | 1 | lda_cosine | linkage detected | 0.7398 | [0.6805, 0.7963] | 0.00059994 |
| cross-day-d1-d23 | 5 | cosine | linkage detected | 0.8225 | [0.7627, 0.8781] | 0.00059994 |
| cross-day-d1-d23 | 5 | lda_cosine | linkage detected | 0.7799 | [0.7117, 0.8419] | 0.00059994 |
| cross-day-d1-d23 | 5 | cosine_rich | linkage detected | 0.8230 | [0.7664, 0.8768] | 0.00049995 |
| cross-day-d1-d23 | 5 | lda_cosine_rich | linkage detected | 0.7811 | [0.7143, 0.8429] | 0.00049995 |
| cross-day-d1-d23 | 10 | cosine | linkage detected | 0.8344 | [0.7747, 0.8898] | 0.00059994 |
| cross-day-d1-d23 | 10 | lda_cosine | linkage detected | 0.8048 | [0.7365, 0.8664] | 0.00059994 |
| cross-day-d1-d23 | 10 | cosine_rich | linkage detected | 0.8437 | [0.7903, 0.8947] | 0.00049995 |
| cross-day-d1-d23 | 10 | lda_cosine_rich | linkage detected | 0.7991 | [0.7304, 0.8631] | 0.00049995 |
| cross-day-d1-d23 | 20 | cosine | linkage detected | 0.8378 | [0.7791, 0.8932] | 0.00059994 |
| cross-day-d1-d23 | 20 | lda_cosine | linkage detected | 0.8240 | [0.7561, 0.8838] | 0.00059994 |
| cross-day-d1-d23 | 20 | cosine_rich | linkage detected | 0.8546 | [0.8022, 0.9046] | 0.00049995 |
| cross-day-d1-d23 | 20 | lda_cosine_rich | linkage detected | 0.8161 | [0.7465, 0.8797] | 0.00049995 |
| cross-day-d1-d23 | 50 | cosine | linkage detected | 0.8443 | [0.7851, 0.9001] | 0.00059994 |
| cross-day-d1-d23 | 50 | lda_cosine | linkage detected | 0.8598 | [0.7946, 0.9156] | 0.00059994 |
| cross-day-d1-d23 | 50 | cosine_rich | linkage detected | 0.8658 | [0.8129, 0.9152] | 0.00049995 |
| cross-day-d1-d23 | 50 | lda_cosine_rich | linkage detected | 0.8713 | [0.8056, 0.9260] | 0.00049995 |
| cross-day-d1-d23 | 100 | cosine | linkage detected | 0.8284 | [0.7586, 0.8895] | 0.00059994 |
| cross-day-d1-d23 | 100 | lda_cosine | linkage detected | 0.8351 | [0.7609, 0.8995] | 0.00059994 |
| cross-day-d1-d23 | 100 | cosine_rich | linkage detected | 0.8466 | [0.7830, 0.9057] | 0.00049995 |
| cross-day-d1-d23 | 100 | lda_cosine_rich | linkage detected | 0.8488 | [0.7740, 0.9096] | 0.00049995 |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-d1-d23, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=100: excluded users ['23', '24', '26', '32']; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'cross-day-d1-d23': 1}

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

- same-day: not evaluated; supply explicit side conditions
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
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2', '3']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `ab46253863e6d603c4c0a15c77b4e728e1f0e5c245deb411f7fb9f53d14b6e53`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
