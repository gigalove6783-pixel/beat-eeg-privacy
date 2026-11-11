# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-session1-session2 | 1 | cosine | linkage detected | 0.8772 | [0.8469, 0.9042] | 0.00059994 |
| cross-day-session1-session2 | 1 | lda_cosine | linkage detected | 0.8586 | [0.8193, 0.8925] | 0.00059994 |
| cross-day-session1-session2 | 5 | cosine | linkage detected | 0.9287 | [0.8938, 0.9561] | 0.00059994 |
| cross-day-session1-session2 | 5 | lda_cosine | linkage detected | 0.9286 | [0.8966, 0.9554] | 0.00059994 |
| cross-day-session1-session2 | 5 | cosine_rich | linkage detected | 0.9310 | [0.8996, 0.9557] | 0.00049995 |
| cross-day-session1-session2 | 5 | lda_cosine_rich | linkage detected | 0.9256 | [0.8948, 0.9516] | 0.00049995 |
| cross-day-session1-session2 | 10 | cosine | linkage detected | 0.9390 | [0.9056, 0.9652] | 0.00059994 |
| cross-day-session1-session2 | 10 | lda_cosine | linkage detected | 0.9438 | [0.9158, 0.9663] | 0.00059994 |
| cross-day-session1-session2 | 10 | cosine_rich | linkage detected | 0.9461 | [0.9182, 0.9682] | 0.00049995 |
| cross-day-session1-session2 | 10 | lda_cosine_rich | linkage detected | 0.9359 | [0.9074, 0.9588] | 0.00049995 |
| cross-day-session1-session2 | 20 | cosine | linkage detected | 0.9434 | [0.9111, 0.9694] | 0.00059994 |
| cross-day-session1-session2 | 20 | lda_cosine | linkage detected | 0.9518 | [0.9251, 0.9741] | 0.00059994 |
| cross-day-session1-session2 | 20 | cosine_rich | linkage detected | 0.9545 | [0.9292, 0.9744] | 0.00049995 |
| cross-day-session1-session2 | 20 | lda_cosine_rich | linkage detected | 0.9451 | [0.9191, 0.9670] | 0.00049995 |
| cross-day-session1-session2 | 50 | cosine | linkage detected | 0.9479 | [0.9162, 0.9731] | 0.00059994 |
| cross-day-session1-session2 | 50 | lda_cosine | linkage detected | 0.9564 | [0.9288, 0.9797] | 0.00059994 |
| cross-day-session1-session2 | 50 | cosine_rich | linkage detected | 0.9618 | [0.9370, 0.9806] | 0.00049995 |
| cross-day-session1-session2 | 50 | lda_cosine_rich | linkage detected | 0.9553 | [0.9287, 0.9787] | 0.00049995 |
| cross-day-session1-session2 | 100 | cosine | linkage detected | 0.9518 | [0.9210, 0.9761] | 0.00059994 |
| cross-day-session1-session2 | 100 | lda_cosine | linkage detected | 0.9594 | [0.9310, 0.9822] | 0.00059994 |
| cross-day-session1-session2 | 100 | cosine_rich | linkage detected | 0.9666 | [0.9421, 0.9848] | 0.00049995 |
| cross-day-session1-session2 | 100 | lda_cosine_rich | linkage detected | 0.9671 | [0.9448, 0.9856] | 0.00049995 |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-session1-session2, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=50: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=100: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'cross-day-session1-session2': 1}

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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
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
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `df98ed3b8600876cbdbd9e8e4d49a6d4ec0b60cb1cf5188152c3fe806c39a445`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
