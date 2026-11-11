# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-session1-session2 | 1 | cosine | linkage detected | 0.5063 | [0.4987, 0.5137] | 0.0055994 |
| cross-day-session1-session2 | 1 | lda_cosine | linkage detected | 0.5145 | [0.5031, 0.5249] | 0.00059994 |
| cross-day-session1-session2 | 5 | cosine | linkage detected | 0.5097 | [0.4992, 0.5207] | 0.014299 |
| cross-day-session1-session2 | 5 | lda_cosine | linkage detected | 0.5274 | [0.5110, 0.5444] | 0.00059994 |
| cross-day-session1-session2 | 5 | cosine_rich | linkage detected | 0.6357 | [0.5967, 0.6736] | 0.00049995 |
| cross-day-session1-session2 | 5 | lda_cosine_rich | linkage detected | 0.5914 | [0.5633, 0.6186] | 0.00049995 |
| cross-day-session1-session2 | 10 | cosine | linkage detected | 0.5192 | [0.5079, 0.5301] | 0.00079992 |
| cross-day-session1-session2 | 10 | lda_cosine | linkage detected | 0.5390 | [0.5204, 0.5580] | 0.00059994 |
| cross-day-session1-session2 | 10 | cosine_rich | linkage detected | 0.6705 | [0.6259, 0.7117] | 0.00049995 |
| cross-day-session1-session2 | 10 | lda_cosine_rich | linkage detected | 0.6380 | [0.5985, 0.6749] | 0.00049995 |
| cross-day-session1-session2 | 20 | cosine | linkage detected | 0.5253 | [0.5108, 0.5413] | 0.00059994 |
| cross-day-session1-session2 | 20 | lda_cosine | linkage detected | 0.5523 | [0.5299, 0.5760] | 0.00059994 |
| cross-day-session1-session2 | 20 | cosine_rich | linkage detected | 0.6877 | [0.6411, 0.7307] | 0.00049995 |
| cross-day-session1-session2 | 20 | lda_cosine_rich | linkage detected | 0.6722 | [0.6281, 0.7149] | 0.00049995 |
| cross-day-session1-session2 | 50 | cosine | linkage detected | 0.5353 | [0.5141, 0.5570] | 0.00059994 |
| cross-day-session1-session2 | 50 | lda_cosine | linkage detected | 0.5544 | [0.5264, 0.5852] | 0.00059994 |
| cross-day-session1-session2 | 50 | cosine_rich | linkage detected | 0.7085 | [0.6594, 0.7503] | 0.00049995 |
| cross-day-session1-session2 | 50 | lda_cosine_rich | linkage detected | 0.7227 | [0.6755, 0.7658] | 0.00049995 |
| cross-day-session1-session2 | 100 | cosine | linkage detected | 0.5402 | [0.5153, 0.5682] | 0.0023998 |
| cross-day-session1-session2 | 100 | lda_cosine | linkage detected | 0.5719 | [0.5400, 0.6052] | 0.00059994 |
| cross-day-session1-session2 | 100 | cosine_rich | linkage detected | 0.7248 | [0.6744, 0.7675] | 0.00049995 |
| cross-day-session1-session2 | 100 | lda_cosine_rich | linkage detected | 0.7586 | [0.7120, 0.8006] | 0.00049995 |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-session1-session2, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
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
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `df98ed3b8600876cbdbd9e8e4d49a6d4ec0b60cb1cf5188152c3fe806c39a445`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
