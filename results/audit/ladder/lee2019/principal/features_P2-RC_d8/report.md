# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-session2 | 1 | cosine | linkage detected | 0.6304 | [0.5999, 0.6608] | 0.00059994 |
| same-day-session2 | 1 | lda_cosine | linkage detected | 0.7428 | [0.7160, 0.7700] | 0.00059994 |
| same-day-session2 | 5 | cosine | linkage detected | 0.7558 | [0.7203, 0.7887] | 0.00059994 |
| same-day-session2 | 5 | lda_cosine | linkage detected | 0.8871 | [0.8577, 0.9140] | 0.00059994 |
| same-day-session2 | 5 | cosine_rich | linkage detected | 0.8075 | [0.7708, 0.8411] | 0.00049995 |
| same-day-session2 | 5 | lda_cosine_rich | linkage detected | 0.8799 | [0.8532, 0.9066] | 0.00049995 |
| same-day-session2 | 10 | cosine | linkage detected | 0.8111 | [0.7750, 0.8433] | 0.00059994 |
| same-day-session2 | 10 | lda_cosine | linkage detected | 0.9174 | [0.8892, 0.9427] | 0.00059994 |
| same-day-session2 | 10 | cosine_rich | linkage detected | 0.8609 | [0.8251, 0.8930] | 0.00049995 |
| same-day-session2 | 10 | lda_cosine_rich | linkage detected | 0.9333 | [0.9105, 0.9549] | 0.00049995 |
| same-day-session2 | 20 | cosine | linkage detected | 0.8472 | [0.8102, 0.8798] | 0.00059994 |
| same-day-session2 | 20 | lda_cosine | linkage detected | 0.9346 | [0.9075, 0.9589] | 0.00059994 |
| same-day-session2 | 20 | cosine_rich | linkage detected | 0.9003 | [0.8670, 0.9307] | 0.00049995 |
| same-day-session2 | 20 | lda_cosine_rich | linkage detected | 0.9606 | [0.9413, 0.9772] | 0.00049995 |
| same-day-session2 | 50 | cosine | linkage detected | 0.8946 | [0.8568, 0.9257] | 0.00059994 |
| same-day-session2 | 50 | lda_cosine | linkage detected | 0.9521 | [0.9262, 0.9747] | 0.00059994 |
| same-day-session2 | 50 | cosine_rich | linkage detected | 0.9342 | [0.9043, 0.9589] | 0.00049995 |
| same-day-session2 | 50 | lda_cosine_rich | linkage detected | 0.9788 | [0.9630, 0.9914] | 0.00049995 |
| same-day-session2 | 100 | cosine | linkage detected | 0.9252 | [0.8866, 0.9579] | 0.00059994 |
| same-day-session2 | 100 | lda_cosine | linkage detected | 0.9589 | [0.9332, 0.9805] | 0.00059994 |
| same-day-session2 | 100 | cosine_rich | linkage detected | 0.9515 | [0.9205, 0.9752] | 0.00049995 |
| same-day-session2 | 100 | lda_cosine_rich | linkage detected | 0.9826 | [0.9647, 0.9954] | 0.00049995 |
| cross-day | 1 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 5 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 10 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 50 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 100 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-session2, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=5: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=10: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=20: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=100: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day-session2': 1}

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

- cross-day: not evaluated; supply explicit side conditions
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

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `d6950a0a02e16560aef405a686efa42c3706ba6705b51b65a5fc23cada23cbd1`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
