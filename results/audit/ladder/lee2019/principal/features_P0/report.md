# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-session2 | 1 | cosine | linkage detected | 0.9860 | [0.9801, 0.9911] | 0.00059994 |
| same-day-session2 | 1 | lda_cosine | linkage detected | 0.9933 | [0.9893, 0.9963] | 0.00059994 |
| same-day-session2 | 5 | cosine | linkage detected | 0.9997 | [0.9991, 1.0000] | 0.00059994 |
| same-day-session2 | 5 | lda_cosine | linkage detected | 0.9998 | [0.9993, 1.0000] | 0.00059994 |
| same-day-session2 | 5 | cosine_rich | linkage detected | 0.9987 | [0.9970, 0.9997] | 0.00049995 |
| same-day-session2 | 5 | lda_cosine_rich | linkage detected | 0.9996 | [0.9990, 1.0000] | 0.00049995 |
| same-day-session2 | 10 | cosine | linkage detected | 1.0000 | [0.9999, 1.0000] | 0.00059994 |
| same-day-session2 | 10 | lda_cosine | linkage detected | 0.9999 | [0.9997, 1.0000] | 0.00059994 |
| same-day-session2 | 10 | cosine_rich | linkage detected | 0.9997 | [0.9991, 1.0000] | 0.00049995 |
| same-day-session2 | 10 | lda_cosine_rich | linkage detected | 0.9999 | [0.9996, 1.0000] | 0.00049995 |
| same-day-session2 | 20 | cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day-session2 | 20 | lda_cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day-session2 | 20 | cosine_rich | linkage detected | 0.9999 | [0.9998, 1.0000] | 0.00049995 |
| same-day-session2 | 20 | lda_cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00049995 |
| same-day-session2 | 50 | cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day-session2 | 50 | lda_cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day-session2 | 50 | cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00049995 |
| same-day-session2 | 50 | lda_cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00049995 |
| same-day-session2 | 100 | cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day-session2 | 100 | lda_cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day-session2 | 100 | cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00049995 |
| same-day-session2 | 100 | lda_cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00049995 |
| cross-day | 1 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 5 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 10 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 50 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |
| cross-day | 100 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-session2, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=5: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=10: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=20: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=50: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day-session2, k=100: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.

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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `d6950a0a02e16560aef405a686efa42c3706ba6705b51b65a5fc23cada23cbd1`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
