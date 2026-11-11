# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-d1-d23 | 1 | bayes_factor | not detected: insufficient evidence, not absence | 0.5052 | [0.4958, 0.5150] | 0.23048 |
| cross-day-d1-d23 | 5 | bayes_factor | linkage detected | 0.5158 | [0.4989, 0.5336] | 0.00059994 |
| cross-day-d1-d23 | 10 | bayes_factor | linkage detected | 0.5240 | [0.5029, 0.5456] | 0.00059994 |
| cross-day-d1-d23 | 20 | bayes_factor | linkage detected | 0.5329 | [0.5093, 0.5581] | 0.00059994 |
| cross-day-d1-d23 | 50 | bayes_factor | linkage detected | 0.5494 | [0.5233, 0.5770] | 0.00059994 |
| cross-day-d1-d23 | 100 | bayes_factor | linkage detected | 0.5473 | [0.5123, 0.5838] | 0.00059994 |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-d1-d23, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-d1-d23, k=100: excluded users ['23', '24', '26', '32']; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'cross-day-d1-d23': 5}

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
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2', '3']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `c6830152e7a23878cecfb0ae5ef9fe949cfb9f3482d5c391bb9434109753933e`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
