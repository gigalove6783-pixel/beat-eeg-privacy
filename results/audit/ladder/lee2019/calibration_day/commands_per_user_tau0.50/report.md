# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| cross-day-session1-session2 | 1 | bayes_factor | linkage detected | 0.5179 | [0.5084, 0.5277] | 0.00059994 |
| cross-day-session1-session2 | 5 | bayes_factor | linkage detected | 0.5596 | [0.5329, 0.5885] | 0.00059994 |
| cross-day-session1-session2 | 10 | bayes_factor | linkage detected | 0.5923 | [0.5575, 0.6297] | 0.00059994 |
| cross-day-session1-session2 | 20 | bayes_factor | linkage detected | 0.6178 | [0.5762, 0.6605] | 0.00059994 |
| cross-day-session1-session2 | 50 | bayes_factor | linkage detected | 0.6434 | [0.5934, 0.6921] | 0.00059994 |
| cross-day-session1-session2 | 100 | bayes_factor | linkage detected | 0.6647 | [0.6103, 0.7163] | 0.00059994 |
| same-day | 1 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 5 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 10 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 20 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 50 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |
| same-day | 100 | — | not_evaluable: segment/condition: same-day analysis is unavailable; supply at least two disjoint within-day segments and explicit A/B pairs, or omit segment labels to derive halves | — | — | — |

- cross-day-session1-session2, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day-session1-session2, k=100: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

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
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `ef0d9e8b21caddc68e1afcdee20794e9024e4cb7c0bb6539bda33eadf18237b4`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
