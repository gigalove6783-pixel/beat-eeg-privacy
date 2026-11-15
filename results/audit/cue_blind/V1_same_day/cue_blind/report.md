# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day-d2-d3 | 20 | cosine | linkage detected | 0.9992 | [0.9975, 1.0000] | 9.999e-05 |
| same-day-d2-d3 | 20 | lda_cosine | linkage detected | 0.9999 | [0.9997, 1.0000] | 9.999e-05 |
| same-day-d2-d3 | 20 | cosine_rich | linkage detected | 0.9948 | [0.9907, 0.9979] | 9.999e-05 |
| same-day-d2-d3 | 20 | lda_cosine_rich | linkage detected | 0.9994 | [0.9984, 1.0000] | 9.999e-05 |
| cross-day | 20 | — | not_evaluable: day/condition: cross-day analysis is unavailable; supply explicit A/B pairs from different recording days | — | — | — |

- same-day-d2-d3, k=20: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day-d2-d3': 20}

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

## Provenance

Chronological day order: `['1', '2', '3']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `6024b115823c7c271068ce8942cd3b69073bec3c6c8adb2ab0b14a6c28036d26`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
