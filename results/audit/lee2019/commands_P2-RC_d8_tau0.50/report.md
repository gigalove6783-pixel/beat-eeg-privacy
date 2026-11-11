# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | bayes_factor | linkage detected | 0.5315 | [0.5203, 0.5434] | 0.00059994 |
| cross-day | 1 | bayes_factor | linkage detected | 0.5202 | [0.5095, 0.5317] | 0.00059994 |
| same-day | 5 | bayes_factor | linkage detected | 0.6260 | [0.5974, 0.6567] | 0.00059994 |
| cross-day | 5 | bayes_factor | linkage detected | 0.5711 | [0.5419, 0.6022] | 0.00059994 |
| same-day | 10 | bayes_factor | linkage detected | 0.6663 | [0.6318, 0.7026] | 0.00059994 |
| cross-day | 10 | bayes_factor | linkage detected | 0.6019 | [0.5679, 0.6375] | 0.00059994 |
| same-day | 20 | bayes_factor | linkage detected | 0.7023 | [0.6628, 0.7415] | 0.00059994 |
| cross-day | 20 | bayes_factor | linkage detected | 0.6300 | [0.5878, 0.6719] | 0.00059994 |
| same-day | 50 | bayes_factor | linkage detected | 0.7421 | [0.6986, 0.7841] | 0.00059994 |
| cross-day | 50 | bayes_factor | linkage detected | 0.6595 | [0.6136, 0.7044] | 0.00059994 |
| same-day | 100 | bayes_factor | linkage detected | 0.7633 | [0.7137, 0.8112] | 0.00059994 |
| cross-day | 100 | bayes_factor | linkage detected | 0.6796 | [0.6278, 0.7287] | 0.00059994 |

- same-day, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=1: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=100: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=100: excluded users []; strongest comparator bayes_factor: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day': 1, 'cross-day': 1}

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

- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `4ca0a56010dc32e253cd13ce59d32275c1ddcc99f9209ec05d9a1827e6afc24b`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
