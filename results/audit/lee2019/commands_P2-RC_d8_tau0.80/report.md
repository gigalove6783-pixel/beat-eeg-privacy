# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | bayes_factor | linkage detected | 0.5604 | [0.5446, 0.5758] | 0.00059994 |
| cross-day | 1 | bayes_factor | linkage detected | 0.5368 | [0.5201, 0.5541] | 0.00059994 |
| same-day | 5 | bayes_factor | linkage detected | 0.6653 | [0.6258, 0.7024] | 0.00059994 |
| cross-day | 5 | bayes_factor | linkage detected | 0.6192 | [0.5788, 0.6584] | 0.00059994 |
| same-day | 10 | bayes_factor | linkage detected | 0.7107 | [0.6685, 0.7504] | 0.00059994 |
| cross-day | 10 | bayes_factor | linkage detected | 0.6527 | [0.6066, 0.6951] | 0.00059994 |
| same-day | 20 | bayes_factor | linkage detected | 0.7478 | [0.7034, 0.7902] | 0.00059994 |
| cross-day | 20 | bayes_factor | linkage detected | 0.6772 | [0.6266, 0.7218] | 0.00059994 |
| same-day | 50 | bayes_factor | linkage detected | 0.7866 | [0.7443, 0.8271] | 0.00059994 |
| cross-day | 50 | bayes_factor | linkage detected | 0.7072 | [0.6532, 0.7543] | 0.00059994 |
| same-day | 100 | bayes_factor | linkage detected | 0.8206 | [0.7782, 0.8593] | 0.00059994 |
| cross-day | 100 | bayes_factor | linkage detected | 0.7454 | [0.6950, 0.7923] | 0.00059994 |

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
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `4ca0a56010dc32e253cd13ce59d32275c1ddcc99f9209ec05d9a1827e6afc24b`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
