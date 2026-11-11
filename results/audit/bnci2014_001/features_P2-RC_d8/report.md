# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | cosine | not detected: insufficient evidence, not absence | 0.5647 | [0.5029, 0.6223] (unreliable: small folds) | 0.0553 |
| same-day | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.5546 | [0.5197, 0.5970] (unreliable: small folds) | 0.0553 |
| same-day | 1 | pair_logistic | not detected: insufficient evidence, not absence | 0.5223 | [0.4900, 0.5630] (unreliable: small folds) | 0.069124 |
| cross-day | 1 | cosine | not detected: insufficient evidence, not absence | 0.5037 | [0.4823, 0.5312] (unreliable: small folds) | 1 |
| cross-day | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.5017 | [0.4758, 0.5317] (unreliable: small folds) | 0.4977 |
| cross-day | 1 | pair_logistic | not detected: insufficient evidence, not absence | 0.4972 | [0.4670, 0.5287] (unreliable: small folds) | 0.80184 |
| same-day | 5 | cosine | not detected: insufficient evidence, not absence | 0.6133 | [0.5522, 0.6713] (unreliable: small folds) | 0.0553 |
| same-day | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.6446 | [0.5936, 0.6991] (unreliable: small folds) | 0.0553 |
| same-day | 5 | cosine_rich | linkage detected | 0.6738 | [0.5503, 0.7945] (unreliable: small folds) | 0.046083 |
| same-day | 5 | lda_cosine_rich | linkage detected | 0.6216 | [0.5516, 0.6871] (unreliable: small folds) | 0.046083 |
| same-day | 5 | pair_logistic | not detected: insufficient evidence, not absence | 0.5323 | [0.4928, 0.5640] (unreliable: small folds) | 0.069124 |
| cross-day | 5 | cosine | not detected: insufficient evidence, not absence | 0.5062 | [0.4632, 0.5453] (unreliable: small folds) | 1 |
| cross-day | 5 | lda_cosine | not detected: insufficient evidence, not absence | 0.5144 | [0.4733, 0.5548] (unreliable: small folds) | 0.41475 |
| cross-day | 5 | cosine_rich | not detected: insufficient evidence, not absence | 0.6169 | [0.4980, 0.7362] (unreliable: small folds) | 0.18433 |
| cross-day | 5 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.5769 | [0.5116, 0.6453] (unreliable: small folds) | 0.11521 |
| cross-day | 5 | pair_logistic | not detected: insufficient evidence, not absence | 0.5065 | [0.4669, 0.5504] (unreliable: small folds) | 1 |
| same-day | 10 | cosine | not detected: insufficient evidence, not absence | 0.6235 | [0.5678, 0.6836] (unreliable: small folds) | 0.0553 |
| same-day | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.6590 | [0.5839, 0.7390] (unreliable: small folds) | 0.0553 |
| same-day | 10 | cosine_rich | linkage detected | 0.7099 | [0.5753, 0.8436] (unreliable: small folds) | 0.046083 |
| same-day | 10 | lda_cosine_rich | linkage detected | 0.6810 | [0.5841, 0.7830] (unreliable: small folds) | 0.046083 |
| same-day | 10 | pair_logistic | not detected: insufficient evidence, not absence | 0.5381 | [0.5048, 0.5692] (unreliable: small folds) | 0.069124 |
| cross-day | 10 | cosine | not detected: insufficient evidence, not absence | 0.4785 | [0.4090, 0.5385] (unreliable: small folds) | 1 |
| cross-day | 10 | lda_cosine | not detected: insufficient evidence, not absence | 0.5146 | [0.4717, 0.5637] (unreliable: small folds) | 0.4977 |
| cross-day | 10 | cosine_rich | not detected: insufficient evidence, not absence | 0.6076 | [0.4406, 0.7688] (unreliable: small folds) | 0.35945 |
| cross-day | 10 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.5864 | [0.4665, 0.7037] (unreliable: small folds) | 0.15207 |
| cross-day | 10 | pair_logistic | not detected: insufficient evidence, not absence | 0.4986 | [0.4599, 0.5296] (unreliable: small folds) | 1 |
| same-day | 20 | cosine | not detected: insufficient evidence, not absence | 0.6635 | [0.6093, 0.7217] (unreliable: small folds) | 0.0553 |
| same-day | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.6833 | [0.6012, 0.7594] (unreliable: small folds) | 0.0553 |
| same-day | 20 | cosine_rich | linkage detected | 0.7344 | [0.6099, 0.8579] (unreliable: small folds) | 0.046083 |
| same-day | 20 | lda_cosine_rich | linkage detected | 0.7265 | [0.6211, 0.8308] (unreliable: small folds) | 0.046083 |
| same-day | 20 | pair_logistic | not detected: insufficient evidence, not absence | 0.5523 | [0.5076, 0.5957] (unreliable: small folds) | 0.0553 |
| cross-day | 20 | cosine | not detected: insufficient evidence, not absence | 0.5147 | [0.4422, 0.5746] (unreliable: small folds) | 1 |
| cross-day | 20 | lda_cosine | not detected: insufficient evidence, not absence | 0.5590 | [0.4639, 0.6378] (unreliable: small folds) | 0.46083 |
| cross-day | 20 | cosine_rich | not detected: insufficient evidence, not absence | 0.6412 | [0.4453, 0.7934] (unreliable: small folds) | 0.23963 |
| cross-day | 20 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.6129 | [0.4504, 0.7658] (unreliable: small folds) | 0.15207 |
| cross-day | 20 | pair_logistic | not detected: insufficient evidence, not absence | 0.4872 | [0.4630, 0.5127] (unreliable: small folds) | 1 |
| same-day | 50 | cosine | not detected: insufficient evidence, not absence | 0.6694 | [0.5708, 0.7667] (unreliable: small folds) | 0.0553 |
| same-day | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.6833 | [0.5833, 0.7889] (unreliable: small folds) | 0.0553 |
| same-day | 50 | cosine_rich | linkage detected | 0.7750 | [0.6500, 0.8833] (unreliable: small folds) | 0.046083 |
| same-day | 50 | lda_cosine_rich | linkage detected | 0.7583 | [0.6125, 0.8833] (unreliable: small folds) | 0.046083 |
| same-day | 50 | pair_logistic | not detected: insufficient evidence, not absence | 0.5278 | [0.4389, 0.6111] (unreliable: small folds) | 0.58065 |
| cross-day | 50 | cosine | not detected: insufficient evidence, not absence | 0.4889 | [0.3917, 0.5934] (unreliable: small folds) | 1 |
| cross-day | 50 | lda_cosine | not detected: insufficient evidence, not absence | 0.5708 | [0.4458, 0.6896] (unreliable: small folds) | 0.4977 |
| cross-day | 50 | cosine_rich | not detected: insufficient evidence, not absence | 0.6542 | [0.4385, 0.8531] (unreliable: small folds) | 0.26267 |
| cross-day | 50 | lda_cosine_rich | not detected: insufficient evidence, not absence | 0.6608 | [0.4770, 0.8219] (unreliable: small folds) | 0.12903 |
| cross-day | 50 | pair_logistic | not detected: insufficient evidence, not absence | 0.4944 | [0.4333, 0.5495] (unreliable: small folds) | 1 |
| same-day | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |
| cross-day | 100 | — | not_evaluable: k=100, user: 9/9 excluded; supply a window in every required segment for >=90% of users and at least 6 users, or reduce k | — | — | — |

- same-day, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

## Detection over k

{'same-day': 5, 'cross-day': None}

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

- Below 20 users: estimates are unstable
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- cross-day/pair_logistic: mean AUC below 0.5
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- cross-day/cosine: mean AUC below 0.5
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- cross-day/pair_logistic: mean AUC below 0.5
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- cross-day/pair_logistic: mean AUC below 0.5
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- cross-day/cosine: mean AUC below 0.5
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- cross-day/pair_logistic: mean AUC below 0.5
- Gallery: Sensitivity scenario under independence; not population risk

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `da99faa92c3dc1107c4a09ddf5e1d0f06b033a5a3f423f00b70f9d773f583a74`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
