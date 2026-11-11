# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | cosine | linkage detected | 0.5620 | [0.5456, 0.5788] | 0.00059994 |
| same-day | 1 | lda_cosine | linkage detected | 0.6129 | [0.5988, 0.6273] | 0.00059994 |
| same-day | 1 | pair_logistic | linkage detected | 0.5679 | [0.5482, 0.5894] | 0.00059994 |
| cross-day | 1 | cosine | not detected: insufficient evidence, not absence | 0.5004 | [0.4926, 0.5082] | 0.52495 |
| cross-day | 1 | lda_cosine | not detected: insufficient evidence, not absence | 0.5070 | [0.4977, 0.5169] | 0.10329 |
| cross-day | 1 | pair_logistic | linkage detected | 0.5169 | [0.5040, 0.5298] | 0.0059994 |
| same-day | 5 | cosine | linkage detected | 0.6184 | [0.5998, 0.6368] | 0.00059994 |
| same-day | 5 | lda_cosine | linkage detected | 0.6643 | [0.6452, 0.6822] | 0.00059994 |
| same-day | 5 | cosine_rich | linkage detected | 0.7110 | [0.6791, 0.7424] | 0.00049995 |
| same-day | 5 | lda_cosine_rich | linkage detected | 0.6959 | [0.6709, 0.7211] | 0.00049995 |
| same-day | 5 | pair_logistic | linkage detected | 0.6106 | [0.5885, 0.6334] | 0.00059994 |
| cross-day | 5 | cosine | not detected: insufficient evidence, not absence | 0.5070 | [0.4965, 0.5168] | 0.12779 |
| cross-day | 5 | lda_cosine | linkage detected | 0.5210 | [0.5050, 0.5381] | 0.00079992 |
| cross-day | 5 | cosine_rich | linkage detected | 0.6299 | [0.5917, 0.6673] | 0.00049995 |
| cross-day | 5 | lda_cosine_rich | linkage detected | 0.5788 | [0.5517, 0.6056] | 0.00049995 |
| cross-day | 5 | pair_logistic | not detected: insufficient evidence, not absence | 0.5090 | [0.4976, 0.5194] | 0.11309 |
| same-day | 10 | cosine | linkage detected | 0.6283 | [0.6075, 0.6477] | 0.00059994 |
| same-day | 10 | lda_cosine | linkage detected | 0.6693 | [0.6481, 0.6903] | 0.00059994 |
| same-day | 10 | cosine_rich | linkage detected | 0.7563 | [0.7212, 0.7899] | 0.00049995 |
| same-day | 10 | lda_cosine_rich | linkage detected | 0.7536 | [0.7230, 0.7842] | 0.00049995 |
| same-day | 10 | pair_logistic | linkage detected | 0.6283 | [0.6072, 0.6472] | 0.00059994 |
| cross-day | 10 | cosine | linkage detected | 0.5152 | [0.5042, 0.5267] | 0.017398 |
| cross-day | 10 | lda_cosine | linkage detected | 0.5333 | [0.5162, 0.5512] | 0.00059994 |
| cross-day | 10 | cosine_rich | linkage detected | 0.6707 | [0.6275, 0.7124] | 0.00049995 |
| cross-day | 10 | lda_cosine_rich | linkage detected | 0.6340 | [0.5951, 0.6724] | 0.00049995 |
| cross-day | 10 | pair_logistic | not detected: insufficient evidence, not absence | 0.5076 | [0.4970, 0.5176] | 0.11309 |
| same-day | 20 | cosine | linkage detected | 0.6421 | [0.6204, 0.6615] | 0.00059994 |
| same-day | 20 | lda_cosine | linkage detected | 0.6720 | [0.6486, 0.6958] | 0.00059994 |
| same-day | 20 | cosine_rich | linkage detected | 0.7844 | [0.7509, 0.8165] | 0.00049995 |
| same-day | 20 | lda_cosine_rich | linkage detected | 0.7892 | [0.7558, 0.8215] | 0.00049995 |
| same-day | 20 | pair_logistic | linkage detected | 0.6429 | [0.6217, 0.6630] | 0.00059994 |
| cross-day | 20 | cosine | linkage detected | 0.5178 | [0.5037, 0.5317] | 0.013199 |
| cross-day | 20 | lda_cosine | linkage detected | 0.5433 | [0.5232, 0.5651] | 0.00059994 |
| cross-day | 20 | cosine_rich | linkage detected | 0.6877 | [0.6406, 0.7314] | 0.00049995 |
| cross-day | 20 | lda_cosine_rich | linkage detected | 0.6746 | [0.6329, 0.7167] | 0.00049995 |
| cross-day | 20 | pair_logistic | linkage detected | 0.5124 | [0.4995, 0.5255] | 0.014499 |
| same-day | 50 | cosine | linkage detected | 0.6423 | [0.6152, 0.6705] | 0.00059994 |
| same-day | 50 | lda_cosine | linkage detected | 0.6682 | [0.6403, 0.6970] | 0.00059994 |
| same-day | 50 | cosine_rich | linkage detected | 0.8068 | [0.7708, 0.8402] | 0.00049995 |
| same-day | 50 | lda_cosine_rich | linkage detected | 0.8294 | [0.7937, 0.8630] | 0.00049995 |
| same-day | 50 | pair_logistic | linkage detected | 0.6649 | [0.6406, 0.6872] | 0.00059994 |
| cross-day | 50 | cosine | linkage detected | 0.5305 | [0.5110, 0.5500] | 0.0017998 |
| cross-day | 50 | lda_cosine | linkage detected | 0.5507 | [0.5254, 0.5777] | 0.00059994 |
| cross-day | 50 | cosine_rich | linkage detected | 0.7094 | [0.6595, 0.7511] | 0.00049995 |
| cross-day | 50 | lda_cosine_rich | linkage detected | 0.7271 | [0.6824, 0.7680] | 0.00049995 |
| cross-day | 50 | pair_logistic | linkage detected | 0.5135 | [0.4992, 0.5275] | 0.046395 |
| same-day | 100 | cosine | linkage detected | 0.6203 | [0.5875, 0.6537] | 0.00059994 |
| same-day | 100 | lda_cosine | linkage detected | 0.6502 | [0.6187, 0.6842] | 0.00059994 |
| same-day | 100 | cosine_rich | linkage detected | 0.8208 | [0.7839, 0.8542] | 0.00049995 |
| same-day | 100 | lda_cosine_rich | linkage detected | 0.8488 | [0.8126, 0.8835] | 0.00049995 |
| same-day | 100 | pair_logistic | linkage detected | 0.6662 | [0.6355, 0.6970] | 0.00059994 |
| cross-day | 100 | cosine | linkage detected | 0.5339 | [0.5096, 0.5601] | 0.010499 |
| cross-day | 100 | lda_cosine | linkage detected | 0.5651 | [0.5385, 0.5949] | 0.00059994 |
| cross-day | 100 | cosine_rich | linkage detected | 0.7254 | [0.6746, 0.7676] | 0.00049995 |
| cross-day | 100 | lda_cosine_rich | linkage detected | 0.7615 | [0.7164, 0.8013] | 0.00049995 |
| cross-day | 100 | pair_logistic | not detected: insufficient evidence, not absence | 0.5113 | [0.4964, 0.5264] | 0.11309 |

- same-day, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=1: excluded users []; strongest comparator pair_logistic: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=100: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=100: excluded users []; strongest comparator lda_cosine_rich: Observed ranking on target users (post hoc), not selected on development users.

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
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `4972423d4c4120e50a149dbc18e77241e06db5ea8a2abca64413ecd3226f5131`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
