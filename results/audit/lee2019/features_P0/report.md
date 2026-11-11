# Transmitted-output leakage audit

Unseen by the attacker's comparator, not necessarily unseen by the BCI. No anonymity, safety or legal conclusion. Transmitted outputs only.

## Same-day beside cross-day

| Condition | k | Comparator | Status / reading | Mean AUC | 95% interval | Holm p |
|---|---:|---|---|---:|---|---:|
| same-day | 1 | cosine | linkage detected | 0.9816 | [0.9753, 0.9870] | 0.00059994 |
| same-day | 1 | lda_cosine | linkage detected | 0.9846 | [0.9780, 0.9901] | 0.00059994 |
| same-day | 1 | pair_logistic | linkage detected | 0.9619 | [0.9506, 0.9727] | 0.00059994 |
| cross-day | 1 | cosine | linkage detected | 0.8761 | [0.8448, 0.9041] | 0.00059994 |
| cross-day | 1 | lda_cosine | linkage detected | 0.8548 | [0.8135, 0.8902] | 0.00059994 |
| cross-day | 1 | pair_logistic | linkage detected | 0.8125 | [0.7746, 0.8457] | 0.00059994 |
| same-day | 5 | cosine | linkage detected | 0.9984 | [0.9963, 0.9996] | 0.00059994 |
| same-day | 5 | lda_cosine | linkage detected | 0.9993 | [0.9987, 0.9998] | 0.00059994 |
| same-day | 5 | cosine_rich | linkage detected | 0.9965 | [0.9939, 0.9984] | 0.00049995 |
| same-day | 5 | lda_cosine_rich | linkage detected | 0.9983 | [0.9969, 0.9994] | 0.00049995 |
| same-day | 5 | pair_logistic | linkage detected | 0.9944 | [0.9908, 0.9972] | 0.00059994 |
| cross-day | 5 | cosine | linkage detected | 0.9302 | [0.8973, 0.9569] | 0.00059994 |
| cross-day | 5 | lda_cosine | linkage detected | 0.9271 | [0.8948, 0.9544] | 0.00059994 |
| cross-day | 5 | cosine_rich | linkage detected | 0.9322 | [0.9024, 0.9563] | 0.00049995 |
| cross-day | 5 | lda_cosine_rich | linkage detected | 0.9221 | [0.8909, 0.9487] | 0.00049995 |
| cross-day | 5 | pair_logistic | linkage detected | 0.8673 | [0.8276, 0.9010] | 0.00059994 |
| same-day | 10 | cosine | linkage detected | 0.9992 | [0.9977, 0.9999] | 0.00059994 |
| same-day | 10 | lda_cosine | linkage detected | 0.9999 | [0.9998, 1.0000] | 0.00059994 |
| same-day | 10 | cosine_rich | linkage detected | 0.9985 | [0.9968, 0.9996] | 0.00049995 |
| same-day | 10 | lda_cosine_rich | linkage detected | 0.9992 | [0.9981, 0.9999] | 0.00049995 |
| same-day | 10 | pair_logistic | linkage detected | 0.9966 | [0.9939, 0.9986] | 0.00059994 |
| cross-day | 10 | cosine | linkage detected | 0.9375 | [0.9048, 0.9643] | 0.00059994 |
| cross-day | 10 | lda_cosine | linkage detected | 0.9415 | [0.9128, 0.9649] | 0.00059994 |
| cross-day | 10 | cosine_rich | linkage detected | 0.9447 | [0.9173, 0.9669] | 0.00049995 |
| cross-day | 10 | lda_cosine_rich | linkage detected | 0.9347 | [0.9055, 0.9589] | 0.00049995 |
| cross-day | 10 | pair_logistic | linkage detected | 0.8709 | [0.8302, 0.9062] | 0.00059994 |
| same-day | 20 | cosine | linkage detected | 0.9994 | [0.9979, 1.0000] | 0.00059994 |
| same-day | 20 | lda_cosine | linkage detected | 0.9999 | [0.9996, 1.0000] | 0.00059994 |
| same-day | 20 | cosine_rich | linkage detected | 0.9992 | [0.9979, 0.9999] | 0.00049995 |
| same-day | 20 | lda_cosine_rich | linkage detected | 0.9993 | [0.9983, 0.9999] | 0.00049995 |
| same-day | 20 | pair_logistic | linkage detected | 0.9987 | [0.9971, 0.9997] | 0.00059994 |
| cross-day | 20 | cosine | linkage detected | 0.9442 | [0.9125, 0.9696] | 0.00059994 |
| cross-day | 20 | lda_cosine | linkage detected | 0.9521 | [0.9265, 0.9736] | 0.00059994 |
| cross-day | 20 | cosine_rich | linkage detected | 0.9543 | [0.9282, 0.9744] | 0.00049995 |
| cross-day | 20 | lda_cosine_rich | linkage detected | 0.9459 | [0.9199, 0.9673] | 0.00049995 |
| cross-day | 20 | pair_logistic | linkage detected | 0.8754 | [0.8335, 0.9111] | 0.00059994 |
| same-day | 50 | cosine | linkage detected | 0.9997 | [0.9985, 1.0000] | 0.00059994 |
| same-day | 50 | lda_cosine | linkage detected | 1.0000 | [0.9998, 1.0000] | 0.00059994 |
| same-day | 50 | cosine_rich | linkage detected | 0.9993 | [0.9975, 1.0000] | 0.00049995 |
| same-day | 50 | lda_cosine_rich | linkage detected | 0.9998 | [0.9993, 1.0000] | 0.00049995 |
| same-day | 50 | pair_logistic | linkage detected | 0.9996 | [0.9986, 1.0000] | 0.00059994 |
| cross-day | 50 | cosine | linkage detected | 0.9476 | [0.9160, 0.9733] | 0.00059994 |
| cross-day | 50 | lda_cosine | linkage detected | 0.9569 | [0.9291, 0.9796] | 0.00059994 |
| cross-day | 50 | cosine_rich | linkage detected | 0.9617 | [0.9367, 0.9803] | 0.00049995 |
| cross-day | 50 | lda_cosine_rich | linkage detected | 0.9561 | [0.9299, 0.9788] | 0.00049995 |
| cross-day | 50 | pair_logistic | linkage detected | 0.8775 | [0.8384, 0.9126] | 0.00059994 |
| same-day | 100 | cosine | linkage detected | 0.9995 | [0.9979, 1.0000] | 0.00059994 |
| same-day | 100 | lda_cosine | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00059994 |
| same-day | 100 | cosine_rich | linkage detected | 0.9995 | [0.9979, 1.0000] | 0.00049995 |
| same-day | 100 | lda_cosine_rich | linkage detected | 1.0000 | [1.0000, 1.0000] | 0.00049995 |
| same-day | 100 | pair_logistic | linkage detected | 0.9990 | [0.9958, 1.0000] | 0.00059994 |
| cross-day | 100 | cosine | linkage detected | 0.9518 | [0.9210, 0.9761] | 0.00059994 |
| cross-day | 100 | lda_cosine | linkage detected | 0.9594 | [0.9310, 0.9822] | 0.00059994 |
| cross-day | 100 | cosine_rich | linkage detected | 0.9666 | [0.9421, 0.9848] | 0.00049995 |
| cross-day | 100 | lda_cosine_rich | linkage detected | 0.9671 | [0.9448, 0.9856] | 0.00049995 |
| cross-day | 100 | pair_logistic | linkage detected | 0.8923 | [0.8508, 0.9272] | 0.00059994 |

- same-day, k=1: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=1: excluded users []; strongest comparator cosine: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=5: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=5: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=10: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=10: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=20: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=20: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=50: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
- cross-day, k=50: excluded users []; strongest comparator cosine_rich: Observed ranking on target users (post hoc), not selected on development users.
- same-day, k=100: excluded users []; strongest comparator lda_cosine: Observed ranking on target users (post hoc), not selected on development users.
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

- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
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
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: Sensitivity scenario under independence; not population risk
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported
- Gallery: independence model not supported

## Provenance

Chronological day order: `['1', '2']`; source: `explicit`.
Source SHA-256: `73ce3bec1b47c4433d7ab7fa98768022b6e8c1d3f495ec53a1833e98ec8a3525`
Configuration SHA-256: `4972423d4c4120e50a149dbc18e77241e06db5ea8a2abca64413ecd3226f5131`
Input and per-cell population hashes, exclusions and all numeric metrics are in `report.json`.
