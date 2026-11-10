# EQ guarantee and residual leakage (EXPLORATORY - Experiment K)

Generated from the docstrings of beat/exp_k.py `eq_lp` and `residual_rows`.

## eq_lp

```text
Utility-optimal equalizing post-processing for one group.

confusion (n, 2, 3) = P(command | target, user); prior (2,) = target distribution;
wrong_weight w = cost of a wrong move relative to a correct one (w = tau / (1 - tau),
see wrong_weight(); w = 1 gives net progress). Variables: M_u (3 x 3, rows = command,
columns = output, row-stochastic) per user and the shared distribution (marginal:
P(output), 3; confusion: P(output | target), 2 x 3). Maximizes the mean over users of the
weighted utility sum_y prior_y sum_c C_u(y,c) [M_u(c, toward y) - w M_u(c, away from y)]
subject to: marginal  sum_c p_u(c) M_u(c, o) = q(o) for all u, o (p_u = prior @ C_u);
            confusion sum_c C_u(y, c) M_u(c, o) = Q(y, o) for all u, y, o.
"Always hold" is feasible, so the LP is always feasible. Returns M (n,3,3), shared,
optimum and an optimality certificate (strong-duality gap, dual infeasibility).

Guarantee (exact for the confusion matrices C_u given to the LP), per variant:
- EQ-confusion: every user of a group has the same P(output | target) = Q(y, .). If each
  trial is post-processed independently, the target sequence does not depend on the
  user, and commands are conditionally independent across trials given the targets and
  the user with a fixed C_u, then all users of a group have the same distribution of
  output sequences given the targets. Every attacker - target-aware or target-blind,
  any number of commands k - then identifies a user within the group at exactly chance
  1/|group|.
- EQ-marginal: every user of a group has the same P(output) = q under the assumed target
  prior. It covers target-blind attackers only, and only if targets are i.i.d. with
  that prior and independent of the user (sequences additionally need the same
  conditional independence). A target-aware attacker still sees P(output | target),
  which may differ between users of a group.
- Group membership is revealed: an attacker can still learn the group, which carries at
  most log2(number of groups) <= log2(N / g) bits; EQ bounds identification within a
  group only.
- The equality holds for the estimates the LP receives (smoothed calibration counts).
  Finite calibration data (estimation error) and any calibration -> deployment change
  (day-to-day drift, decoder mismatch, sequential dependence, phase shifts) leave
  residual leakage; residual_rows measures it (results/exp_k/residual_leakage.csv).
- Optimality: among per-trial post-processing channels of the decoded command that
  achieve the equality for these estimates, M attains the maximal mean weighted utility
  (certified by the returned duality gap and dual feasibility).
```

## residual_rows (results/exp_k/residual_leakage.csv)

```text
Residual-leakage decomposition for one EQ cell (matrices from the weighted-utility LP).

Components (one row each; residual_summary averages replicates and takes differences):
total      : matrices fitted on the cross-fitted calibration day, measured on the real
             deployment day;
estimation : no-shift simulation - truth = the deployment day's empirical P(c | y, u);
             calibration and deployment streams are redrawn i.i.d. from it with the
             real target sequences, matrices fitted on the synthetic calibration day
             (finite calibration data is the only error source);
floor      : matrices fitted on the truth itself, measured on the same synthetic
             deployment stream - the measurement floor of the finite-sample leakage
             estimators (no estimation error);
split      : real within-day split - fitted on deployment segment A, measured on B.
Derived columns (residual_summary):
total_minus_estimation = total - estimation: everything the i.i.d. same-day simulation
             leaves out - day-to-day drift, mismatch between the cross-fitted calibration
             decoder outputs and the final model's deployment outputs, sequential
             dependence within the deployment day and, for Lee2019, the offline -> online
             phase shift between the deployment segments used by the within-day attacker;
             it is not drift alone;
estimation_minus_floor = estimation - floor: leakage attributable to finite calibration
             data beyond the estimator floor.
Measures: permutation-corrected identity bits per trial given the group (blind, aware),
the strongest within-day Bayes within-group excess at the reference k values, and the
deployment weighted utility of the fitted matrices (utility_weighted).
```
