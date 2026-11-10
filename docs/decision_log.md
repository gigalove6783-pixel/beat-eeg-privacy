# Decision log

Design changes with their reasons, and the E-seal record. Entries are appended, never rewritten.

| Date | Entry | Reason |
|---|---|---|
| 2026-11-10 | Analysis plan v1 adapted from the 2026-10-09 proposal; deviations listed in `docs/analysis_plan.md`. | Results presentation on 2026-11-16. |
| 2026-11-10 | Data fetched from the TU Graz host; checksums printed by `scripts/download_data.py` and stored next to the raw files (`SHA256SUMS`). | BNCI website connection timed out. |
| 2026-11-10 | Experiment B protection requires both the strongest-over-scenarios comparison and the main-scenario comparison to pass. | Operationalizes "the auxiliary scenario alone never establishes protection". |
| 2026-11-10 | T-only development showed P3 identity BA rising with λ (CV 0.54 at λ 0.01 to 0.97 at λ 1) while the training-time adversary head stays at or below chance (Lead check, held-out T run 1: head 0.08–0.11, independent logistic regression 0.56 at λ 0 to 0.995 at λ 1). GRL sign verified correct. Added: adversary-head scores reported beside independent attackers (never as evidence), epoch-budget checkpoints every epoch (RC optimum sat at the lowest checkpoint), MLP attacker convergence. Development re-run before the seal. | Known failure of gradient-reversal removal; no session-E data involved. |
| 2026-11-10 | Pre-seal disclosure: one loader sanity check scored per-subject tangent space + logistic regression trained on all 6 T runs on session E (mean BA 0.758; subjects 2 and 5 lowest, 3 and 8 highest). This equals experiment A's B3 at 6 runs. No setting was chosen or changed from it; experiment A has no tunable choices. | Validate trial extraction against the known subject pattern. |
