# Decision log

Design changes with their reasons, and the E-seal record. Entries are appended, never rewritten.

| Date | Entry | Reason |
|---|---|---|
| 2026-11-10 | Analysis plan v1 adapted from the 2026-10-09 proposal; deviations listed in `docs/analysis_plan.md`. | Results presentation on 2026-11-16. |
| 2026-11-10 | Data fetched from the TU Graz host; checksums printed by `scripts/download_data.py` and stored next to the raw files (`SHA256SUMS`). | BNCI website connection timed out. |
| 2026-11-10 | Pre-seal disclosure: one loader sanity check scored per-subject tangent space + logistic regression trained on all 6 T runs on session E (mean BA 0.758; subjects 2 and 5 lowest, 3 and 8 highest). This equals experiment A's B3 at 6 runs. No setting was chosen or changed from it; experiment A has no tunable choices. | Validate trial extraction against the known subject pattern. |
