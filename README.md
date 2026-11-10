# BEAT: what a motor-imagery BCI reveals about its user

**What a motor-imagery BCI transmits identifies its user — almost perfectly within a day, and
far above chance across days — and two evaluation practices found in published EEG
identity-protection work make defences look far more protective than they are. We pre-registered
these analyses on 9 users (BCI Competition IV 2a) and tested four claims on 54 users
(Lee2019/OpenBMI), where three of the four held.**

[![CI](https://github.com/gigalove6783-pixel/beat-eeg-privacy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gigalove6783-pixel/beat-eeg-privacy/actions/workflows/ci.yml)

> **What this does not show**
> - **Closed sets only.** Identification is among 9 users (chance 0.111) and among 54 users
>   (chance 0.0185). Neither result transfers automatically to other populations or to open-set
>   verification.
> - **No anonymity claim.** Attackers are logistic regression, RBF-SVM and MLP on transmitted
>   outputs, plus covariance-space attackers in G. No white-box or adaptive encoder attack was
>   tried, so every identity score is a lower bound, and an attacker's failure is not
>   anonymisation.
> - **Replay, not closed loop.** The arm replays recorded EEG; no person controlled it. Replay
>   processing time is not online latency.
> - **Exploratory parts.** Only A–C and L are confirmatory. D, E and G are exploratory (D and E
>   reuse held-out data that had already shaped them). L's main scenario links two phases
>   recorded on the same day.
> - **Self-attested seals.** Seals 1–3 are Git commits and tags made by the project itself, all
>   on 2026-11-10. No third party timestamped them, so they show the order of commits, not when
>   held-out data were first seen.
> - **Out of scope.** Device compromise, identifying metadata and unobserved sensitive attributes.

## Headline findings

Identity is measured as balanced accuracy (BA). "2a" is BCI IV 2a (9 users, chance 0.111); "L"
is the Lee2019 replication (54 users, chance 0.0185). **Within-day** numbers train and test the
attacker on outputs from the same recording day (2a: runs 1–3 ↔ 4–6 of session E; L: offline ↔
online phase of session 2). This is the main, pre-registered scenario. **Cross-day** numbers train
on the calibration day and test on the deployment day (2a: T → E; L: session 1 → 2).

1. **Raw features identify everyone within a day, and most users across days.** The full
   Riemannian tangent vector gives identity BA 1.00 on 2a (B, *confirmatory*) and 0.994 on 54
   users (L, *confirmatory*) within a day. Across days the same attackers reach 0.951 on 2a and
   0.520 on 54 users (logistic regression; 0.510 RBF-SVM, 0.368 MLP).
2. **A narrow task-only bottleneck protected best, but identity stayed above chance.** On 2a, a
   1-d bottleneck gives 0.253 (common) / 0.209 (recentred) at task BA 0.707 / 0.706 (within-day).
   No T-CV-selected adversarial defence beat it (D, *exploratory*). On 54 users, the 1-d values
   are 0.057 / 0.085 (*descriptive*, 3 seeds), and a recentred 8-d bottleneck lowers within-day
   identity BA from 0.994 to 0.340 at −0.6 pp task BA (C2, *confirmatory*); across days it gives
   0.054.
3. **Pitfall 1: the co-trained adversary misleads.** At λ = 1 the gradient-reversal identity head
   scores 0.026, below chance, while independent attackers recover 0.951 from the same frozen
   outputs (2a, *confirmatory* run B; the head is a diagnostic). On 54 users the head scores
   0.040 against 0.514 within a day (C1, *confirmatory*); across days the gap is 17.6–19.9 pp.
4. **Commands leak on 2a; the 54-user confirmation failed.** On 2a, within session E,
   left/right/hold commands alone exceed twice chance within five commands in 22 of 24 settings,
   and reach 0.211–0.478 at 20 commands (C, H3, *confirmatory, descriptive*). On 54 users the
   within-day point estimate rises from 0.037 to 0.085 at 20 commands, but **claim C3 was not
   supported** (Holm p = 0.21). Across days, in C3's setting (P2-RC, τ = 0.5, 20 commands),
   command identification stayed near chance (0.019–0.029). At τ = 0.8 the cross-day point
   estimates at 20 commands exceed twice chance: 0.069 (95% interval 0.043–0.100) for full
   features, 0.065 (0.036–0.099) for P2-RC and 0.075 (0.052–0.099) for P3 (strongest attacker
   per representation; *descriptive*, 3 seeds; the P2-RC interval reaches below 2/54).
5. **Pitfall 2: reference-session-only attackers.** A training-free defence from a recent arXiv
   preprint (Muhaimin & Nagarajan, 2026) reproduces on its own cross-session protocol
   (90.8% → 11.1% identification). One or five labelled deployment trials per user, attacking
   the rest of the same session, restore 81.75% or 97.11% (G, *exploratory*). On 54 users the
   reference-session attack reaches 1.34% (paper 2.4%), and five labels reach 92.7% within a day
   (C4, *confirmatory*, adapted pipeline).

**Position.** High identifiability of EEG is long known from EEG biometrics, and pitfall 1 is the
effect Elazar and Goldberg showed for text. What BEAT adds is a pre-registered audit of a task
BCI's transmitted outputs, including the decoded command stream, which a 2026 review (Sun et al.)
names as a re-identification pathway but which we found no study measuring. The two pitfalls are
practices we found in specific works: co-trained-only identifiers in ID-RemovalNet (published,
IJCAI-25), and reference-session-only attackers in Meng et al. (published, IEEE TNSRE 2023) and
in the Muhaimin & Nagarajan arXiv preprint (not peer-reviewed). We do not claim they are
typical of the field. Formal defences for discrete outputs — randomized response (Warner 1965),
equalized-odds post-processing (Hardt et al. 2016; Putzel & Lee 2022) and perfect-privacy
releases (Rassouli & Gündüz 2021) — are not evaluated in these results. See
[`docs/related_work.md`](docs/related_work.md) §6–7.

### Pre-registered decisions

| Claim | Data | Prediction | Observed | p | Verdict |
|---|---|---|---|---|---|
| H1 | 2a | Tangent space beats CSP with 1 calibration run | 0.689 vs 0.669 BA (+2.0 pp; 6 of 9 users) | 0.10 (exact sign-flip) | Not supported |
| H2 | 2a | Gradient-reversal bottleneck (P3) ≥ 5 pp below the task-only bottleneck (P2) in strongest identity BA, with a positive per-user lower bound and task BA non-inferior (−5 pp), in both the common and the recentred family | λ = 0.01 selected; identity 0.626 vs 0.619 (common), 0.427 vs 0.427 (recentred) | 0.87 / 0.44 (protection, sign-flip) | Not supported |
| H3 | 2a | Identity left in command streams (descriptive) | Within session E: 1 command 0.156–0.198; 20 commands 0.211–0.478 (chance 0.111) | — | Quantified, above chance |
| C1 | L | P3 (λ = 1, d = 8): strongest independent attacker exceeds the co-trained head by ≥ 20 pp | Within-day 0.514 vs 0.040 (47.4 pp); cross-day gap 17.6–19.9 pp | Holm 8.9 × 10⁻²⁶ | Supported |
| C2 | L | Recentred 8-d bottleneck (P2-RC) ≥ 20 pp below full features (P0), task BA non-inferior (−5 pp) | Within-day 0.994 → 0.340 (cross-day 0.520 → 0.054); task −0.62 pp (lower bound −2.27 pp) | Holm 4.6 × 10⁻⁵ | Supported |
| C3 | L | P2-RC commands at τ = 0.5: strongest BA at 20 commands exceeds 2/54 and exceeds that at 1 command | Within-day 0.085 vs 0.037 (cross-day 0.019–0.029 at 20) | Holm 0.21 | **Not supported** |
| C4 | L | Five deployment labels per user exceed 10/54 against the preprint's defence (benchmark gate first) | 92.7% within-day; benchmark gate passed (1.34% vs 2.4%) | Holm 4.5 × 10⁻³⁵ | Supported |

Sources: `results/exp_a/h1.json`, `results/exp_b/decision.json`,
`results/exp_c/observation_strongest.csv`, `results/exp_d/tradeoff.csv` (2a cross-day),
`results/exp_l/decision.json` and `results/exp_l/identity_summary.csv` (L cross-day: rows with
scenario = auxiliary). Sections 5 and 6.7 of the [report](docs/report.md) give the full tables.
`python -m beat.verify` re-checks these numbers against the files (registry
[`docs/numbers.yaml`](docs/numbers.yaml)). The low C3 power is our
reading of the design, not a tested result: at 20 commands each user contributes only five test
groups per direction.

| ![Pitfall 1: co-trained head vs independent attackers](results/figures/summary/fig1_pitfall1_head_vs_attacker.png) | ![Utility–identity frontier](results/figures/summary/fig2_utility_identity_frontier.png) |
|---|---|
| **Pitfall 1** on 2a: B (confirmatory), D and E (exploratory) | **Frontier** on 2a: task-only bottlenecks vs adversarial defences |
| ![Pitfall 2: enrolment curve](results/figures/summary/fig4_pitfall2_enrolment.png) | ![Lee2019 confirmatory claims](results/figures/summary/fig6_lee2019_confirmatory.png) |
| **Pitfall 2** on 2a: G (exploratory) | **Lee2019, 54 users:** claims C1–C4 (confirmatory) |

The same folder has three more figures: [command accumulation](results/figures/summary/fig3_command_accumulation.png),
[pre-registration timeline](results/figures/summary/fig5_preregistration_timeline.png) and
[identity vs cohort size](results/figures/summary/fig7_lee2019_users.png).

## Audit checklist (short form)

1. Retrain independent attackers on frozen outputs. Never cite a co-trained head as evidence;
   treat a below-chance score as a warning.
2. Attack with a few labelled deployment-period outputs, and report an enrolment curve as well as
   reference-session attacks.
3. For each number, state the scenario, the number of identities and the chance level. Compare
   cohorts relative to chance.
4. Before crediting an adversarial objective, compare it with a task-only bottleneck and with
   recentring, and measure utility on the same held-out data.
5. Report identity against observation length. Seal the analysis before scoring, and report
   failed claims, aborted runs and non-convergence.

The full list of ten items, with evidence for each, is in
[report §9](docs/report.md#9-audit-checklist-for-identity-protection-claims-in-bcis).

## How it was done

- **Data.**
  - BNCI2014-001 / BCI IV 2a: 9 users, sessions T and E on different days, 22 channels,
    left- vs right-hand imagery.
  - Lee2019/OpenBMI: 54 users, two sessions on different days, each with an offline and an
    online phase of 100 trials, reduced to 20 motor channels.
  - Preprocessing is a causal 8–30 Hz filter and the window 1–4 s after the cue. G follows
    the preprocessing of its source paper instead.
- **Pre-registration and seals.** The plan, configs, code and every training-data-only
  selection were committed before held-out data were scored:

  | Seal | Commit | Covers | Status |
  |---|---|---|---|
  | E seal | `6ab5668` | Plan v1: A, B, C (A/B final runs at `e6c13c8`) | Confirmatory |
  | Seal 2 | `b216b7d` | Plan v2: D, E | Exploratory (reuses session E) |
  | — | — | G at the preprint's published operating point | Exploratory |
  | Seal 3 | `9757c3f` | Plan v3: L, claims C1–C4; session 2 unopened | Confirmatory |

  Each fit step on the defended side raises `LeakageError` if it receives held-out trials, and
  the tests enforce this. [`seals/manifest.yaml`](seals/manifest.yaml) lists the sealed files of
  each seal and the commits allowed to change them later (the next seals and logged deviation 7);
  `python -m beat.verify --seals` checks the Git history and the working tree against it. The
  seals are self-attested: they fix the order of commits, not their time.
- **Independent attackers.** Logistic regression, RBF-SVM and MLP are retrained on the frozen
  outputs of every representation, scenario and seed. The reported score comes from the
  strongest of them. The main scenario trains the attacker on labelled outputs collected after
  deployment:
  - 2a: runs 1–3 ↔ 4–6 of session E.
  - L: offline ↔ online phase of session 2, recorded the same day.

  The auxiliary scenario is cross-day.
- **Failures are reported.** H1, H2 and C3 failed and are reported as results.
  - Experiment C's sealed rejection rule held 35% of session-E trials, so the primary C
    analysis replaced it; the sealed variant is still reported.
  - L's first run stopped at a seal-verifier check (CRLF line endings) before session 2 was
    opened.
  - L's attempt 1 then hit the sealed memory guard. Attempt 2, an unchanged re-execution,
    produced the results.
  - 286 of 586 MLP fits (45 of 85 confirmatory) did not converge within the fixed budget. Their
    predictions were kept, as the sealed rule requires.

  Details: [`docs/analysis_plan.md`](docs/analysis_plan.md) (plans v1–v3, decision rules) and
  [`docs/decision_log.md`](docs/decision_log.md) (seals, deviations, run history).

## Repository map

| Path | Contents |
|---|---|
| `beat/` | Pipeline: `data.py` (loading, leakage guard), `features.py`, `models.py`, `attacks.py`, `stats.py`, `splits.py`, `lee2019.py`, `exp_a.py` … `exp_g.py`, `exp_l.py`, `figures.py`, `run_all.py` |
| `configs/` | Data settings and one YAML per experiment, fixed at the seals |
| `scripts/` | `download_data.py` (BNCI2014-001), `prepare_lee2019.py` (Lee2019), `make_summary_figures.py`, `export_demo_data.py` |
| `tests/` | Leakage guards, unit tests and synthetic end-to-end runs (see [Tests and CI](#tests-and-ci)) |
| `results/` | Tables (CSV/JSON) and figures per experiment; `results/figures/summary/` holds the summary figures |
| `docs/` | [`report.md`](docs/report.md) (technical report), [`explainer_ko.md`](docs/explainer_ko.md) (Korean explainer), `analysis_plan.md`, `decision_log.md`, `related_work.md` |
| `demo/` | Static interactive results page |
| `presentation/` | Final talk slides in Korean with speaker notes (`BEAT_final_talk.pptx`, PDF copy `BEAT_final_talk.pdf`) |

## Reproduction

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # pinned versions of the sealed runs (CPU PyTorch index)
.venv\Scripts\python -m pip install -e . --no-deps        # installs `beat`, so scripts\*.py can import it
.venv\Scripts\python scripts\download_data.py    # BNCI2014-001, about 750 MB, SHA-256 printed
.venv\Scripts\python scripts\prepare_lee2019.py  # Lee2019: streams 108 files (~66 GB raw), keeps reduced epochs
.venv\Scripts\python scripts\verify_inputs.py    # data files against the committed checksums/
.venv\Scripts\python -m pytest
.venv\Scripts\python -m beat.verify              # seal manifest and registered numbers (no data needed)
.venv\Scripts\python -m beat.run_all             # A, B and C (both gate variants) and their figures
```

`scripts\prepare_lee2019.py` imports `beat`, which is importable only after `pip install -e .`
(the script itself is left unchanged because it produced L's inputs). The editable install also
provides the commands `beat-run-all` and `beat-verify`.

**Input checksums.** `checksums/bnci2014_001.sha256` holds the SHA-256 of the 18 BNCI2014-001
files (as printed by `download_data.py`), `checksums/lee2019_raw.sha256` the 108 raw Lee2019
files (recorded by `prepare_lee2019.py` before each raw file was deleted), and
`checksums/lee2019_processed.sha256` the 108 reduced `.npz` files. The last were hashed on
2026-11-11, after the L run, so they show that the files have not changed since then, not that they
are the files L read. `python scripts/verify_inputs.py --lee2019-raw` re-downloads each raw Lee2019
file as a stream (about 66 GB, nothing written to disk) and compares it with the committed hash.

`beat.run_all` covers A–C only. Run the other experiments separately:

| Exp. | Command | Measured time |
|---|---|---|
| A | `python -m beat.exp_a` | 2 min |
| B | `python -m beat.exp_b dev`, then `final` | 3 + 4 min |
| C | `python -m beat.exp_c dev`, then `final` (add `--variant sealed_gate` for the sealed rule) | 13 min per gate variant |
| D | `python -c "from beat import exp_d; exp_d.main(['final'])"` | 13.3 min |
| E | `python -c "from beat import exp_e; exp_e.main(['final'])"` | 43 min (2 workers) |
| G | `python -m beat.exp_g bnci` | not timed |
| L | `python -m beat.exp_l dev`, then `timing`, then `final --seal-commit 9757c3fceca585db3f56530d54091ab0fa28ce0b` | 34.3 min (6 workers requested, 5 scheduled) |
| Figures | `python scripts/make_summary_figures.py` | — |

- **Machine.** Times were measured on an 8-thread laptop CPU without a GPU. The whole chain has
  never been timed as one run.
- **D and E.** They are started through an import because `python -m beat.exp_d final` fails
  with a pickling error. The sealed code was left unchanged (decision log).
- **L.** `final` checks the sealed plan, config, selections and sources against the seal-3
  commit before opening session 2. It writes only to the sealed result paths, so re-running it
  overwrites `results/exp_l/`.
- **Lee2019 raw files.** `prepare_lee2019.py` deletes each raw file after reduction and records
  its SHA-256 in `data/processed/lee2019/SHA256SUMS`.

### Tests and CI

The tests need no EEG data. Apart from one loader check, every test generates synthetic signals
in pytest's temporary directory. `conftest.py` assigns two markers, because the sealed test files
cannot be edited:

- `data`: reads the raw BNCI2014-001 files under `data/` (`tests/test_foundation.py::test_loaded_trial_counts_and_shape`).
- `slow`: full synthetic development + final pipeline round trips (`test_synthetic_*final*`),
  about 25 s to 2 min each.

```powershell
.venv\Scripts\python -m pytest                              # everything; `data` tests skip without data/
.venv\Scripts\python -m pytest -m "not data"                # what CI runs
.venv\Scripts\python -m pytest -m "not data and not slow"   # quick check, about a minute
```

**CI** (`.github/workflows/ci.yml`) runs `pytest -m "not data"`, `python -m beat.verify` (seal
manifest on the full Git history, and the number registry) and `ruff` lint on every push and pull
request, on Windows (where the experiments ran) and Ubuntu, with Python 3.12 and the CPU-only
PyTorch from `requirements.txt`. Lint skips the sealed files, which cannot be edited; the
exclusion list is in `pyproject.toml`. The sealed `beat/exp_l.py` reads memory through the Windows API, so its
end-to-end test is skipped, with that reason, on Ubuntu and runs on Windows. It covers the
leakage guards, the feature, model, attack and statistics units, and the synthetic end-to-end
runs of each experiment. It never downloads EEG and uses no secrets, so it does not reproduce
the reported numbers. Those require the data scripts and experiment commands above.

## Demo

`demo/index.html` is a static page. It opens offline straight from the file system, with no
server or network access, and walks through the replay, both pitfalls, the frontier, the Lee2019
replication and the checklist. Its data (`demo/data/`) is generated by
`python scripts/export_demo_data.py` from `results/` and from experiment C's replay artifacts in
`artifacts/exp_c/`. That data contains decoded commands and aggregate results only, with no EEG
or features. `demo/data/` is committed, so the page works in a fresh clone; regenerating it
requires experiment C's artifacts (not tracked) and the export script.

## Licence and data

- **BNCI2014-001** is licensed CC BY-ND 4.0.
- **Lee2019/OpenBMI** is downloaded from its GigaDB release (DOI 10.5524/100542), published
  under CC0 1.0.
- **No raw data is redistributed.** No raw EEG, reduced epochs or per-user features from either
  dataset are in this repository; the root `data/` folder and `artifacts/` are excluded from Git.

## References

These are the main works; the full, verified list is in the
[report's references](docs/report.md#references).

- M. Tangermann et al. Review of the BCI Competition IV. *Front. Neurosci.*, 6:55, 2012. doi:10.3389/fnins.2012.00055 (data set 2a = BNCI2014-001).
- M.-H. Lee et al. EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy. *GigaScience*, 8(5):giz002, 2019. doi:10.1093/gigascience/giz002
- A. Barachant, S. Bonnet, M. Congedo, C. Jutten. Multiclass brain–computer interface classification by Riemannian geometry. *IEEE Trans. Biomed. Eng.*, 59(4):920–928, 2012. doi:10.1109/TBME.2011.2172210
- Y. Ganin, V. Lempitsky. Unsupervised domain adaptation by backpropagation. *Proc. ICML*, PMLR 37:1180–1189, 2015.
- Y. Elazar, Y. Goldberg. Adversarial removal of demographic attributes from text data. *Proc. EMNLP*, pp. 11–21, 2018. doi:10.18653/v1/D18-1002
- L. Meng, X. Jiang, J. Huang, W. Li, H. Luo, D. Wu. User identity protection in EEG-based brain–computer interfaces. *IEEE Trans. Neural Syst. Rehabil. Eng.*, 31:3576–3586, 2023. doi:10.1109/TNSRE.2023.3310883
- H. Wang et al. ID-RemovalNet: identity removal network for EEG privacy protection with enhancing decoding tasks. *Proc. IJCAI-25*, pp. 4209–4217, 2025. doi:10.24963/ijcai.2025/469
- S. S. Muhaimin, A. Nagarajan. Riemannian suppression of cross-session identity leakage in motor-imagery EEG. arXiv:2610.15626, October 2026 (preprint, not peer-reviewed; the defence tested in G and C4).
- L. Sun et al. Revisiting privacy preservation in brain-computer interfaces. arXiv:2605.11386, 2026 (preprint).
- M. Hardt, E. Price, N. Srebro. Equality of opportunity in supervised learning. *NIPS 2016*.
- P. Putzel, S. Lee. Blackbox post-processing for multiclass fairness. SafeAI 2022 workshop at AAAI-22, CEUR-WS vol. 3087.
- B. Rassouli, D. Gündüz. On perfect privacy. *IEEE J. Sel. Areas Inf. Theory*, 2(1):177–191, 2021. doi:10.1109/JSAIT.2021.3053432
- S. L. Warner. Randomized response: a survey technique for eliminating evasive answer bias. *J. Amer. Statist. Assoc.*, 60(309):63–69, 1965. doi:10.1080/01621459.1965.10480775

Proposal: BEAT team (이동현, 장규범), 2026-10-09. Original proposal: `BEAT 프로젝트 기획안.pdf`.
