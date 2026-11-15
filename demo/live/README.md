# BEAT Audit Live

A local, offline page on which a presenter drops a CSV of BCI outputs and the audience watches
`beat_audit` measure, on the spot, whether those outputs let an attacker link two recordings of
the same person. Reading rules: `results/audit/README.md`. The numbers shown are a demo subset
("시연용 부분집합, 탐색"), not the project's registered results.

## Start

```
.venv\Scripts\python demo/live/server.py            # http://127.0.0.1:8737/
.venv\Scripts\python demo/live/server.py --port 8740 # if 8737 is taken
```

Standard library plus the installed `beat_audit`; listens on 127.0.0.1 only; no network, CDN or
web fonts. Opening `static/index.html` without the server shows how to start it.

## Presenter steps

1. Start the server, open the address in a browser, press F11.
2. Optionally type the seconds per output in "출력 1개의 시간"; the curve then shows time too.
3. Drop `A_…features.csv`, then `B_…commands.csv`, then `C_…shuffled_users.csv` (drag anywhere,
   or Tab to "CSV 파일 선택" and press Enter). Each card stays; up to three side by side.
4. If a run fails or is slow, press "저장된 실행 보기" on the card: a stored run of the same file,
   labelled with its timestamp.
5. "인쇄용 페이지" on a card opens a one-page print view (card, settings, SHA-256, limits);
   print with Ctrl+P.

## Samples

`scripts/make_live_samples.py` builds both sets by subsetting users, days and rows only; values
are copied unchanged.

Subset rule, fixed in advance and independent of any result: the **20 lowest user identifiers**
(users 1-20), deployment **days 2 and 3**, the **first 50 trials (by order) of each segment**.
A, B and C contain the same users, days and trials; C is A with its user labels permuted within
each (day, segment, order) position by the fixed seed 20261116. 20 is the smallest cohort for
which the tool does not warn "Below 20 users: estimates are unstable"; more users would push a
feature run past about 15 s on the demo machine (20 users take about 14 s). The rule is shown in
the small print of every card and on the print page.

- `samples/yang2025/` (ignored, built from `artifacts/audit/inputs/yang2025/`): 4,000 rows each;
  A 190-dim features (P0), B decoded commands (P2-RC_d8, tau 0.60) of the same trials, C = A
  with permuted user labels. Rebuild: `.venv\Scripts\python scripts/make_live_samples.py`.
- `samples/synthetic/` (tracked): generated data of the same sizes (20 users, 2 days, 50 trials
  per segment, 4,000 rows); A 16-dim features, B commands decoded from A, C = A permuted.

Demo settings (shown on every card as "시연 설정"): 2,000 permutations (standard 10,000),
1,000 bootstrap resamples (standard 2,000), k = 1, 5, 10, 25 (every k up to half the trials of a
segment), attacker cosine for features (the tool's standard is five comparators; cosine_rich alone
would roughly triple a 20-user feature run) or bayes_factor for commands, which is the tool's
standard for commands, so nothing was reduced there. The tool's "Nonstandard inference budget"
warning is kept. Feature cards add "더 강한 공격은 더 높게 잴 수 있음".

The feature comparator reduction is not a pre-specified setting. It was decided for run time
during timing, after the values of file A under both comparators had been seen (k = 25,
other day: cosine 0.786, cosine_rich 0.818). Only the user, day and row rule above was fixed
before any outcome. Feature cards and their print pages say the same. Each card is read on its own; overlapping intervals between cards are not a difference.

## Stored runs and screenshots

`stored/<sha256>.json` are written by the same worker as live runs:
`.venv\Scripts\python demo/live/server.py --store demo/live/samples/*/*.csv`.
`screenshots/` holds 1920x1080 captures: three cards, one card in progress, the print page, card A alone.

Tests: `.venv\Scripts\python -m pytest tests/test_live_demo.py`.
