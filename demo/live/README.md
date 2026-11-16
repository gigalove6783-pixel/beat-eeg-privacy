# BEAT Audit Live

A local, offline page on which a presenter drops a CSV of BCI outputs and the audience watches
`beat_audit` measure, on the spot, whether those outputs let an attacker link two recordings of
the same person. Reading rules: `results/audit/README.md`. The numbers shown are a demo subset
("시연용 부분집합, 탐색"), not the project's registered results.

## Stage procedure

1. Before the talk, double-click `demo/live/start_demo.bat`; wait for the full-screen page and
   check that the top line reads "준비 완료 · 실제 코호트 표본 3개". For about two seconds after
   the page appears it reads "입력 대기 중 …" and nothing starts a run.
2. On stage, press Page Down (clicker) or the right arrow three times, once each time the previous
   card has finished: A 특징 벡터, B 명령, C 대조군. Space and Enter do not advance the sequence;
   they only press a button that has keyboard focus, and no button has focus when the page opens.
   If a run starts by accident, press "취소" on its card (Tab reaches it); the next Page Down runs
   the same file again.
3. If a run fails or stalls, press "저장된 실행 보기" on that card.

Leave the console window open during the talk; closing it stops the server. Alt+F4 closes the
page window.

## Start

`start_demo.bat` (ASCII, CRLF) runs `launch.py` with `.venv\Scripts\python.exe`. The launcher
reports a missing `.venv`, missing samples (and falls back to the tracked synthetic set when the
real-cohort samples are incomplete) or a busy port (it takes the next free port from 8737),
starts the server in its own process, waits until the page and the audit tool answer and opens
Chrome as a full-screen app window with a profile of its own
(`%LOCALAPPDATA%\BEATAuditLive\chrome-profile`), or the default browser if Chrome is absent
(then press F11). A second double-click while it runs opens the running page and starts no second
server. Closing the console window stops the server and its audit workers.

By hand:

```
.venv\Scripts\python demo/live/server.py            # http://127.0.0.1:8737/
.venv\Scripts\python demo/live/server.py --port 8740 # if 8737 is taken
```

Standard library plus the installed `beat_audit`; listens on 127.0.0.1 only; no network, CDN or
web fonts. Opening `static/index.html` without the server shows how to start it.

## Page controls

- The readiness line at the top names the active set (real-cohort samples when all three files
  are present locally, the tracked synthetic set otherwise) or the missing file, and says that the
  buttons run prepared sample files.
- "다음 실행: …" runs the next file of the sequence A, B, C and after C offers "처음부터 다시"
  (which clears the cards). It is disabled while a run is in progress and for about two seconds
  after the page loads. Page Down and the right arrow do the same from anywhere on the page except a
  text field; auto-repeated presses and a key already held while the page loaded are ignored.
  Cancelling a run started this way returns the sequence to that file. "A 특징 벡터", "B 명령",
  "C 대조군" rerun one file.
- A prepared file is named to the server by its role only (`POST /api/samples/A|B|C`); the server
  reads that file of the active set from disk in that request and runs it through the same
  validation and audit path as an upload. No path or body from the client is used; nothing is
  precomputed or cached. Any other identifier is refused.
- Optionally type the seconds per output in "출력 1개의 시간"; the curve then shows time too.
- Dropping a CSV anywhere, or "CSV 파일 선택", still runs one's own file as before.
- "저장된 실행 보기" on a card shows a stored run of the same file, labelled with its timestamp.
- "인쇄용 페이지" on a card opens a one-page print view (card, settings, SHA-256, limits);
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
