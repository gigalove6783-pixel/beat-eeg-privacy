/* BEAT Audit Live — page logic. Every number shown comes from the local server's run of
   beat_audit (or a stored run of the same code path, labelled as such). */
"use strict";

const MAX_CARDS = 3;
const SVG = "http://www.w3.org/2000/svg";
const REL = { "same-day": "같은 날", "cross-day": "다른 날" };
const HEADLINE_RELATIONS = ["cross-day", "same-day"];
const STANDARD_TEXT = "표준";

const state = { cards: [], maxBytes: 32 * 1024 * 1024, seq: 0, printReturn: null };
const $ = (id) => document.getElementById(id);

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "style") Object.assign(node.style, value); // CSSOM: allowed by the CSP
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const present = (list) => list.filter((x) => x !== null && x !== undefined && x !== false);

function s(tag, attrs, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined) continue;
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) if (child) node.append(child);
  return node;
}

const f2 = (v) => (Math.round(v * 100) / 100).toFixed(2);
const interval = (iv) => `${f2(iv[0])}–${f2(iv[1])}`;
const num = (v) => Number(v).toLocaleString("ko-KR");

function duration(seconds) {
  if (seconds < 60) return `${num(Math.round(seconds * 10) / 10)}초`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return rest ? `${minutes}분 ${rest}초` : `${minutes}분`;
}

function clock(iso) {
  const d = new Date(iso);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function detected(wording) { return typeof wording === "string" && wording.startsWith("linkage detected"); }

function verdictText(wording) {
  if (detected(wording)) return "연결 검출됨";
  if (typeof wording === "string" && wording.startsWith("not detected")) return "검출되지 않음 — 증거 부족이지 없음이 아님";
  return "판정 없음";
}

function kindText(input) {
  return input.kind === "features" ? `특징 벡터 ${input.dimension}차원` : `명령 ${input.alphabet}종`;
}

const kindWord = (kind) => (kind === "features" ? "특징 벡터" : "명령");

function whatText(input) {
  const size = input.kind === "features" ? `${input.dimension}차원` : `${input.alphabet}종`;
  return [size, `사용자 ${input.users}명`, `기록일 ${input.days.length}일`, `${num(input.rows)}행`,
    input.cues ? "과제 단서 포함" : "과제 단서 없음"].join(" · ");
}

function kLabel(k, period) {
  return period ? `출력 ${k}개 (${duration(k * period)})` : `출력 ${k}개`;
}

/* ---------- reading of a finished summary ---------- */

function okCells(summary, relation) {
  const condition = summary.conditions.find((c) => c.relation === relation);
  return condition ? condition.cells.filter((c) => c.status === "ok") : [];
}

function headline(summary) {
  for (const relation of HEADLINE_RELATIONS) {
    const cells = okCells(summary, relation);
    if (cells.length) return { relation, cell: cells[cells.length - 1], cells };
  }
  return null;
}

function sentence(summary, head) {
  const period = summary.settings.seconds_per_output;
  const rel = REL[head.relation];
  const c = head.cell;
  const span = period ? `(${duration(c.k * period)} 분량)` : "";
  if (detected(c.wording)) {
    const first = head.cells.find((x) => detected(x.wording));
    const from = first && first.k !== c.k ? ` 연결은 출력 ${first.k}개부터 검출됩니다.` : "";
    return `${rel}에 기록한 출력 ${c.k}개${span}씩을 비교하면, 공격자는 같은 사람의 묶음을 AUC ${f2(c.auc)}로 가려냅니다(0.5가 우연).${from}`;
  }
  return `이 파일에서는 ${rel} 기록끼리 같은 사람으로 연결된다는 증거가 나오지 않았습니다(출력 ${c.k}개, AUC ${f2(c.auc)}, 0.5가 우연). 증거 부족이지 없음이 아닙니다.`;
}

/* ---------- graphics ---------- */

function scaleBar(cell) {
  const W = 520, H = 64, L = 14, R = 14, y = 26;
  const x = (v) => L + Math.max(0, Math.min(1, v)) * (W - L - R);
  const ok = detected(cell.wording);
  return s("svg", { viewBox: `0 0 ${W} ${H}`, class: "scale", role: "img",
    "aria-label": `AUC ${f2(cell.auc)}, 95% 구간 ${interval(cell.interval)}, 0.5는 우연` },
    s("line", { x1: x(0), x2: x(1), y1: y, y2: y, class: "scale-axis" }),
    [0, 0.25, 0.75, 1].map((v) => s("line", { x1: x(v), x2: x(v), y1: y - 6, y2: y + 6, class: "scale-tick" })),
    s("line", { x1: x(0.5), x2: x(0.5), y1: y - 18, y2: y + 18, class: "scale-chance" }),
    s("rect", { x: x(cell.interval[0]), y: y - 9, width: Math.max(3, x(cell.interval[1]) - x(cell.interval[0])), height: 18, rx: 4, class: ok ? "scale-band" : "scale-band muted" }),
    s("circle", { cx: x(cell.auc), cy: y, r: 9, class: ok ? "scale-dot" : "scale-dot muted" }),
    s("text", { x: x(0), y: H - 4, class: "scale-label", "text-anchor": "start", text: "0" }),
    s("text", { x: x(0.5), y: H - 4, class: "scale-label strong", "text-anchor": "middle", text: "0.5 우연" }),
    s("text", { x: x(1), y: H - 4, class: "scale-label", "text-anchor": "end", text: "1" }));
}

/* series: [{relation, points: [{k, auc, interval, wording?}]}] */
function curve(series, kGrid, period, opts = {}) {
  const W = 560, H = period ? 190 : 170, L = 50, R = 108, T = 12, B = period ? 52 : 32;
  const lo = 0.3, hi = 1.0;
  const n = kGrid.length;
  const x = (k) => L + (n === 1 ? 0.5 : kGrid.indexOf(k) / (n - 1)) * (W - L - R);
  const y = (v) => T + (hi - Math.max(lo, Math.min(hi, v))) / (hi - lo) * (H - T - B);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "curve", role: "img",
    "aria-label": series.map((sr) => `${REL[sr.relation]}: ` + sr.points.map((p) => `k=${p.k} AUC ${f2(p.auc)}`).join(", ")).join("; ") || "아직 값 없음" });
  for (const v of [0.5, 0.75, 1.0]) {
    svg.append(s("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v === 0.5 ? "grid chance" : "grid" }));
    svg.append(s("text", { x: L - 10, y: y(v) + 5, class: v === 0.5 ? "tick strong" : "tick", "text-anchor": "end", text: v === 0.5 ? "우연" : v.toFixed(v === 0.75 ? 2 : 1) }));
  }
  kGrid.forEach((k) => {
    svg.append(s("text", { x: x(k), y: H - B + 22, class: "tick", "text-anchor": "middle", text: `${k}` }));
    if (period) svg.append(s("text", { x: x(k), y: H - B + 42, class: "tick sub", "text-anchor": "middle", text: duration(k * period) }));
  });
  svg.append(s("text", { x: W - R + 20, y: H - B + 22, class: "tick sub", text: "← 출력 수 k" }));
  const ends = [];
  series.forEach((sr) => {
    const pts = sr.points.filter((p) => kGrid.includes(p.k)).sort((a, b) => a.k - b.k);
    if (!pts.length) return;
    const cls = sr.relation === "cross-day" ? "s-cross" : "s-same";
    const g = s("g", { class: cls });
    for (const p of pts) g.append(s("line", { x1: x(p.k), x2: x(p.k), y1: y(p.interval[0]), y2: y(p.interval[1]), class: "whisker" }));
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.k).toFixed(1)},${y(p.auc).toFixed(1)}`).join("");
      g.append(s("path", { d, class: "line" }));
      if (opts.growKey && opts.growKey === sr.relation) {
        const a = pts[pts.length - 2], b = pts[pts.length - 1];
        g.append(s("path", { d: `M${x(a.k)},${y(a.auc)}L${x(b.k)},${y(b.auc)}`, class: "line grow" }));
      }
    }
    for (const p of pts) {
      if (sr.relation === "cross-day") g.append(s("circle", { cx: x(p.k), cy: y(p.auc), r: 6, class: "mark" }));
      else g.append(s("rect", { x: x(p.k) - 5.5, y: y(p.auc) - 5.5, width: 11, height: 11, class: "mark" }));
    }
    const last = pts[pts.length - 1];
    ends.push({ g, x: x(last.k), y: y(last.auc), text: `${REL[sr.relation]} ${f2(last.auc)}` });
    svg.append(g);
  });
  if (ends.length === 2 && Math.abs(ends[0].y - ends[1].y) < 22) {
    const [a, b] = ends[0].y <= ends[1].y ? ends : [ends[1], ends[0]];
    const mid = Math.max(T + 22, (a.y + b.y) / 2);
    a.y = mid - 11; b.y = mid + 11;
  }
  for (const e of ends) e.g.append(s("text", { x: e.x + 12, y: e.y + 6, class: "end-label", text: e.text }));
  return svg;
}

/* ---------- cards ---------- */

function newCard(filename) {
  const tpl = $("tpl-card").content.firstElementChild.cloneNode(true);
  const card = { id: `c${++state.seq}`, n: state.seq, el: tpl, filename, points: [], next: 0, status: "uploading" };
  tpl.id = card.id;
  tpl.setAttribute("aria-labelledby", `${card.id}-title`);
  tpl.querySelector(".card-title").id = `${card.id}-title`;
  if (state.cards.length >= MAX_CARDS) {
    const oldest = state.cards[0];
    removeCard(oldest, true);
    announce(`세 장이 차서 가장 오래된 카드(${oldest.filename})를 내렸습니다.`);
  }
  state.cards.push(card);
  $("cards").append(tpl);
  $("clear").hidden = false;
  layout();
  return card;
}

function layout() {
  $("cards").dataset.count = String(state.cards.length);
  $("app").classList.toggle("empty", state.cards.length === 0);
}

function removeCard(card, silent) {
  stopPolling(card);
  if (card.status === "running" && card.runId) fetch(`/api/runs/${card.runId}/cancel`, { method: "POST" }).catch(() => {});
  card.el.remove();
  state.cards = state.cards.filter((c) => c !== card);
  $("clear").hidden = state.cards.length === 0;
  layout();
  if (!silent) { announce(`${card.filename} 카드를 닫았습니다.`); $("pick").focus(); }
}

function cardTitle(card, kind) {
  if (card.sample) return `${card.sample.role} · ${card.sample.title}`;
  return `파일 ${card.n}${kind ? ` · ${kindWord(kind)}` : ""}`;
}

function setHead(card, input, kind) {
  const el = card.el;
  el.querySelector(".card-title").textContent = cardTitle(card, input ? input.kind : kind);
  el.querySelector(".file").textContent = card.sample ? card.sample.description : card.filename;
  el.querySelector(".file").title = card.filename;
  el.querySelector(".what").textContent = input ? whatText(input) : "";
}

/* Progress updates call this several times a second: the buttons are rebuilt only when the set of
   labels changes, and keyboard focus inside the row moves to the button with the same label (or the
   first one) so a presenter who tabbed onto a button keeps it. */
function actions(card, list) {
  const row = card.el.querySelector(".card-head .actions");
  const key = list.map(([label]) => label).join("|");
  if (card.actionKey === key) return;
  card.actionKey = key;
  const focused = row.contains(document.activeElement) ? document.activeElement.textContent : null;
  const buttons = list.map(([label, handler]) =>
    h("button", { type: "button", class: "quiet", onclick: handler, "aria-label": `${label}: ${card.filename}` }, label));
  row.replaceChildren(...buttons);
  if (focused !== null && buttons.length) (buttons.find((b) => b.textContent === focused) || buttons[0]).focus();
}

function storedButton(card) {
  return card.storedAvailable ? [["저장된 실행 보기", () => showStored(card)]] : [];
}

function renderProgress(card) {
  const body = card.el.querySelector(".card-body");
  const plan = card.plan;
  const elapsed = card.elapsed || 0;
  body.replaceChildren();
  card.el.classList.add("running");
  card.el.setAttribute("aria-busy", "true");
  const done = plan ? card.step?.done ?? 0 : 0;
  const total = plan ? plan.total : 0;
  const pct = total ? Math.round(done / total * 100) : 0;
  const step = card.step && card.step.k ? ` · 지금: ${REL[card.step.relation]}, 출력 ${card.step.k}개` : "";
  body.append(
    h("div", { class: "run-line" },
      h("span", { class: "run-label", text: plan ? "계산 중" : (card.status === "uploading" ? "파일 보내는 중" : "파일 확인 중") }),
      h("span", { class: "run-time", text: duration(elapsed) })),
    h("div", { class: "bar-track", role: "progressbar", "aria-label": "감사 진행", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(pct) },
      h("div", { class: "bar-fill", style: { width: `${pct}%` }})),
    h("p", { class: "run-step", text: plan ? `단계 ${done} / ${total}${step}` : "도구가 열 구성과 값을 검사합니다" }));
  if (plan) {
    const series = ["cross-day", "same-day"].filter((r) => plan.relations.includes(r))
      .map((relation) => ({ relation, points: card.points.filter((p) => p.relation === relation) }));
    body.append(h("div", { class: "curve-box" }, curve(series, plan.k_grid, card.period, { growKey: card.growKey })),
      h("p", { class: "note", text: "진행 중 값: 다중비교 보정 전이며 최종 판정이 아닙니다." }));
  }
  actions(card, [...storedButton(card), ["취소", () => cancel(card)]]);
}

const reduced = (st) => st.comparators.join(",") !== st.standard.comparators.join(",");

function comparatorText(st) {
  return reduced(st)
    ? `${st.comparators.join(", ")}(${STANDARD_TEXT} ${st.standard.comparators.length}종: ${st.standard.comparators.join(", ")} — 더 강한 공격은 더 높게 잴 수 있음)`
    : `${st.comparators.join(", ")}(${STANDARD_TEXT}과 같음)`;
}

function settingsText(st) {
  return `시연 설정: 순열 ${num(st.permutations)}회(${STANDARD_TEXT} ${num(st.standard.permutations)}) · 부트스트랩 ${num(st.bootstraps)}회(${STANDARD_TEXT} ${num(st.standard.bootstraps)}) · 공격 ${comparatorText(st)} · k ${st.k_grid.join(", ")}`;
}

/* Shown only where the comparator set was actually reduced (feature files). */
const REDUCTION_NOTE = "공격 축소는 미리 정한 설정이 아닙니다. 실행 시간을 재던 중 실제 코호트 파일 A의 두 공격 값(cosine 0.786, cosine_rich 0.818)을 본 뒤, 실행 시간 때문에 정했습니다. 결과를 보기 전에 고정한 것은 사용자·기록일·시행 규칙뿐입니다.";

/* The tool's own warnings, verbatim, each with a Korean reading where one is known. */
const WARNING_KO = [
  [/^Nonstandard inference budget/, "추론 예산 축소"],
  [/^Below (\d+) users/, (m) => `사용자 ${m[1]}명 미만: 추정 불안정`],
];

function warningItems(summary) {
  return summary.warnings.map((w) => {
    const hit = WARNING_KO.map(([re, ko]) => [w.match(re), ko]).find(([m]) => m);
    const ko = hit ? (typeof hit[1] === "function" ? hit[1](hit[0]) : hit[1]) : null;
    return [ko ? `${ko} ` : "", h("q", { text: w })];
  });
}

function resultBody(card, { forPrint } = {}) {
  const summary = card.summary;
  const head = headline(summary);
  const frag = [];
  if (!head) {
    frag.push(h("p", { class: "error-text", text: "평가할 수 있는 셀이 없습니다." }));
    return frag;
  }
  const c = head.cell;
  const ok = detected(c.wording);
  const period = summary.settings.seconds_per_output;
  frag.push(
    h("div", { class: "headline" },
      h("p", { class: "h-auc", text: f2(c.auc) }),
      h("div", { class: "h-side" },
        h("p", { class: "h-label", text: `AUC · ${REL[head.relation]} · ${kLabel(c.k, period)}` }),
        h("p", { class: "h-iv", text: `95% 구간 ${interval(c.interval)}` }),
        c.unreliable ? h("p", { class: "h-iv", text: "구간 불안정: 폴드가 작음 (unreliable: small folds)" }) : null)),
    scaleBar(c),
    h("div", { class: ok ? "verdict yes" : "verdict no" },
      ok || !c.wording.startsWith("not detected")
        ? h("p", { class: "v-ko", text: verdictText(c.wording) })
        : h("p", { class: "v-ko" }, "검출되지 않음 ", h("span", { class: "v-qual", text: "— 증거 부족이지 없음이 아님" })),
      h("p", { class: "v-tool" }, "도구 판정: ", h("q", { text: c.wording }))),
    h("p", { class: "plain", text: sentence(summary, head) }));
  const series = ["cross-day", "same-day"].map((relation) => ({ relation, points: okCells(summary, relation) }))
    .filter((sr) => sr.points.length);
  frag.push(h("div", { class: "curve-box" }, curve(series, summary.settings.k_grid, period)));
  const pairs = ["same-day", "cross-day"].map((r) => [r, okCells(summary, r).find((x) => x.k === c.k)]).filter(([, x]) => x);
  if (pairs.length === 2) {
    frag.push(h("dl", { class: "compare", "aria-label": `같은 날과 다른 날, 출력 ${c.k}개` },
      pairs.map(([r, x]) => h("div", {},
        h("dt", { text: `${REL[r]} · 출력 ${x.k}개` }),
        h("dd", {}, h("b", { text: f2(x.auc) }), ` ${interval(x.interval)}`)))));
  }
  if (c.excluded_users.length) frag.push(h("p", { class: "note", text: `분석 사용자 ${c.users_analysed}명 (제외 ${c.excluded_users.length}명)` }));
  if (forPrint) return frag;
  return frag;
}

function metaFoot(card) {
  const summary = card.summary;
  const st = summary.settings;
  const warnings = warningItems(summary);
  const run = card.stored
    ? `저장된 실행 · ${clock(card.stored.stored_utc)} 기록 · 계산 ${duration(summary.elapsed_seconds)}`
    : `방금 계산 · ${duration(summary.elapsed_seconds)} · ${clock(summary.finished_utc)}`;
  return h("div", { class: "meta" },
    h("p", { class: "tier", text: "시연용 부분집합, 탐색 — 프로젝트의 등록된 수치가 아닙니다." }),
    h("p", { text: settingsText(st) }),
    reduced(st) ? h("p", { text: REDUCTION_NOTE }) : null,
    card.sample?.selection ? h("p", { text: card.sample.selection }) : null,
    warnings.length ? h("p", {}, "도구 경고: ", warnings.map((w, i) => [i ? " · " : "", w])) : null,
    h("p", { class: "run", text: `${run} · ${summary.filename}` }));
}

function renderResult(card) {
  const el = card.el;
  el.classList.remove("running");
  el.removeAttribute("aria-busy");
  el.classList.toggle("stored", Boolean(card.stored));
  const banner = el.querySelector(".stored-banner");
  banner.hidden = !card.stored;
  if (card.stored) banner.textContent = `저장된 실행 · ${clock(card.stored.stored_utc)} 기록 · 지금 계산한 것 아님`;
  setHead(card, card.summary.input);
  el.querySelector(".card-body").replaceChildren(...resultBody(card));
  const foot = el.querySelector(".card-foot");
  foot.replaceChildren(metaFoot(card));
  actions(card, [["인쇄용 페이지", () => openPrint(card)], ["닫기", () => removeCard(card)]]);
}

function renderError(card, message, detail) {
  const el = card.el;
  el.classList.remove("running");
  el.removeAttribute("aria-busy");
  el.classList.add("failed");
  if (!el.querySelector(".card-title").textContent) setHead(card, null);
  el.querySelector(".card-body").replaceChildren(...present([
    h("p", { class: "error-title", text: "이 파일은 감사하지 못했습니다" }),
    h("p", { class: "error-text", text: message }),
    detail ? h("p", { class: "detail" }, "도구 메시지: ", h("code", { text: detail })) : null,
    h("p", { class: "note", text: "다른 파일을 바로 놓을 수 있습니다." })]));
  el.querySelector(".card-foot").replaceChildren();
  actions(card, [...storedButton(card), ["닫기", () => removeCard(card)]]);
  announce(`${card.filename}: ${message}`);
}

/* ---------- running ---------- */

function running() { return state.cards.find((c) => c.status === "running" || c.status === "uploading"); }

async function handleFile(file) {
  if (!file) return;
  const busy = running();
  if (busy) { announce(`앞의 파일(${busy.filename})을 분석하는 중입니다. 끝난 뒤 다시 놓아 주세요.`); return; }
  const card = newCard(file.name);
  setHead(card, null);
  const periodValue = $("period").value.trim();
  card.period = periodValue && Number(periodValue) > 0 ? Number(periodValue) : null;
  if (!/\.csv$/i.test(file.name)) { card.status = "error"; return renderError(card, "CSV 파일(.csv)만 받습니다."); }
  if (file.size > state.maxBytes) { card.status = "error"; return renderError(card, `파일이 너무 큽니다(${(file.size / 1048576).toFixed(1)} MB). 최대 ${Math.round(state.maxBytes / 1048576)} MB까지 받습니다.`); }
  if (periodValue && !(Number(periodValue) > 0)) { card.status = "error"; return renderError(card, "출력 1개의 시간(초)은 0보다 큰 숫자로 적어 주세요."); }
  card.t0 = performance.now();
  renderProgress(card);
  card.ticker = setInterval(() => {
    card.elapsed = (performance.now() - card.t0) / 1000;
    const t = card.el.querySelector(".run-time");
    if (t) t.textContent = duration(card.elapsed);
  }, 200);
  announce(`${file.name} 감사를 시작합니다.`);
  let response, body;
  try {
    response = await fetch("/api/runs", { method: "POST", body: file, headers: {
      "Content-Type": "text/csv", "X-File-Name": encodeURIComponent(file.name),
      ...(card.period ? { "X-Seconds-Per-Output": String(card.period) } : {}) } });
    body = await response.json();
  } catch (err) {
    card.status = "error";
    stopPolling(card);
    return renderError(card, "서버에 연결하지 못했습니다. 서버가 켜져 있는지 확인해 주세요.");
  }
  if (!response.ok) {
    card.status = "error";
    stopPolling(card);
    return renderError(card, body.error?.message || "서버가 이 파일을 받지 않았습니다.");
  }
  Object.assign(card, { runId: body.id, sha256: body.sha256, sample: body.sample, storedAvailable: body.stored, status: "running" });
  setHead(card, null, body.precheck.kind);
  card.el.querySelector(".what").textContent = `사용자 ${body.precheck.users}명 · 기록일 ${body.precheck.days}일 · ${num(body.precheck.rows)}행`;
  renderProgress(card);
  poll(card);
}

function stopPolling(card) {
  clearTimeout(card.timer);
  clearInterval(card.ticker);
}

async function poll(card) {
  if (card.status !== "running") return;
  let data;
  try {
    const r = await fetch(`/api/runs/${card.runId}?since=${card.next}`);
    data = await r.json();
  } catch (err) {
    card.status = "error";
    stopPolling(card);
    return renderError(card, "서버와 연결이 끊겼습니다.");
  }
  if (card.status !== "running") return;
  let changed = false;
  for (const e of data.events) {
    if (e.phase === "plan") { card.plan = e; changed = true; }
    else if (e.phase === "step") { card.step = e; changed = true; }
    else if (e.phase === "point" && isHeadlineComparator(card.plan, e)) {
      card.points = card.points.filter((p) => !(p.k === e.k && p.relation === e.relation)).concat([e]);
      card.growKey = e.relation;
      changed = true;
    }
  }
  card.next = data.next;
  if (data.status === "done") {
    card.status = "done";
    stopPolling(card);
    card.summary = data.summary;
    renderResult(card);
    const head = headline(card.summary);
    announce(`${card.filename}: ${head ? `${verdictText(head.cell.wording)}, AUC ${f2(head.cell.auc)}` : "완료"}`);
    return;
  }
  if (data.status === "error") {
    card.status = "error";
    stopPolling(card);
    return renderError(card, data.error.message, data.error.detail);
  }
  if (data.status === "cancelled") return;
  if (changed) {
    if (card.plan) {
      card.el.querySelector(".what").textContent =
        `사용자 ${card.plan.users}명 · 기록일 ${card.plan.days.length}일 · ${num(card.plan.rows)}행`;
    }
    renderProgress(card);
  }
  card.timer = setTimeout(() => poll(card), 250);
}

function isHeadlineComparator(plan, e) {
  if (!plan) return false;
  if (plan.kind === "commands") return e.comparator === "bayes_factor";
  return e.comparator === "cosine";
}

async function cancel(card) {
  if (card.runId) await fetch(`/api/runs/${card.runId}/cancel`, { method: "POST" }).catch(() => {});
  card.status = "cancelled";
  stopPolling(card);
  renderError(card, "실행을 취소했습니다.");
}

async function showStored(card) {
  if (card.status === "running" && card.runId) {
    await fetch(`/api/runs/${card.runId}/cancel`, { method: "POST" }).catch(() => {});
  }
  stopPolling(card);
  let record;
  try {
    const r = await fetch(`/api/stored/${card.sha256}`);
    record = await r.json();
    if (!r.ok) throw new Error(record.error?.message);
  } catch (err) {
    card.status = "error";
    return renderError(card, "저장된 실행을 불러오지 못했습니다.");
  }
  card.status = "done";
  card.el.classList.remove("failed");
  card.stored = record;
  card.summary = record.summary;
  renderResult(card);
  announce(`${card.filename}: 저장된 실행(${clock(record.stored_utc)} 기록)을 보여 줍니다.`);
}

/* ---------- print ---------- */

function openPrint(card) {
  const summary = card.summary;
  const st = summary.settings;
  const input = summary.input;
  const sheet = $("print-sheet");
  const rows = summary.conditions.flatMap((cond) => cond.cells.map((x) => h("tr", {},
    h("td", { text: REL[cond.relation] || cond.relation }), h("td", { class: "num", text: x.k }),
    h("td", { class: "num", text: x.status === "ok" ? f2(x.auc) : "—" }),
    h("td", { class: "num", text: x.status === "ok" ? interval(x.interval) : "—" }),
    h("td", { text: x.comparator || "—" }),
    h("td", { class: "num", text: x.status === "ok" ? x.holm_p.toPrecision(2) : "—" }),
    h("td", { text: x.status === "ok" ? x.wording : `${x.status}: ${x.reason || ""}` }))));
  const title = card.sample ? `${card.sample.role} · ${card.sample.title}` : card.filename;
  sheet.replaceChildren(
    h("header", { class: "sheet-head" },
      h("p", { class: "brand", text: "BEAT Audit Live · 결과 기록" }),
      h("h1", { id: "print-title", text: title }),
      h("p", { text: `${kindWord(input.kind)} · ${whatText(input)}` })),
    card.stored ? h("p", { class: "stored-banner", text: `저장된 실행 — ${clock(card.stored.stored_utc)}에 기록. 지금 계산한 결과가 아닙니다.` }) : "",
    h("section", { class: "sheet-card" }, resultBody(card, { forPrint: true })),
    h("section", {},
      h("h2", { text: "입력" }),
      h("dl", { class: "facts" },
        h("dt", { text: "파일" }), h("dd", { text: summary.filename }),
        h("dt", { text: "SHA-256" }), h("dd", {}, h("code", { text: input.sha256 })),
        h("dt", { text: "전송 출력" }), h("dd", { text: `${kindText(input)}, 과제 단서 ${input.cues ? `있음 (${input.cue_levels.join(", ")})` : "없음"}` }),
        h("dt", { text: "기록일 순서" }), h("dd", { text: input.day_order.join(" → ") }),
        card.sample?.selection ? [h("dt", { text: "표본 규칙" }), h("dd", { text: card.sample.selection })] : null,
        h("dt", { text: "실행" }), h("dd", { text: card.stored ? `저장된 실행, ${clock(card.stored.stored_utc)} 기록` : `실시간, ${clock(summary.finished_utc)}, ${duration(summary.elapsed_seconds)}` }))),
    h("section", {},
      h("h2", { text: "설정" }),
      h("table", { class: "cells" },
        h("thead", {}, h("tr", {}, h("th", { text: "항목" }), h("th", { text: "이 실행 (시연 설정)" }), h("th", { text: "도구 표준" }))),
        h("tbody", {},
          h("tr", {}, h("td", { text: "순열 검정" }), h("td", { text: `${num(st.permutations)}회` }), h("td", { text: `${num(st.standard.permutations)}회` })),
          h("tr", {}, h("td", { text: "부트스트랩" }), h("td", { text: `${num(st.bootstraps)}회` }), h("td", { text: `${num(st.standard.bootstraps)}회` })),
          h("tr", {}, h("td", { text: `공격 (${kindWord(input.kind)})` }), h("td", { text: st.comparators.join(", ") }), h("td", { text: st.standard.comparators.join(", ") })),
          h("tr", {}, h("td", { text: "갤러리·closed-set·sequential" }), h("td", { text: "생략" }), h("td", { text: "실행" })),
          h("tr", {}, h("td", { text: "출력 수 k" }), h("td", { text: st.k_grid.join(", ") }), h("td", { text: "1, 5, 10, 20, 50, 100" })),
          h("tr", {}, h("td", { text: "출력 1개의 시간" }), h("td", { text: st.seconds_per_output ? `${st.seconds_per_output}초` : "지정 안 함" }), h("td", { text: "—" })))),
      h("p", { class: "small" }, `도구 ${summary.tool_version} · source ${summary.hashes.source.slice(0, 16)}… · config ${summary.hashes.configuration.slice(0, 16)}… · seed ${st.seed} / fold seed ${st.fold_seed}`),
      reduced(st) ? h("p", { class: "small", text: `더 강한 공격은 더 높게 잴 수 있음. ${REDUCTION_NOTE}` }) : null,
      summary.warnings.length ? h("p", { class: "small" }, "도구 경고: ", warningItems(summary).map((w, i) => [i ? " · " : "", w])) : null),
    h("section", {},
      h("h2", { text: "모든 셀" }),
      h("table", { class: "cells" },
        h("thead", {}, h("tr", {}, ["조건", "k", "AUC", "95% 구간", "공격", "Holm p", "도구 판정"].map((t) => h("th", { text: t })))),
        h("tbody", {}, rows))),
    h("section", {},
      h("h2", { text: "읽을 때의 한계" }),
      h("ul", { class: "limits" },
        h("li", { text: "시연용 부분집합, 탐색: 프로젝트의 등록된 수치가 아닙니다." }),
        h("li", { text: "AUC는 두 출력 묶음을 같은 사람으로 연결하는 점수입니다. 신원 확인율이 아닙니다. 0.5가 우연입니다." }),
        h("li", { text: "공격자는 전송된 출력과, 있으면 각 시행의 과제 단서를 봅니다. 대상 사용자는 공격자가 학습에 쓴 사용자와 겹치지 않습니다(열린 집합)." }),
        h("li", { text: "검출되지 않음은 증거 부족이지 없음이 아닙니다. 어떤 결과도 이 출력이 안전하거나 익명이라는 뜻이 아닙니다." }),
        h("li", { text: reduced(st)
          ? "추론 예산과 공격 종류를 줄인 시연 설정입니다. 도구는 이를 비표준 예산으로 표기합니다."
          : "추론 예산을 줄인 시연 설정입니다(공격은 도구 표준과 같음). 도구는 이를 비표준 예산으로 표기합니다." }),
        h("li", { text: "공격을 바꾸거나 데이터를 늘리면 값이 달라질 수 있습니다. 한 번의 실행은 하한의 한 측정입니다." }))));
  state.printReturn = document.activeElement;
  const view = $("print-view");
  for (const el of document.body.children) if (el !== view) el.inert = true;
  view.hidden = false;
  document.body.classList.add("printing");
  $("print-now").focus();
}

function closePrint() {
  const view = $("print-view");
  view.hidden = true;
  for (const el of document.body.children) el.inert = false;
  document.body.classList.remove("printing");
  state.printReturn?.focus();
}

/* Tab and Shift+Tab cycle inside the open print view. */
function trapPrintFocus(e) {
  const view = $("print-view");
  const items = [...view.querySelectorAll("button, a[href], [tabindex]:not([tabindex='-1'])")];
  if (!items.length) return;
  const first = items[0], last = items[items.length - 1];
  if (!view.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
  else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

/* ---------- wiring ---------- */

function announce(text) { $("status").textContent = text; }

function wire() {
  $("pick").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });
  $("clear").addEventListener("click", () => { [...state.cards].forEach((c) => removeCard(c, true)); announce("카드를 모두 지웠습니다."); $("pick").focus(); });
  $("print-now").addEventListener("click", () => window.print());
  $("print-close").addEventListener("click", closePrint);
  document.addEventListener("keydown", (e) => {
    if ($("print-view").hidden) return;
    if (e.key === "Escape") closePrint();
    else if (e.key === "Tab") trapPrintFocus(e);
  });
  let depth = 0;
  const veil = $("drop-veil");
  window.addEventListener("dragenter", (e) => { if ([...e.dataTransfer.types].includes("Files")) { depth++; veil.hidden = false; e.preventDefault(); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) veil.hidden = true; });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    veil.hidden = true;
    const files = [...e.dataTransfer.files];
    if (files.length > 1) announce("한 번에 한 파일씩 놓아 주세요. 첫 파일만 감사합니다.");
    handleFile(files[0]);
  });
}

async function init() {
  wire();
  layout();
  if (location.protocol === "file:") return offline();
  try {
    const r = await fetch("/api/info");
    if (!r.ok) throw new Error(String(r.status));
    const info = await r.json();
    state.maxBytes = info.max_bytes;
  } catch (err) {
    offline();
  }
}

function offline() {
  $("offline").hidden = false;
  $("app").hidden = true;
}

init();
