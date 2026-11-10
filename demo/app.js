// BEAT demo. All numbers come from data/*.json written by scripts/export_demo_data.py.
const NS = "http://www.w3.org/2000/svg";
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

const CONDITIONS = {
  "P0": { name: "Raw features (253-d)", short: "Raw", color: "#c2410c", dash: "" },
  "P0-RC": { name: "Raw + recentering", short: "Raw + RC", color: "#a16207", dash: "5 4" },
  "P2": { name: "8-d task bottleneck", short: "Bottleneck", color: "#1d4ed8", dash: "" },
  "P2-RC": { name: "Bottleneck + recentering", short: "Bottleneck + RC", color: "#0e7490", dash: "5 4" },
  "P3": { name: "Bottleneck + identity adversary", short: "Adversary", color: "#15803d", dash: "" },
  "P3-RC": { name: "Bottleneck + adversary + recentering", short: "Adversary + RC", color: "#a21caf", dash: "5 4" },
};
const ATTACKERS = { logistic: "logistic regression", rbf_svm: "RBF SVM", mlp: "MLP" };
const EXP_LETTER = { exp_b: "B", exp_c: "C", exp_d: "D", exp_e: "E" };
const cond = (c) => CONDITIONS[c] || { name: c, short: c, color: "#4a5163", dash: "" };

// ------------------------------------------------------------------ helpers
const $ = (sel, root = document) => root.querySelector(sel);
function el(tag, attrs = {}, parent) {
  const isSvg = ["svg", "g", "line", "path", "circle", "rect", "text", "polyline", "title", "desc"].includes(tag);
  const node = isSvg ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}
const f2 = (v) => (v == null ? "–" : v.toFixed(2));
const f3 = (v) => (v == null ? "–" : v.toFixed(3));
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
const FORMATS = {
  raw: (v) => String(v),
  2: f2,
  3: f3,
  pct,
  range2: (v) => `${f2(v[0])}–${f2(v[1])}`,
  rangems: (v) => `${v[0].toFixed(1)}–${v[1].toFixed(1)} ms`,
  ofn: (v) => `${v[0]} of ${v[1]}`,
};

const tip = $("#tooltip");
function showTip(target, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = target.getBoundingClientRect();
  const t = tip.getBoundingClientRect();
  let x = r.left + r.width / 2 - t.width / 2;
  x = Math.max(8, Math.min(x, innerWidth - t.width - 8));
  let y = r.top - t.height - 10;
  if (y < 60) y = r.bottom + 10;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; };
function hoverable(node, html, onActivate) {
  // Chart dots are small; an invisible 24 px circle gives touch users a usable target.
  const dot = node.querySelector("circle");
  if (dot) node.insertBefore(el("circle", { cx: dot.getAttribute("cx"), cy: dot.getAttribute("cy"), r: 12, fill: "transparent", class: "hit" }), dot);
  node.setAttribute("tabindex", "0");
  node.setAttribute("role", "img");
  node.setAttribute("aria-label", html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim());
  const show = () => showTip(node, html);
  node.addEventListener("mouseenter", show);
  node.addEventListener("focus", show);
  node.addEventListener("mouseleave", hideTip);
  node.addEventListener("blur", hideTip);
  if (onActivate) {
    node.addEventListener("click", onActivate);
    node.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onActivate(); } });
  }
}
addEventListener("scroll", hideTip, { passive: true });

function radios(container, name, options, value, onChange) {
  container.innerHTML = "";
  for (const o of options) {
    const label = el("label", {}, container);
    const input = el("input", { type: "radio", name, value: String(o.value) }, label);
    input.checked = String(o.value) === String(value);
    el("span", { text: o.label }, label);
    if (o.title) label.title = o.title;
  }
  bindRadios(container, onChange);
}
function bindRadios(container, onChange) {
  container.addEventListener("change", (e) => { if (e.target.matches("input[type=radio]")) onChange(e.target.value); });
}
function chips(container, items, onToggle) {
  container.innerHTML = "";
  for (const it of items) {
    const b = el("button", { type: "button", class: "chip", "aria-pressed": String(it.on), disabled: it.disabled ? "" : null, title: it.title }, container);
    b.style.setProperty("--c", it.color);
    b.textContent = it.label;
    if (!it.disabled) b.addEventListener("click", () => {
      const on = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", String(on));
      onToggle(it.key, on);
    });
  }
}
function table(container, head, rows) {
  container.innerHTML = "";
  const wrap = el("div", { class: "table-scroll" }, container);
  const t = el("table", {}, wrap);
  const tr = el("tr", {}, el("thead", {}, t));
  head.forEach((h) => el("th", { scope: "col", text: h }, tr));
  const tb = el("tbody", {}, t);
  // data-label lets the narrow-screen CSS show each row as label/value pairs.
  rows.forEach((r) => { const row = el("tr", {}, tb); r.forEach((c, i) => el("td", { text: c, "data-label": head[i] }, row)); });
}

// Responsive SVG chart frame drawn at the container's pixel width.
function frame(container, { xDomain, yDomain, xScale = "linear", xTicks, yTicks, xLabel, yLabel, xFmt = String, yFmt = f2 }) {
  container.innerHTML = "";
  const W = Math.max(280, container.clientWidth);
  const H = Math.round(Math.min(380, Math.max(250, W * 0.66)));
  const m = { l: 48, r: 14, t: 12, b: 44 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group" }, container);
  const tx = xScale === "log" ? Math.log10 : (v) => v;
  const [x0, x1] = xDomain.map(tx);
  const x = (v) => m.l + ((tx(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const y = (v) => H - m.b - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * (H - m.t - m.b);
  const g = el("g", { class: "axis" }, svg);
  for (const v of yTicks) {
    el("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, g);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", text: yFmt(v) }, g);
  }
  for (const v of xTicks) {
    el("line", { class: "grid-line", x1: x(v), x2: x(v), y1: m.t, y2: H - m.b }, g);
    el("text", { x: x(v), y: H - m.b + 16, "text-anchor": "middle", text: xFmt(v) }, g);
  }
  el("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", text: xLabel }, svg);
  el("text", { class: "axis-title", x: -(m.t + H - m.b) / 2, y: 13, transform: "rotate(-90)", "text-anchor": "middle", text: yLabel }, svg);
  return { svg, x, y, W, H, m };
}
function refLine(fr, { y, x, label }) {
  if (y != null) {
    el("line", { class: "ref-line", x1: fr.m.l, x2: fr.W - fr.m.r, y1: fr.y(y), y2: fr.y(y) }, fr.svg);
    el("text", { class: "ref-label", x: fr.W - fr.m.r - 4, y: fr.y(y) - 4, "text-anchor": "end", text: label }, fr.svg);
  } else {
    el("line", { class: "ref-line", y1: fr.m.t, y2: fr.H - fr.m.b, x1: fr.x(x), x2: fr.x(x) }, fr.svg);
    el("text", { class: "ref-label", x: fr.x(x) + 4, y: fr.m.t + 12, text: label }, fr.svg);
  }
}
function onResize(node, draw) {
  let w = 0;
  new ResizeObserver(() => { if (node.clientWidth !== w) { w = node.clientWidth; draw(); } }).observe(node);
}

// ------------------------------------------------------------------ boot
const DATA = await fetch("data/results.json").then((r) => r.json());
const H = DATA.headline;
const CHANCE = H.chance.value;

document.querySelectorAll("[data-h]").forEach((node) => {
  const [key, ...rest] = node.dataset.h.split(".");
  const entry = H[key];
  const value = rest.length ? get(entry.value, rest.join(".")) : entry.value;
  node.textContent = (FORMATS[node.dataset.fmt] || String)(value);
  node.title = `Source: ${entry.source.file}`;
});
$("#seal-hash").textContent = DATA.seal.commit.slice(0, 7);
$("#seal-hash").title = `${DATA.seal.commit} (${DATA.seal.source.file})`;
document.querySelectorAll("[data-placeholder]").forEach((a) => a.addEventListener("click", (e) => e.preventDefault()));
{
  const s = $("#exp-status");
  el("p", { text: "Experiments in this demo:" }, s);
  const ul = el("ul", {}, s);
  for (const [k, v] of Object.entries(DATA.experiments)) {
    el("li", { text: `${v.title}: ${v.status === "available" ? "results shown" : "being evaluated; results pending"}` }, ul);
  }
}

// ------------------------------------------------------------------ replay
const RM = DATA.replay.meta;
const CTRL = DATA.control.rows;
const OBS = DATA.observation.rows;
const state = { subject: 1, condition: "P2", mode: "command", tau: 0.8, runIndex: 0, seed: String(RM.seeds[0]), step: 0 };
const subjectCache = new Map();
const loadSubject = (s) => {
  if (!subjectCache.has(s)) subjectCache.set(s, fetch(DATA.replay.files[s]).then((r) => r.json()));
  return subjectCache.get(s);
};

radios($("#ctl-subject"), "subject", Object.keys(DATA.replay.files).map((s) => ({ value: s, label: `S${s}` })), state.subject,
  (v) => { state.subject = Number(v); renderReplay(true); });
radios($("#ctl-tau"), "tau", RM.taus.map((t) => ({ value: t, label: t === 0.5 ? "0.5 (no hold)" : String(t) })), state.tau,
  (v) => { state.tau = Number(v); renderReplay(true); });
bindRadios($("#ctl-mode"), (v) => { state.mode = v; renderReplay(false); });
const condSel = $("#ctl-condition");
for (const c of RM.conditions) {
  el("option", { value: c, text: `${cond(c).name}${c.startsWith("P3") ? ` (λ = ${H.selected_lambda.value})` : ""}` }, condSel);
}
condSel.value = state.condition;
condSel.addEventListener("change", () => { state.condition = condSel.value; renderReplay(true); });
const runSel = $("#ctl-run");
const seedSel = $("#ctl-seed");
RM.seeds.forEach((s) => el("option", { value: s, text: `${s + 1} of ${RM.seeds.length}` }, seedSel));
seedSel.addEventListener("change", () => { state.seed = seedSel.value; renderReplay(true); });
runSel.addEventListener("change", () => { state.runIndex = Number(runSel.value); renderReplay(true); });

function simulate(cmds) {
  const { step_deg: step, range_deg: range } = RM.controller;
  let a = 0;
  return cmds.map((c) => {
    const d = c === "R" ? 1 : c === "L" ? -1 : 0;
    const p = a + step * d;
    if (!(d !== 0 && Math.abs(p) > range)) a = p;
    return a;
  });
}

// arm dial
const armSvg = $("#arm-svg");
const CX = 160, CY = 168, R = 128;
const polar = (deg, r = R) => [CX + r * Math.sin((deg * Math.PI) / 180), CY - r * Math.cos((deg * Math.PI) / 180)];
{
  const [ax, ay] = polar(-90), [bx, by] = polar(90);
  el("path", { d: `M ${ax} ${ay} A ${R} ${R} 0 0 1 ${bx} ${by}`, fill: "#f1f4fa", stroke: "#c9cfdb" }, armSvg);
  for (const t of [-90, -45, 0, 45, 90]) {
    const [x1, y1] = polar(t, R - 8), [x2, y2] = polar(t, R);
    el("line", { x1, y1, x2, y2, stroke: "#8e95a6" }, armSvg);
    const [lx, ly] = polar(t, R + 12);
    el("text", { x: lx, y: ly + 4, "text-anchor": "middle", "font-size": 11, fill: "#4a5163", text: `${t}°` }, armSvg);
  }
  el("text", { x: CX - R - 4, y: CY + 18, "font-size": 12, "font-weight": 700, fill: "#4a5163", text: "◀ left" }, armSvg);
  el("text", { x: CX + R + 4, y: CY + 18, "text-anchor": "end", "font-size": 12, "font-weight": 700, fill: "#4a5163", text: "right ▶" }, armSvg);
}
const targetArm = el("g", { class: "rot" }, armSvg);
el("line", { x1: CX, y1: CY, x2: CX, y2: CY - R + 14, stroke: "var(--target)", "stroke-width": 3, "stroke-dasharray": "6 5", "stroke-linecap": "round" }, targetArm);
el("circle", { cx: CX, cy: CY - R + 14, r: 6, fill: "#fff", stroke: "var(--target)", "stroke-width": 2.5 }, targetArm);
const arm = el("g", { class: "rot" }, armSvg);
el("line", { x1: CX, y1: CY, x2: CX, y2: CY - R + 22, stroke: "var(--arm)", "stroke-width": 9, "stroke-linecap": "round" }, arm);
el("circle", { cx: CX, cy: CY - R + 22, r: 9, fill: "var(--arm)" }, arm);
el("circle", { cx: CX, cy: CY, r: 10, fill: "#161a26" }, armSvg);
const armStyle = document.createElement("style");
armStyle.textContent = "#arm-svg .rot{transition:transform .35s ease}";
document.head.appendChild(armStyle);
const setRot = (g, deg) => { g.style.transform = `rotate(${deg}deg)`; g.style.transformOrigin = `${CX}px ${CY}px`; };

// trajectory
const trajSvg = $("#traj-svg");
function drawTrajectory(target, angles, step) {
  trajSvg.querySelectorAll(":scope > :not(desc)").forEach((n) => n.remove());
  const W = Math.max(280, Math.round(trajSvg.parentElement.clientWidth - 32)), Hh = W < 480 ? 190 : 220;
  const m = { l: 40, r: 10, t: 10, b: 30 };
  trajSvg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
  const lim = Math.min(RM.controller.range_deg, Math.max(30, Math.ceil(Math.max(...target.map(Math.abs), ...angles.map(Math.abs)) / 15) * 15));
  const n = RM.episode_length;
  const x = (i) => m.l + (i / n) * (W - m.l - m.r);
  const y = (v) => m.t + ((lim - v) / (2 * lim)) * (Hh - m.t - m.b);
  const g = el("g", { class: "axis" }, trajSvg);
  for (let v = -lim; v <= lim; v += lim / 2) {
    el("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v === 0 ? "#b9bfcc" : null }, g);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", text: `${v}°` }, g);
  }
  for (let i = 0; i <= n; i += 5) el("text", { x: x(i), y: Hh - m.b + 16, "text-anchor": "middle", text: i }, g);
  el("text", { class: "axis-title", x: W - m.r, y: Hh - 2, "text-anchor": "end", text: "trial" }, trajSvg);
  const pts = (arr, upto) => [[x(0), y(0)], ...arr.slice(0, upto).map((v, i) => [x(i + 1), y(v)])].map((p) => p.join(",")).join(" ");
  el("polyline", { points: pts(target, n), fill: "none", stroke: "var(--target)", "stroke-width": 2.5, "stroke-dasharray": "6 5" }, trajSvg);
  el("polyline", { points: pts(angles, step), fill: "none", stroke: "var(--arm)", "stroke-width": 3, "stroke-linejoin": "round" }, trajSvg);
  const cur = step === 0 ? 0 : angles[step - 1];
  el("line", { x1: x(step), x2: x(step), y1: m.t, y2: Hh - m.b, stroke: "#161a26", "stroke-opacity": 0.25 }, trajSvg);
  el("circle", { cx: x(step), cy: y(cur), r: 5, fill: "var(--arm)" }, trajSvg);
}

let current = null;
let timer = null;
const playBtn = $("#btn-play");
const scrub = $("#scrub");
function stop() {
  clearInterval(timer); timer = null;
  playBtn.textContent = "▶ Play"; playBtn.setAttribute("aria-label", "Play");
}
function play() {
  if (state.step >= RM.episode_length) state.step = 0;
  playBtn.textContent = "❚❚ Pause"; playBtn.setAttribute("aria-label", "Pause");
  draw();
  timer = setInterval(() => {
    state.step += 1; draw();
    if (state.step >= RM.episode_length) stop();
  }, reduceMotion.matches ? 900 : 650);
}
playBtn.addEventListener("click", () => (timer ? stop() : play()));
$("#btn-step").addEventListener("click", () => { stop(); state.step = Math.min(RM.episode_length, state.step + 1); draw(); });
$("#btn-reset").addEventListener("click", () => { stop(); state.step = 0; draw(); });
scrub.addEventListener("input", () => { stop(); state.step = Number(scrub.value); draw(); });

async function renderReplay(resetStep) {
  const data = await loadSubject(String(state.subject));
  if (runSel.options.length !== data.runs.length) {
    runSel.innerHTML = "";
    data.runs.forEach((r, i) => el("option", { value: i, text: `Run ${r}` }, runSel));
  }
  runSel.value = String(state.runIndex);
  const tauKey = String(state.tau);
  const cmds = [...data.commands[state.condition][state.seed][tauKey][state.runIndex]];
  const labels = [...data.labels[state.runIndex]];
  const stream = state.mode === "feature" ? "feature" : `command@${tauKey}`;
  const attacks = {};
  for (const k of RM.k) attacks[k] = [...data.attacks[state.condition][state.seed][stream][String(k)][state.runIndex]].map(Number);
  current = { data, cmds, labels, angles: simulate(cmds), target: simulate(labels), attacks, stream };
  if (resetStep) { stop(); state.step = 0; }
  renderStats();
  draw();
}

function draw() {
  if (!current) return;
  const { cmds, labels, angles, target, attacks, stream } = current;
  const n = RM.episode_length, s = state.step;
  scrub.value = s;
  $("#scrub-out").textContent = `${s} / ${n}`;
  setRot(arm, s ? angles[s - 1] : 0);
  setRot(targetArm, s ? target[s - 1] : 0);
  const word = { L: "left", R: "right", H: "hold" };
  $("#arm-readout").innerHTML = s === 0
    ? "Press <b>Play</b> or step through the episode."
    : `Trial ${s}: imagined <b>${word[labels[s - 1]]}</b> → sent <b>${word[cmds[s - 1]]}</b> · arm ${angles[s - 1]}°, target ${target[s - 1]}°`;
  drawTrajectory(target, angles, s);

  const strip = $("#cmd-strip");
  strip.innerHTML = "";
  cmds.forEach((c, i) => {
    const shown = i < s;
    const cls = !shown ? "" : c === "H" ? "hold" : c === labels[i] ? "ok" : "bad";
    const li = el("li", { class: `${cls}${i === s - 1 ? " now" : ""}`, text: shown ? c : "·" }, strip);
    li.setAttribute("aria-label", shown ? `Trial ${i + 1}: imagined ${word[labels[i]]}, sent ${word[c]}` : `Trial ${i + 1}: not yet sent`);
  });

  const rows = $("#attack-rows");
  rows.innerHTML = "";
  const truth = state.subject;
  for (const k of RM.k) {
    const preds = attacks[k];
    const row = el("div", { class: "arow" }, rows);
    el("div", { class: "arow-label", html: `<b>k = ${k}</b><small>${k === 1 ? "each output" : `${k} outputs per guess`}</small>` }, row);
    const slots = el("div", { class: "slots" }, row);
    slots.style.setProperty("--n", preds.length);
    let hits = 0, done = 0;
    preds.forEach((p, j) => {
      const ready = s >= (j + 1) * k;
      const filling = !ready && s > j * k;
      if (ready) { done += 1; if (p === truth) hits += 1; }
      const slot = el("div", { class: `slot ${ready ? (p === truth ? "hit" : "miss") : "pending"}` }, slots);
      if (ready) slot.textContent = k === 1 ? String(p) : `S${p}`;
      else if (filling) slot.style.background = `linear-gradient(90deg, #dfe6f5 ${((s - j * k) / k) * 100}%, #fafbfc 0)`;
      slot.setAttribute("aria-label", ready ? `Guess ${j + 1}: user ${p}, ${p === truth ? "correct" : "wrong"}` : `Guess ${j + 1}: waiting for outputs`);
      slot.title = slot.getAttribute("aria-label");
    });
    const info = RM.attacker[`${state.condition}|${stream}|${k}`];
    const agg = OBS.find((r) => r.experiment === "exp_c" && r.condition === state.condition && r.stream === stream && r.k === k);
    el("div", { class: "arow-sum", html: `${done ? `<b>${hits}/${done}</b> right here` : "&nbsp;"}<br>all users: <b>${f2(agg?.ba_mean)}</b> <span title="${ATTACKERS[info.attacker]}, ${info.scenario} scenario">(${ATTACKERS[info.attacker]})</span>` }, row);
  }
  const scen = new Set(RM.k.map((k) => RM.attacker[`${state.condition}|${stream}|${k}`].scenario));
  $("#attack-hint").innerHTML = `True user: <b>S${truth}</b> · attacker sees <b>${state.mode === "feature" ? "feature vectors" : "commands only"}</b>`;
  $("#attack-note").textContent = `Each box is one stored guess from a separate group of k consecutive outputs; green = correct user, red = wrong. Chance is ${f2(CHANCE)}. `
    + `Attackers were trained ${scen.has("main") ? "on the other half of session E (runs 1–3 ↔ 4–6)" : ""}${scen.size > 1 ? " or " : ""}${scen.has("auxiliary") ? "on session T" : ""}, never on the run shown. `
    + "Features and commands move the arm identically; only what the attacker sees differs. "
    + `Arm, commands and boxes show a single model seed (${Number(state.seed) + 1} of ${RM.seeds.length}) and one E run; “all users” values are 5-seed means of the strongest attacker.`;
}

function renderStats() {
  const row = CTRL.find((r) => r.condition === state.condition && r.tau === state.tau && r.mode === state.mode);
  const stream = state.mode === "feature" ? "feature" : `command@${state.tau}`;
  const ob = (k) => OBS.find((r) => r.experiment === "exp_c" && r.condition === state.condition && r.stream === stream && r.k === k);
  const items = [
    ["Mean angle error", `${row.mae_deg_mean.toFixed(1)}°`],
    ["Wrong-direction moves", `${row.wrong_direction_moves_mean.toFixed(1)} / episode`],
    ["Commands held", pct(row.hold_rate_mean)],
    ["Episodes ending within 10°", pct(row.reach_rate_mean)],
    ["Identity score, 1 output", f2(ob(1).ba_mean)],
    ["Identity score, 20 outputs", f2(ob(20).ba_mean)],
  ];
  const dl = $("#replay-stats");
  dl.innerHTML = "";
  for (const [k, v] of items) { const d = el("div", {}, dl); el("dt", { text: k }, d); el("dd", { text: v }, d); }
}
renderReplay(true);
onResize(trajSvg.parentElement, draw);

// ------------------------------------------------------------------ fooled adversary
const ADV = DATA.adversary.rows.filter((r) => r.experiment === "exp_b");
const LAMBDAS = [...new Set(ADV.map((r) => r.target))].sort((a, b) => a - b);
let fam = "P3";
const lam = $("#lambda");
lam.max = LAMBDAS.length - 1;
lam.value = String(LAMBDAS.indexOf(1) >= 0 ? LAMBDAS.indexOf(1) : LAMBDAS.length - 1);
bindRadios($("#fooled-family"), (v) => { fam = v; drawFooled(); });
lam.addEventListener("input", drawFooled);
function drawFooled() {
  const rows = ADV.filter((r) => r.condition === fam).sort((a, b) => a.target - b.target);
  const sel = rows[Number(lam.value)];
  const fr = frame($("#fooled-chart"), {
    xDomain: [LAMBDAS[0] / 1.6, LAMBDAS.at(-1) * 1.6], yDomain: [0, 1], xScale: "log",
    xTicks: LAMBDAS, yTicks: [0, 0.25, 0.5, 0.75, 1], xLabel: "adversary strength λ (log scale)", yLabel: "identity score",
  });
  refLine(fr, { y: CHANCE, label: `chance ${f2(CHANCE)}` });
  el("rect", { x: fr.x(sel.target) - 14, width: 28, y: fr.m.t, height: fr.H - fr.m.t - fr.m.b, fill: "#f2a900", "fill-opacity": 0.14 }, fr.svg);
  const selected = rows.find((r) => r.selected);
  if (selected) el("text", { class: "ref-label", x: fr.x(selected.target) + 6, y: fr.H - fr.m.b - 8, text: "selected" }, fr.svg);
  const series = [
    { key: "adversary_main_ba", label: "Co-trained adversary head (5-seed mean)", color: "var(--head)", dash: "5 4" },
    { key: "independent_ba", label: "Strongest independent attacker (5-seed mean)", color: "var(--ind)", dash: "" },
  ];
  for (const s of series) {
    el("polyline", { points: rows.map((r) => `${fr.x(r.target)},${fr.y(r[s.key])}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 3, "stroke-dasharray": s.dash }, fr.svg);
    rows.forEach((r, i) => {
      const g = el("g", { class: `pt${r === sel ? " active" : ""}` }, fr.svg);
      el("circle", { cx: fr.x(r.target), cy: fr.y(r[s.key]), r: 6, fill: s.color, stroke: "#fff", "stroke-width": 2 }, g);
      hoverable(g, `<b>${s.label}</b><br>λ = ${r.target} · identity score ${f3(r[s.key])}${s.key === "independent_ba" ? `<br>${ATTACKERS[r.strongest_attacker]}, ${r.strongest_scenario} scenario` : ""}`,
        () => { lam.value = String(i); drawFooled(); });
    });
  }
  // Legend sits outside the plot so it never covers points at narrow widths.
  $("#fooled-legend").innerHTML = series.map((s) =>
    `<span><svg width="24" height="8" aria-hidden="true"><line x1="0" x2="24" y1="4" y2="4" stroke="${s.color}" stroke-width="3" stroke-dasharray="${s.dash}"/></svg>${s.label}</span>`).join("");
  $("#lambda-out").textContent = `λ = ${sel.target}${sel.selected ? " (selected on session T)" : ""}`;
  lam.setAttribute("aria-valuetext", `lambda ${sel.target}`);
  $("#v-head").textContent = f2(sel.adversary_main_ba);
  $("#v-head-sub").textContent = `5-seed mean, main scenario · ${(sel.adversary_main_ba / CHANCE).toFixed(1)}× chance (chance ${f2(CHANCE)})`;
  $("#v-ind").textContent = f2(sel.independent_ba);
  $("#v-ind-sub").textContent = `5-seed mean, strongest: ${ATTACKERS[sel.strongest_attacker]}, ${sel.strongest_scenario} scenario · ${(sel.independent_ba / CHANCE).toFixed(1)}× chance`;
  table($("#fooled-table"), ["λ", "head (5-seed mean)", "strongest independent (5-seed mean)", "strongest attacker"],
    rows.map((r) => [String(r.target) + (r.selected ? " (selected)" : ""), f3(r.adversary_main_ba), f3(r.independent_ba), ATTACKERS[r.strongest_attacker]]));
}
onResize($("#fooled-chart"), drawFooled);

// ------------------------------------------------------------------ utility vs exposure
const TRADE = DATA.tradeoff.rows;
const seriesKey = (r) => `${r.experiment}|${r.condition}${r.d != null ? `|d${r.d}` : ""}`;
const multiExp = (rows) => new Set(rows.map((r) => r.experiment)).size > 1;
const seriesLabel = (key, rows) => {
  const [exp, c, d] = key.split("|");
  return `${multiExp(rows) ? `${EXP_LETTER[exp] || exp}: ` : ""}${cond(c).short}${d ? ` ${d}` : ""}`;
};
const tradeOn = new Map();
const expOn = new Map(Object.entries(DATA.experiments).map(([k, v]) => [k, v.status === "available"]));
function tradeChips() {
  const keys = [...new Set(TRADE.filter((r) => expOn.get(r.experiment)).map(seriesKey))];
  keys.forEach((k) => { if (!tradeOn.has(k)) tradeOn.set(k, true); });
  chips($("#trade-toggles"), keys.map((k) => ({ key: k, label: seriesLabel(k, TRADE), color: cond(k.split("|")[1]).color, on: tradeOn.get(k) })),
    (k, on) => { tradeOn.set(k, on); drawTrade(); });
  const tradeExps = Object.entries(DATA.experiments).filter(([k]) => k !== "exp_c");
  chips($("#trade-exps"), tradeExps.map(([k, v]) => {
    const has = TRADE.some((r) => r.experiment === k);
    return { key: k, label: `Exp. ${EXP_LETTER[k]}${has ? "" : " · pending"}`, color: "#4a5163", on: has && expOn.get(k), disabled: !has, title: v.title };
  }), (k, on) => { expOn.set(k, on); tradeChips(); drawTrade(); });
}
function drawTrade() {
  const rows = TRADE.filter((r) => expOn.get(r.experiment) && tradeOn.get(seriesKey(r)));
  const all = TRADE.filter((r) => expOn.get(r.experiment));
  const lo = Math.floor((Math.min(...all.map((r) => r.task_low ?? r.task_ba)) - 0.01) * 20) / 20;
  const hi = Math.ceil((Math.max(...all.map((r) => r.task_high ?? r.task_ba)) + 0.01) * 20) / 20;
  const yt = []; for (let v = lo; v <= hi + 1e-9; v += 0.05) yt.push(Math.round(v * 100) / 100);
  const fr = frame($("#trade-chart"), {
    xDomain: [0, 1.02], yDomain: [lo, hi], xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: yt,
    xLabel: "identity score (lower = more private)", yLabel: "left/right accuracy (BA)",
  });
  refLine(fr, { x: CHANCE, label: "chance" });
  const groups = new Map();
  rows.forEach((r) => { const k = seriesKey(r); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); });
  for (const [k, pts] of groups) {
    const c = cond(k.split("|")[1]);
    pts.sort((a, b) => a.target - b.target);
    if (pts.length > 1) el("polyline", { points: pts.map((r) => `${fr.x(r.identity_ba)},${fr.y(r.task_ba)}`).join(" "), fill: "none", stroke: c.color, "stroke-width": 1.5, "stroke-dasharray": c.dash, "stroke-opacity": 0.7 }, fr.svg);
    for (const r of pts) {
      const g = el("g", { class: "pt" }, fr.svg);
      if (r.identity_low != null) el("line", { x1: fr.x(r.identity_low), x2: fr.x(r.identity_high), y1: fr.y(r.task_ba), y2: fr.y(r.task_ba), stroke: c.color, "stroke-opacity": 0.3, "stroke-width": 1.5 }, g);
      if (r.task_low != null) el("line", { x1: fr.x(r.identity_ba), x2: fr.x(r.identity_ba), y1: fr.y(r.task_low), y2: fr.y(r.task_high), stroke: c.color, "stroke-opacity": 0.3, "stroke-width": 1.5 }, g);
      el("circle", { cx: fr.x(r.identity_ba), cy: fr.y(r.task_ba), r: r.selected ? 7 : 5.5, fill: c.color, stroke: "#fff", "stroke-width": 1.5 }, g);
      hoverable(g, `<b>${EXP_LETTER[r.experiment] || r.experiment}: ${cond(r.condition).name}</b>${r.target ? `<br>λ = ${r.target}${r.selected ? " (selected)" : ""}` : ""}${r.d != null ? `<br>d = ${r.d}` : ""}`
        + `<br>accuracy ${f3(r.task_ba)} [${f3(r.task_low)}–${f3(r.task_high)}]<br>identity ${f3(r.identity_ba)} [${f3(r.identity_low)}–${f3(r.identity_high)}]`
        + `<br>strongest: ${ATTACKERS[r.strongest_attacker] || r.strongest_attacker}`);
    }
  }
  table($("#trade-table"), ["model", "λ", "accuracy", "95% CI", "identity", "95% CI"],
    rows.map((r) => [`${EXP_LETTER[r.experiment]}: ${cond(r.condition).name}`, r.target ? String(r.target) : "–", f3(r.task_ba), `${f3(r.task_low)}–${f3(r.task_high)}`, f3(r.identity_ba), `${f3(r.identity_low)}–${f3(r.identity_high)}`]));
}
tradeChips();
onResize($("#trade-chart"), drawTrade);

// ------------------------------------------------------------------ observation length
const OBS_TAUS = [...new Set(OBS.filter((r) => r.mode === "command").map((r) => r.tau))].sort();
let obsMode = "command", obsTau = OBS_TAUS[0], obsStream = `command@${obsTau}`;
const setStream = () => {
  obsStream = obsMode === "feature" ? "feature" : `command@${obsTau}`;
  $("#obs-tau").querySelectorAll("input").forEach((i) => { i.disabled = obsMode === "feature"; });
  drawObs();
};
radios($("#obs-stream"), "obs-mode", [{ value: "command", label: "Commands" }, { value: "feature", label: "Features" }], obsMode,
  (v) => { obsMode = v; setStream(); });
radios($("#obs-tau"), "obs-tau", OBS_TAUS.map((t) => ({ value: t, label: `τ ${t}` })), obsTau,
  (v) => { obsTau = Number(v); setStream(); });
const obsOn = new Map();
const obsKeys = [...new Set(OBS.map(seriesKey))];
obsKeys.forEach((k) => obsOn.set(k, true));
chips($("#obs-toggles"), obsKeys.map((k) => ({ key: k, label: seriesLabel(k, OBS), color: cond(k.split("|")[1]).color, on: true })),
  (k, on) => { obsOn.set(k, on); drawObs(); });
function drawObs() {
  const rows = OBS.filter((r) => r.stream === obsStream && obsOn.get(seriesKey(r)));
  const ks = [...new Set(OBS.map((r) => r.k))].sort((a, b) => a - b);
  const top = Math.max(0.5, ...rows.map((r) => r.ba_high ?? r.ba_mean));
  const yMax = Math.min(1, Math.ceil(top * 10) / 10);
  const yt = []; for (let v = 0; v <= yMax + 1e-9; v += yMax > 0.6 ? 0.2 : 0.1) yt.push(Math.round(v * 10) / 10);
  const fr = frame($("#obs-chart"), {
    xDomain: [ks[0] / 1.35, ks.at(-1) * 1.35], yDomain: [0, yMax], xScale: "log", xTicks: ks, yTicks: yt,
    xLabel: "consecutive outputs seen, k (log scale)", yLabel: "identity score",
  });
  refLine(fr, { y: CHANCE, label: "chance" });
  refLine(fr, { y: 2 * CHANCE, label: "2 × chance" });
  const groups = new Map();
  rows.forEach((r) => { const k = seriesKey(r); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); });
  const dx = (i) => (i - (groups.size - 1) / 2) * 3;
  [...groups].forEach(([key, pts], i) => {
    const c = cond(key.split("|")[1]);
    pts.sort((a, b) => a.k - b.k);
    const X = (r) => fr.x(r.k) + dx(i);
    if (pts.every((r) => r.ba_low != null)) {
      const band = [...pts.map((r) => `${X(r)},${fr.y(r.ba_high)}`), ...pts.slice().reverse().map((r) => `${X(r)},${fr.y(r.ba_low)}`)];
      el("path", { d: `M${band.join("L")}Z`, fill: c.color, "fill-opacity": 0.08 }, fr.svg);
    }
    el("polyline", { points: pts.map((r) => `${X(r)},${fr.y(r.ba_mean)}`).join(" "), fill: "none", stroke: c.color, "stroke-width": 2.5, "stroke-dasharray": c.dash }, fr.svg);
    for (const r of pts) {
      const g = el("g", { class: "pt" }, fr.svg);
      el("circle", { cx: X(r), cy: fr.y(r.ba_mean), r: 5, fill: c.color, stroke: "#fff", "stroke-width": 1.5 }, g);
      hoverable(g, `<b>${cond(r.condition).name}</b><br>${r.stream === "feature" ? "features" : `commands, τ = ${r.tau}`}, k = ${r.k}<br>identity ${f3(r.ba_mean)} [${f3(r.ba_low)}–${f3(r.ba_high)}]<br>${ATTACKERS[r.attacker] || r.attacker}, ${r.scenario} scenario`);
    }
  });
  table($("#obs-table"), ["model", ...ks.map((k) => `k = ${k}`)],
    [...groups].map(([key, pts]) => [seriesLabel(key, OBS), ...ks.map((k) => f3(pts.find((r) => r.k === k)?.ba_mean))]));
}
onResize($("#obs-chart"), drawObs);
