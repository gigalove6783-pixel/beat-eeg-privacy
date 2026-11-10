// BEAT demo. All numbers come from data/*.json written by scripts/export_demo_data.py.
const NS = "http://www.w3.org/2000/svg";
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

// Palette of results/figures/summary (Okabe–Ito hues); orange, sky and pink are darkened
// so every line and marker keeps at least 3:1 contrast on the white cards.
const PAL = { grl: "#b37400", task: "#009e73", head: "#0072b2", ind: "#d55e00", maxent: "#2b7fb8", ensemble: "#a8507f", raw: "#4a5163" };
const CONDITIONS = {
  "P0": { name: "Raw features (253-d)", short: "Raw", color: PAL.raw, dash: "" },
  "P0-RC": { name: "Raw + recentering", short: "Raw + RC", color: PAL.raw, dash: "5 4" },
  "P2": { name: "8-d task bottleneck", short: "Bottleneck", color: PAL.task, dash: "" },
  "P2-RC": { name: "Bottleneck + recentering", short: "Bottleneck + RC", color: PAL.task, dash: "5 4" },
  "P3": { name: "Bottleneck + identity adversary", short: "Adversary", color: PAL.grl, dash: "" },
  "P3-RC": { name: "Bottleneck + adversary + recentering", short: "Adversary + RC", color: PAL.grl, dash: "5 4" },
};
const ATTACKERS = {
  logistic: "logistic regression", rbf_svm: "RBF SVM", mlp: "MLP",
  beat_logistic: "logistic regression", beat_rbf_svm: "RBF SVM", beat_mlp: "MLP",
  paper_ts_svm: "tangent-space SVM (paper's attacker)", paper_xdawn_svm: "XDAWN-SVM (paper)",
  paper_csp_svm: "CSP-SVM (paper)", paper_csp_lda: "CSP-LDA (paper)", mean_tangent_template: "mean tangent template",
};
const EXP_LETTER = { exp_a: "A", exp_b: "B", exp_c: "C", exp_d: "D", exp_e: "E", exp_g: "G", exp_l: "L", exp_k: "K", exp_m: "M" };
// Result-table scenario names as labelled in README.md.
const DAY_OF = { main: "within-day", auxiliary: "cross-day" };
const dayOf = (s) => DAY_OF[s] || s;
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
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
// p-values: below 1e-4 in scientific notation so they never read as p = 0.
const pval = (v) => {
  if (v == null) return null;
  if (v >= 0.01) return v.toFixed(2);
  if (v >= 1e-4) return String(Number(v.toPrecision(2)));
  const [m, e] = v.toExponential(1).split("e");
  return `${m} × 10${[...String(Number(e))].map((ch) => SUP[ch]).join("")}`;
};
const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
const FORMATS = {
  raw: (v) => String(v),
  pval: (v) => pval(v) ?? "–",
  pct1: (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`),
  pp1: (v) => (v == null ? "–" : `${v < 0 ? "−" : "+"}${Math.abs(v * 100).toFixed(1)} pp`),
  2: f2,
  3: f3,
  pct,
  range2: (v) => `${f2(v[0])}–${f2(v[1])}`,
  range3: (v) => `${f3(v[0])}–${f3(v[1])}`,
  rangems: (v) => `${v[0].toFixed(1)}–${v[1].toFixed(1)} ms`,
  ofn: (v) => `${v[0]} of ${v[1]}`,
  pctv: (v) => `${v.toFixed(2)}%`,
  pct2: (v) => `${(v * 100).toFixed(2)}%`,
  rangepct: (v) => `${(v[0] * 100).toFixed(1)}–${(v[1] * 100).toFixed(1)}%`,
};
const range2 = (vals) => FORMATS.range2([Math.min(...vals), Math.max(...vals)]);
const srcText = (s) => `Source: ${s.file}${s.where ? ` (${Object.entries(s.where).map(([k, v]) => `${k} = ${v}`).join(", ")})` : ""}`;

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
let focusTip = null;
const hideTip = () => { tip.hidden = true; focusTip = null; };
// A tooltip opened by keyboard focus follows its element when focus scrolls the page.
const focusTipOn = (node, html) => {
  node.addEventListener("focus", () => { showTip(node, html); focusTip = node; });
  node.addEventListener("blur", hideTip);
};
// Real data points can coincide, so dots are purely visual: one pointer layer over the plot shows the
// nearest point (and any coinciding with it). Keyboard users get the same values from each chart's
// table or control.
const NEAR_PX = 22;
function plotPoints(fr, pts) {
  pts.forEach((p) => p.g.setAttribute("aria-hidden", "true"));
  const layer = el("rect", { class: "plot-hit", x: fr.m.l, y: fr.m.t, width: fr.W - fr.m.l - fr.m.r, height: fr.H - fr.m.t - fr.m.b, fill: "transparent" }, fr.svg);
  let hot = [];
  const nearest = (e) => {
    const r = fr.svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * (fr.W / r.width), py = (e.clientY - r.top) * (fr.H / r.height);
    const d = (p) => Math.hypot(p.x - px, p.y - py);
    const best = pts.reduce((a, p) => (a == null || d(p) < d(a) ? p : a), null);
    return best && d(best) <= NEAR_PX ? pts.filter((p) => Math.hypot(p.x - best.x, p.y - best.y) <= 3) : [];
  };
  const clear = () => { hot.forEach((p) => p.g.classList.remove("hover")); hot = []; layer.style.cursor = ""; hideTip(); };
  const show = (e) => {
    const near = nearest(e);
    if (near.length === hot.length && near.every((p, i) => p === hot[i])) return;
    clear();
    if (!near.length) return;
    hot = near;
    hot.forEach((p) => p.g.classList.add("hover"));
    if (hot.some((p) => p.onActivate)) layer.style.cursor = "pointer";
    showTip(hot[0].g, hot.map((p) => p.html).join('<hr class="tip-sep">'));
  };
  layer.addEventListener("pointermove", show);
  layer.addEventListener("pointerdown", show);
  layer.addEventListener("pointerleave", clear);
  layer.addEventListener("click", (e) => { const p = nearest(e).find((q) => q.onActivate); if (p) p.onActivate(); });
}
addEventListener("scroll", () => {
  if (focusTip && document.activeElement === focusTip) showTip(focusTip, tip.innerHTML);
  else hideTip();
}, { passive: true });

function radios(container, name, options, value, onChange) {
  renderRadios(container, name, options, value);
  bindRadios(container, onChange);
}
// Re-renders options only; the container's change listener is bound once.
function renderRadios(container, name, options, value) {
  container.innerHTML = "";
  for (const o of options) {
    const label = el("label", {}, container);
    const input = el("input", { type: "radio", name, value: String(o.value) }, label);
    input.checked = String(o.value) === String(value);
    el("span", { text: o.label }, label);
    if (o.title) label.title = o.title;
  }
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
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Chart of ${yLabel} against ${xLabel}; the values are listed in the table below it.` }, container);
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
  const xt = el("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", text: xLabel }, svg);
  // Narrow screens: recentre, then shrink, a long axis title rather than clip it at the SVG edge.
  const avail = W - 8, len = xt.getComputedTextLength();
  if (len > W - m.l - m.r) xt.setAttribute("x", W / 2);
  if (len > avail) xt.style.fontSize = `${parseFloat(getComputedStyle(xt).fontSize) * avail / len}px`;
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
// Distinct marker shapes so series never depend on colour alone.
function marker(g, shape, cx, cy, r, color) {
  const s = r * 1.15;
  const paths = {
    square: `M${cx - s * 0.85},${cy - s * 0.85}h${s * 1.7}v${s * 1.7}h${-s * 1.7}Z`,
    triangle: `M${cx},${cy - s * 1.1}L${cx + s},${cy + s * 0.75}L${cx - s},${cy + s * 0.75}Z`,
    diamond: `M${cx},${cy - s * 1.15}L${cx + s},${cy}L${cx},${cy + s * 1.15}L${cx - s},${cy}Z`,
    down: `M${cx},${cy + s * 1.1}L${cx + s},${cy - s * 0.75}L${cx - s},${cy - s * 0.75}Z`,
    star: `${Array.from({ length: 10 }, (_, i) => {
      const a = (Math.PI / 5) * i - Math.PI / 2, rr = i % 2 ? s * 0.6 : s * 1.45;
      return `${i ? "L" : "M"}${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`;
    }).join("")}Z`,
  };
  const attrs = { fill: color, stroke: "#fff", "stroke-width": 1.5, class: "mk" };
  return shape === "circle" ? el("circle", { cx, cy, r, ...attrs }, g) : el("path", { d: paths[shape], ...attrs }, g);
}
function legendItem(label, color, { dash = "", shape } = {}) {
  const m = shape ? `<svg width="16" height="16" aria-hidden="true" viewBox="0 0 16 16">${shape === "circle" ? `<circle cx="8" cy="8" r="5.5" fill="${color}"/>` : `<path fill="${color}" d="${{ square: "M3,3h10v10h-10Z", triangle: "M8,2L14,13L2,13Z", diamond: "M8,1.5L14,8L8,14.5L2,8Z", down: "M8,14L14,3L2,3Z", star: "M8,0.5L9.9,5.6L15.2,5.7L11,9L12.5,14.2L8,11.1L3.5,14.2L5,9L0.8,5.7L6.1,5.6Z" }[shape]}"/>`}</svg>` : "";
  return `<span><svg width="24" height="8" aria-hidden="true"><line x1="0" x2="24" y1="4" y2="4" stroke="${color}" stroke-width="3" stroke-dasharray="${dash}"/></svg>${m}${label}</span>`;
}
function onResize(node, draw) {
  let w = 0;
  let raf = 0;
  // Redraw in the next frame, not inside the observer callback: drawing changes the node's height,
  // which inside the callback triggers the "ResizeObserver loop" error.
  new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => { if (node.clientWidth !== w) { w = node.clientWidth; draw(); } });
  }).observe(node);
}

// ------------------------------------------------------------------ boot
const DATA = window.BEAT_DATA;
const H = { ...DATA.headline, ...DATA.lee2019.headline };
const L_PENDING = DATA.lee2019.status === "pending";
// Replication lines in the hero only appear once the sealed run is exported.
document.querySelectorAll("[data-l]").forEach((n) => { n.hidden = L_PENDING; });
const CHANCE = H.chance.value;

document.querySelectorAll("[data-h]").forEach((node) => {
  const [key, ...rest] = node.dataset.h.split(".");
  const entry = H[key];
  const value = rest.length ? get(entry.value, rest.join(".")) : entry.value;
  node.textContent = (FORMATS[node.dataset.fmt] || String)(value);
  node.title = srcText(entry.source);
});
// K numbers quoted in the Korean introduction, straight from frontier.csv's headline cell.
document.querySelectorAll("[data-kv]").forEach((node) => {
  const [defence, column] = node.dataset.kv.split("|");
  node.textContent = f3(DATA.k_frontier.rows.find((r) => r.defence === defence)[column]);
  node.title = `Source: ${DATA.k_frontier.source.file} (defence = ${defence}, ${column})`;
});
// Finding 3: name the scenario of the strongest attacker behind the 20-command range.
{
  const c20 = DATA.observation.rows.filter((r) => r.experiment === "exp_c" && r.mode === "command" && r.k === 20);
  const cross = c20.filter((r) => r.scenario === "auxiliary");
  $("#cmd-scenario").textContent = `At 20 commands the strongest attacker is within-day in ${c20.length - cross.length} of ${c20.length} settings`
    + (cross.length ? `; it is cross-day (trained on session T) for ${cross.map((r) => `${cond(r.condition).name} at τ = ${r.tau} (${f3(r.ba_mean)})`).join(", ")}.` : ".");
  $("#cmd-scenario").title = "Source: results/exp_c/observation_strongest.csv (mode = command, k = 20, scenario)";
}
$("#seal-hash").textContent = DATA.seal.commit.slice(0, 7);
$("#seal-hash").title = `${DATA.seal.commit} (${DATA.seal.source.file})`;
document.querySelectorAll("[data-placeholder]").forEach((a) => a.addEventListener("click", (e) => e.preventDefault()));

// Evidence badges: README's three tiers; an experiment's tier comes from the report's status table via DATA.experiments.
const TIER = { confirmatory: ["conf", "Confirmatory-sealed"], estimate: ["est", "Pre-registered estimate"], exploratory: ["expl", "Exploratory"] };
const tierBadge = (tier, title) => `<span class="badge ${TIER[tier][0]}"${title ? ` title="${title}"` : ""}>${TIER[tier][1]}</span>`;
const badge = (exp) => {
  const e = DATA.experiments[exp];
  return tierBadge(e.evidence, `Experiment ${e.letter}: ${e.status_text} (${e.outcome_source.file} §${e.outcome_source.where.section})`);
};
document.querySelectorAll("[data-evidence]").forEach((node) => { node.outerHTML = badge(node.dataset.evidence); });

// Study at a glance
// Which sealed replication claims re-test each 9-user experiment (docs/report.md §6.7).
const REPLICATES = { exp_b: ["C1", "C2"], exp_c: ["C3"], exp_g: ["C4"] };
const CLAIM_WORD = { supported: "supported", not_supported: "not supported" };
const SECTION_OF = { exp_a:"#method", exp_b: "#frontier", exp_c: "#replay", exp_d: "#frontier", exp_e: "#fooled", exp_g: "#pitfall2", exp_l: "#replication", exp_k: "#k-frontier" };
{
  const list = $("#glance-list");
  for (const [k, e] of Object.entries(DATA.experiments)) {
    const li = el("li", { class: `glance-item${e.status === "pending" ? " pending" : ""}` }, list);
    el("span", { class: "glance-letter", "aria-hidden": "true", text: e.letter }, li);
    const body = el("div", {}, li);
    el("h3", { text: e.title }, body);
    el("p", { class: "glance-tags", html: `${badge(k)} <span class="status ${e.status}">${e.status === "available" ? "Complete" : "Pending"}</span>` }, body);
    el("p", { class: "glance-q", text: e.question }, body);
    const out = el("p", { class: "glance-out" }, body);
    if (e.outcome) { el("b", { text: "Outcome: " }, out); out.append(e.outcome); out.title = srcText(e.outcome_source); }
    else out.textContent = "Results pending: the sealed final run has not been exported yet.";
    const tested = (REPLICATES[k] || []).map((id) => DATA.lee2019.claims.find((c) => c.id === id));
    if (tested.length && !L_PENDING) {
      const rp = el("p", { class: "glance-repl", html: `${badge("exp_l")} <b>On 54 users:</b> ` }, body);
      rp.append(tested.map((c) => `${c.id} ${CLAIM_WORD[c.result.decision] || c.result.decision}`).join("; "));
      rp.title = srcText(DATA.lee2019.source);
    }
    el("a", { href: SECTION_OF[k], text: k === "exp_a" ? "Details in Method" : "See the evidence", "aria-label": `See the evidence for experiment ${e.letter}` }, body);
  }
  $("#glance-note").textContent = `Questions and outcomes: ${DATA.experiments.exp_a.outcome_source.file} §5; completion is detected from each experiment's result file.`;
}

// ------------------------------------------------------------------ replay
const RM = DATA.replay.meta;
const CTRL = DATA.control.rows;
const OBS = DATA.observation.rows;
const state = { subject: 1, condition: "P2", mode: "command", tau: 0.8, runIndex: 0, seed: String(RM.seeds[0]), step: 0 };
const subjectCache = new Map();
const loadSubject = (s) => {
  // Script injection instead of fetch so the page also works when opened from the file system.
  if (!subjectCache.has(s)) subjectCache.set(s, new Promise((resolve, reject) => {
    const tag = el("script", { src: DATA.replay.files[s] }, document.head);
    tag.onload = () => resolve(window.BEAT_REPLAY[s]);
    tag.onerror = () => reject(new Error(`Could not load ${DATA.replay.files[s]}`));
  }));
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
    el("div", { class: "arow-sum", html: `${done ? `<b>${hits}/${done}</b> right here` : "&nbsp;"}<br>all users: <b>${f2(agg?.ba_mean)}</b> <span title="${ATTACKERS[info.attacker]}, ${dayOf(info.scenario)}">(${ATTACKERS[info.attacker]}, ${dayOf(info.scenario)})</span>` }, row);
  }
  const scen = new Set(RM.k.map((k) => RM.attacker[`${state.condition}|${stream}|${k}`].scenario));
  $("#attack-hint").innerHTML = `True user: <b>S${truth}</b> · attacker sees <b>${state.mode === "feature" ? "feature vectors" : "commands only"}</b>`;
  $("#attack-note").textContent = `Each box is one stored guess from a separate group of k consecutive outputs; green = correct user, red = wrong. Chance is ${f2(CHANCE)}. `
    + `Attackers were trained ${scen.has("main") ? "within-day, on the other half of session E (runs 1–3 ↔ 4–6)" : ""}${scen.size > 1 ? " or " : ""}${scen.has("auxiliary") ? "cross-day, on session T" : ""}, never on the run shown. `
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
    [`Identity score, 1 output (${dayOf(ob(1).scenario)})`, f2(ob(1).ba_mean)],
    [`Identity score, 20 outputs (${dayOf(ob(20).scenario)})`, f2(ob(20).ba_mean)],
  ];
  const dl = $("#replay-stats");
  dl.innerHTML = "";
  for (const [k, v] of items) { const d = el("div", {}, dl); el("dt", { text: k }, d); el("dd", { text: v }, d); }
}
renderReplay(true);
onResize(trajSvg.parentElement, draw);

// ------------------------------------------------------------------ pitfall 1: co-trained adversary vs independent attackers
const ADV = DATA.adversary.rows.filter((r) => r.experiment === "exp_b");
const TRADE = DATA.tradeoff.rows;
const EEG = DATA.eegnet.rows;
const FAMILIES = [["common", "Common reference"], ["rc", "Recentred"]];
const seedsOf = (exp) => DATA.experiments[exp].seeds.n;
const seedMeans = (exp) => `Means over ${seedsOf(exp)} model seeds (${DATA.experiments[exp].seeds.source.file}).`;
const fromD = (kind) => (fam) => TRADE.filter((r) => r.experiment === "exp_d" && r.kind === kind && r.family === fam)
  .map((r) => ({ lam: r.target, head: r.head_main_ba, ind: r.identity_ba, attacker: r.strongest_attacker, scenario: r.strongest_scenario, selected: r.selected, task: r.task_ba }));
const P1 = {
  B: {
    label: "B · gradient reversal", exp: "exp_b", head: "Co-trained adversary head",
    families: [["P3", "Common reference"], ["P3-RC", "Recentred"]],
    rows: (fam) => ADV.filter((r) => r.condition === fam).map((r) => ({
      lam: r.target, head: r.adversary_main_ba, ind: r.independent_ba, attacker: r.strongest_attacker, scenario: r.strongest_scenario, selected: r.selected,
      task: TRADE.find((t) => t.experiment === "exp_b" && t.condition === fam && t.target === r.target)?.task_ba })),
    source: "results/exp_b/adversary_comparison.csv (adversary_main_ba vs independent_ba); task accuracy results/exp_b/tradeoff.csv",
    note: "Head: the identity head trained with the encoder, scored within-day. Independent: the strongest of three retrained attackers (logistic regression, RBF SVM, MLP) over both scenarios: within-day = trained and tested within session E (runs 1–3 ↔ 4–6, both directions averaged); cross-day = trained on session-T outputs, tested on E.",
    takeaway: () => `Judged by its own head the model looks private; judged by an independent attacker it leaks more as λ grows. Cross-validation on session T therefore selected λ = ${H.selected_lambda.value}, effectively no adversary, and pre-registered hypothesis H2 was ${H.h2_accepted.value ? "supported" : "not supported"}.`,
  },
  D_maxent: {
    label: "D · maximum entropy", exp: "exp_d", head: "Co-trained adversary head", families: FAMILIES, rows: fromD("maxent"),
    source: "results/exp_d/tradeoff.csv (head_main_ba vs identity_ba, kind = maxent)",
    note: "The adversary pushes its posterior toward uniform instead of reversing gradients. λ and epoch budget were selected on session T; the selected point is marked. Independent: strongest retrained attacker over within-day and cross-day.",
  },
  D_ensemble: {
    label: "D · ensemble refresh", exp: "exp_d", head: "Co-trained adversary heads", families: FAMILIES, rows: fromD("ensemble"),
    source: "results/exp_d/tradeoff.csv (head_main_ba vs identity_ba, kind = ensemble)",
    note: "Five adversary heads behind gradient reversal, one re-initialised per epoch. λ and epoch budget were selected on session T; the selected point is marked. Independent: strongest retrained attacker over within-day and cross-day.",
  },
  E: {
    label: "E · EEGNet", exp: "exp_e", head: "Co-trained identifier", families: [["none", "No alignment"], ["ea", "Euclidean alignment"]],
    rows: (fam) => EEG.filter((r) => r.alignment === fam).map((r) => ({ lam: r.target, head: r.identifier_ba, ind: r.strongest_ba, attacker: r.strongest_attacker, scenario: r.strongest_scenario, selected: false, task: r.task_ba })),
    source: `${DATA.eegnet.source} (identifier_ba vs strongest_ba)`,
    note: "Every λ is reported and none was selected. The identifier is trained with the encoder on session T and scored on all of session E, a cross-day measure. Independent: strongest retrained attacker over within-day and cross-day on the same frozen EEGNet features.",
    takeaway: () => { const n = DATA.eegnet.near_chance; return `Where the identifier sits close to chance it reads ${FORMATS.range2(n.identifier)}, while independent attackers reach ${FORMATS.range2(n.independent)} on the same features. Unlike B it never falls below chance. Without alignment at high λ left/right accuracy collapses too, so those are not usable operating points.`; },
  },
};
let p1Study = "B", fam = "P3";
const lam = $("#lambda");
radios($("#p1-study"), "p1-study", Object.entries(P1).map(([k, s]) => ({ value: k, label: s.label })), p1Study, (v) => {
  p1Study = v; fam = P1[v].families[0][0];
  renderRadios($("#fooled-family"), "fam", P1[v].families.map(([value, label]) => ({ value, label })), fam);
  setLambdaDefault(); drawFooled();
});
renderRadios($("#fooled-family"), "fam", P1.B.families.map(([value, label]) => ({ value, label })), fam);
bindRadios($("#fooled-family"), (v) => { fam = v; drawFooled(); });
const p1Rows = () => P1[p1Study].rows(fam).sort((a, b) => a.lam - b.lam);
function setLambdaDefault() {
  const rows = p1Rows();
  const i = rows.findIndex((r) => r.lam === 1);
  lam.max = rows.length - 1;
  lam.value = String(i >= 0 ? i : rows.length - 1);
}
setLambdaDefault();
lam.addEventListener("input", drawFooled);
function drawFooled() {
  const study = P1[p1Study];
  const rows = p1Rows();
  const sel = rows[Math.min(Number(lam.value), rows.length - 1)];
  const idx = rows.map((_, i) => i);
  const fr = frame($("#fooled-chart"), {
    xDomain: [-0.5, rows.length - 0.5], yDomain: [0, 1], xTicks: idx, xFmt: (i) => String(rows[i].lam),
    yTicks: [0, 0.25, 0.5, 0.75, 1], xLabel: "adversary strength λ", yLabel: "identity score",
  });
  refLine(fr, { y: CHANCE, label: `chance ${f2(CHANCE)}` });
  const X = (r) => fr.x(rows.indexOf(r));
  el("rect", { x: X(sel) - 14, width: 28, y: fr.m.t, height: fr.H - fr.m.t - fr.m.b, fill: "#f2a900", "fill-opacity": 0.14 }, fr.svg);
  const selected = rows.find((r) => r.selected);
  if (selected) el("text", { class: "ref-label", x: X(selected) + 6, y: fr.H - fr.m.b - 8, text: "selected on T" }, fr.svg);
  const series = [
    { key: "head", label: `${study.head}`, color: PAL.head, dash: "5 4", shape: "circle" },
    { key: "ind", label: "Strongest independent attacker", color: PAL.ind, dash: "", shape: "square" },
  ];
  const pts = [];
  for (const s of series) {
    el("polyline", { points: rows.map((r) => `${X(r)},${fr.y(r[s.key])}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 3, "stroke-dasharray": s.dash }, fr.svg);
    rows.forEach((r, i) => {
      const g = el("g", { class: `pt${r === sel ? " active" : ""}` }, fr.svg);
      marker(g, s.shape, X(r), fr.y(r[s.key]), 6, s.color);
      pts.push({ g, x: X(r), y: fr.y(r[s.key]), onActivate: () => { lam.value = String(i); drawFooled(); },
        html: `<b>${s.label}</b><br>λ = ${r.lam} · identity score ${f3(r[s.key])}${s.key === "ind" ? `<br>${ATTACKERS[r.attacker] || r.attacker}, ${dayOf(r.scenario)}` : ""}` });
    });
  }
  plotPoints(fr, pts);
  // Legend sits outside the plot so it never covers points at narrow widths.
  $("#fooled-legend").innerHTML = series.map((s) => legendItem(s.label, s.color, s)).join("");
  $("#p1-title").textContent = `${study.label}: ${P1[p1Study].families.find((f) => f[0] === fam)[1].toLowerCase()}`;
  $("#p1-badge").innerHTML = badge(study.exp);
  $("#p1-note").textContent = `${seedMeans(study.exp)} ${study.note} Source: ${study.source}.`;
  $("#lambda-out").textContent = `λ = ${sel.lam}${sel.selected ? " (selected on session T)" : ""}`;
  lam.setAttribute("aria-valuetext", `lambda ${sel.lam}`);
  $("#v-head-label").textContent = `${study.head} says`;
  $("#v-head").textContent = f2(sel.head);
  $("#v-head-sub").textContent = `${(sel.head / CHANCE).toFixed(1)}× chance (chance ${f2(CHANCE)})`;
  $("#v-ind").textContent = f2(sel.ind);
  $("#v-ind-sub").textContent = `${ATTACKERS[sel.attacker] || sel.attacker}, ${dayOf(sel.scenario)} · ${(sel.ind / CHANCE).toFixed(1)}× chance`;
  $("#v-task").innerHTML = sel.task == null ? "" : `Left/right accuracy at this λ: <b>${f3(sel.task)}</b> (balanced accuracy)`;
  $("#p1-takeaway").textContent = study.takeaway ? study.takeaway()
    : `Across all λ in this family the heads score ${range2(rows.map((r) => r.head))} while independent attackers reach ${range2(rows.map((r) => r.ind))} on the same outputs.`;
  table($("#fooled-table"), ["λ", study.head, "strongest independent", "strongest attacker", "left/right accuracy"],
    rows.map((r) => [String(r.lam) + (r.selected ? " (selected)" : ""), f3(r.head), f3(r.ind), `${ATTACKERS[r.attacker] || r.attacker}, ${dayOf(r.scenario)}`, f3(r.task)]));
}
onResize($("#fooled-chart"), drawFooled);

// ------------------------------------------------------------------ utility–identity frontier
const FR_SERIES = [
  // Raw features are B's sealed P0 / P0-RC rows (README: 2a within-day 1.00, confirmatory-sealed B).
  { key: "raw", label: "Raw features, 253-d", color: PAL.raw, shape: "square", exp: "exp_b",
    pick: (f) => TRADE.filter((r) => r.experiment === "exp_b" && r.condition === (f === "common" ? "P0" : "P0-RC")) },
  { key: "task", label: "Task-only bottleneck, width d", color: PAL.task, shape: "circle", exp: "exp_d", line: "d",
    pick: (f) => TRADE.filter((r) => r.experiment === "exp_d" && r.family === f && r.kind === "task") },
  { key: "grl", label: "Gradient reversal (B)", color: PAL.grl, shape: "triangle", exp: "exp_b", line: "target",
    pick: (f) => TRADE.filter((r) => r.experiment === "exp_b" && r.condition === (f === "common" ? "P3" : "P3-RC")) },
  { key: "maxent", label: "Maximum entropy (D)", color: PAL.maxent, shape: "diamond", exp: "exp_d", line: "target",
    pick: (f) => TRADE.filter((r) => r.experiment === "exp_d" && r.family === f && r.kind === "maxent") },
  { key: "ensemble", label: "Ensemble refresh (D)", color: PAL.ensemble, shape: "down", exp: "exp_d", line: "target",
    pick: (f) => TRADE.filter((r) => r.experiment === "exp_d" && r.family === f && r.kind === "ensemble") },
];
let frontierFam = "common";
const frontierOn = new Map(FR_SERIES.map((s) => [s.key, true]));
radios($("#frontier-family"), "frontier-family", FAMILIES.map(([value, label]) => ({ value, label })), frontierFam,
  (v) => { frontierFam = v; drawFrontier(); });
chips($("#frontier-toggles"), FR_SERIES.map((s) => ({ key: s.key, label: s.label, color: s.color, on: true })),
  (k, on) => { frontierOn.set(k, on); drawFrontier(); });
$("#frontier-badges").innerHTML = `B: ${badge("exp_b")} D: ${badge("exp_d")}`;
const evidenceOf = (exp) => `${EXP_LETTER[exp]}, ${TIER[DATA.experiments[exp].evidence][1].toLowerCase()}`;
function drawFrontier() {
  const all = FR_SERIES.flatMap((s) => s.pick(frontierFam));
  const lo = Math.floor((Math.min(...all.map((r) => r.task_ba)) - 0.005) * 50) / 50;
  const hi = Math.ceil((Math.max(...all.map((r) => r.task_ba)) + 0.005) * 50) / 50;
  const yt = []; for (let v = lo; v <= hi + 1e-9; v += 0.02) yt.push(Math.round(v * 100) / 100);
  const fr = frame($("#frontier-chart"), {
    xDomain: [0, 1.02], yDomain: [lo, hi], xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: yt, yFmt: (v) => v.toFixed(2),
    xLabel: "strongest attacker's identity score (← more private)", yLabel: "left/right accuracy (BA)",
  });
  refLine(fr, { x: CHANCE, label: "chance" });
  const shown = [], hits = [];
  for (const s of FR_SERIES) {
    if (!frontierOn.get(s.key)) continue;
    const pts = s.pick(frontierFam).slice().sort((a, b) => (s.line ? a[s.line] - b[s.line] : 0));
    if (s.line && pts.length > 1) el("polyline", { points: pts.map((r) => `${fr.x(r.identity_ba)},${fr.y(r.task_ba)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": s.key === "task" ? 2.5 : 1.5, "stroke-opacity": s.key === "task" ? 1 : 0.75 }, fr.svg);
    for (const r of pts) {
      const g = el("g", { class: "pt" }, fr.svg);
      const cx = fr.x(r.identity_ba), cy = fr.y(r.task_ba);
      if (r.selected) el("circle", { cx, cy, r: 10, fill: "none", stroke: "#161a26", "stroke-width": 1.5 }, g);
      marker(g, s.shape, cx, cy, 5.5, s.color);
      if (s.key === "task") el("text", { class: "pt-label", x: cx + 8, y: cy - 7, text: `d=${r.d}` }, g);
      const param = s.key === "task" ? `d = ${r.d}` : s.key === "raw" ? "" : `λ = ${r.target}${r.selected ? " (selected on session T)" : ""}`;
      hits.push({ g, x: cx, y: cy, html: `<b>${s.label}</b> · ${evidenceOf(s.exp)}${param ? `<br>${param}` : ""}<br>accuracy ${f3(r.task_ba)} · identity ${f3(r.identity_ba)}<br>strongest: ${ATTACKERS[r.strongest_attacker] || r.strongest_attacker}, ${dayOf(r.strongest_scenario)}` });
      shown.push([s, r, param]);
    }
  }
  plotPoints(fr, hits);
  $("#frontier-legend").innerHTML = FR_SERIES.filter((s) => frontierOn.get(s.key))
    .map((s) => legendItem(`${s.label} · ${evidenceOf(s.exp)}`, s.color, { shape: s.shape, dash: s.line ? "" : "0 24" })).join("")
    + `<span><svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="none" stroke="#161a26" stroke-width="1.5"/></svg>selected on session T</span>`;
  table($("#frontier-table"), ["model", "evidence", "setting", "left/right accuracy", "identity score", "strongest attacker"],
    shown.map(([s, r, param]) => [s.label, evidenceOf(s.exp), param || "–", f3(r.task_ba), f3(r.identity_ba),
      `${ATTACKERS[r.strongest_attacker] || r.strongest_attacker}, ${dayOf(r.strongest_scenario)}`]));
  // Dominance as defined in docs/report.md §6.2: some task-only bottleneck decodes at least as well and leaks less.
  const task = FR_SERIES[1].pick(frontierFam);
  const adv = FR_SERIES.slice(2).flatMap((s) => s.pick(frontierFam));
  const dominated = adv.filter((p) => task.some((t) => t.task_ba >= p.task_ba && t.identity_ba < p.identity_ba)).length;
  const best = task.reduce((a, b) => (b.identity_ba < a.identity_ba ? b : a));
  $("#frontier-note").textContent = `In this family ${dominated} of ${adv.length} adversarial points are dominated by a task-only bottleneck `
    + `(equal or higher accuracy, lower identity score). The least identifying bottleneck (d = ${best.d}) reaches identity ${f3(best.identity_ba)} at accuracy ${f3(best.task_ba)}: `
    + `the lowest exposure of any defence tested, still ${(best.identity_ba / CHANCE).toFixed(1)}× chance. Values are seed means of the strongest independent attacker `
    + `(D: ${seedsOf("exp_d")} seeds; B: ${seedsOf("exp_b")} seeds). `
    + "Sources: results/exp_d/tradeoff.csv; raw-feature and gradient-reversal points results/exp_b/tradeoff.csv; seeds configs/exp_b.yaml (D inherits them).";
}
onResize($("#frontier-chart"), drawFrontier);

// ------------------------------------------------------------------ observation length
const seriesKey = (r) => `${r.experiment}|${r.condition}${r.d != null ? `|d${r.d}` : ""}`;
const multiExp = (rows) => new Set(rows.map((r) => r.experiment)).size > 1;
const seriesLabel = (key, rows) => {
  const [exp, c, d] = key.split("|");
  return `${multiExp(rows) ? `${EXP_LETTER[exp] || exp}: ` : ""}${cond(c).short}${d ? ` ${d}` : ""}`;
};
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
  const hits = [];
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
      hits.push({ g, x: X(r), y: fr.y(r.ba_mean), html: `<b>${cond(r.condition).name}</b><br>${r.stream === "feature" ? "features" : `commands, τ = ${r.tau}`}, k = ${r.k}<br>identity ${f3(r.ba_mean)} [${f3(r.ba_low)}–${f3(r.ba_high)}]<br>${ATTACKERS[r.attacker] || r.attacker}, ${dayOf(r.scenario)}` });
    }
  });
  plotPoints(fr, hits);
  // One row per point so the table carries everything the pointer tooltip shows.
  table($("#obs-table"), ["model", "stream", "k", "identity score", "interval", "strongest attacker"],
    [...groups].flatMap(([key, pts]) => pts.map((r) => [seriesLabel(key, OBS), r.stream === "feature" ? "features" : `commands, τ = ${r.tau}`,
      String(r.k), f3(r.ba_mean), r.ba_low == null ? "–" : `${f3(r.ba_low)}–${f3(r.ba_high)}`,
      `${ATTACKERS[r.attacker] || r.attacker}, ${dayOf(r.scenario)}`])));
}
onResize($("#obs-chart"), drawObs);

// ------------------------------------------------------------------ pitfall 2: enrolment
const P2 = DATA.pitfall2;
const ENROL = P2.enrolment.rows, SCEN = P2.scenarios.rows, REPRO = P2.reproduction.rows;
const N_ORDER = ["1", "2", "5", "10", "20", "all"].filter((n) => ENROL.some((r) => r.enrolment_n === n));
const P2_KS = [...new Set(ENROL.map((r) => r.k))].sort((a, b) => a - b);
let p2Attacker = "beat_logistic", p2K = P2_KS[0];
{
  const sel = $("#p2-attacker");
  for (const a of [...new Set(ENROL.map((r) => r.attacker))]) el("option", { value: a, text: ATTACKERS[a] || a }, sel);
  sel.value = p2Attacker;
  sel.addEventListener("change", () => { p2Attacker = sel.value; drawP2(); });
  radios($("#p2-k"), "p2-k", P2_KS.map((k) => ({ value: k, label: k === 1 ? "1 trial" : `${k} trials` })), p2K, (v) => { p2K = Number(v); drawP2(); });
}
const pctf = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const qfmt = (v, fmt) => (v == null ? "–" : fmt === "ba3" ? f3(v) : fmt === "pct1" ? pctf(v, 1) : fmt === "pct2" ? pctf(v, 2)
  : fmt === "pp1" ? `${v < 0 ? "−" : "+"}${Math.abs(v * 100).toFixed(1)} pp` : String(v));
function drawP2() {
  const pick = (c) => N_ORDER.map((n) => ENROL.find((r) => r.condition === c && r.attacker === p2Attacker && r.enrolment_n === n && r.k === p2K));
  const fr = frame($("#p2-chart"), {
    xDomain: [-0.5, N_ORDER.length - 0.5], yDomain: [0, 1], xTicks: N_ORDER.map((_, i) => i), xFmt: (i) => N_ORDER[i],
    yTicks: [0, 0.25, 0.5, 0.75, 1], yFmt: (v) => `${v * 100}%`, xLabel: "labelled deployment trials per user, n", yLabel: "identification accuracy",
  });
  refLine(fr, { y: CHANCE, label: `chance ${pctf(CHANCE)}` });
  const ref = SCEN.find((r) => r.condition === "defended" && r.scenario === "reference_session" && r.attacker === p2Attacker && r.k === p2K);
  if (ref) {
    el("line", { x1: fr.m.l, x2: fr.W - fr.m.r, y1: fr.y(ref.accuracy), y2: fr.y(ref.accuracy), stroke: PAL.head, "stroke-dasharray": "2 4", "stroke-width": 2 }, fr.svg);
    el("text", { class: "ref-label", x: fr.m.l + 6, y: fr.y(ref.accuracy) - 6, text: `reference session (cross-day), defended: ${pctf(ref.accuracy)}` }, fr.svg);
  }
  const series = [
    { c: "defended", label: "Defended output", color: PAL.head, dash: "", shape: "circle" },
    { c: "undefended", label: "Undefended", color: PAL.raw, dash: "5 4", shape: "square" },
  ];
  const ne = new Set(), pts = [];
  for (const s of series) {
    const rows = pick(s.c);
    // Break the line where a point is not evaluable instead of bridging the gap.
    let seg = [];
    const flush = () => { if (seg.length > 1) el("polyline", { points: seg.join(" "), fill: "none", stroke: s.color, "stroke-width": 3, "stroke-dasharray": s.dash }, fr.svg); seg = []; };
    rows.forEach((r, i) => { if (r && r.accuracy != null) seg.push(`${fr.x(i)},${fr.y(r.accuracy)}`); else { flush(); ne.add(i); } });
    flush();
    rows.forEach((r, i) => {
      if (!r || r.accuracy == null) return;
      const g = el("g", { class: "pt" }, fr.svg);
      marker(g, s.shape, fr.x(i), fr.y(r.accuracy), 6, s.color);
      pts.push({ g, x: fr.x(i), y: fr.y(r.accuracy), html: `<b>${s.label}</b><br>${ATTACKERS[p2Attacker] || p2Attacker}, n = ${N_ORDER[i]}, ${p2K} test trial${p2K > 1 ? "s" : ""}<br>identification ${pctf(r.accuracy, 2)}` });
    });
  }
  plotPoints(fr, pts);
  for (const i of ne) el("text", { class: "ref-label", x: fr.x(i), y: fr.H - fr.m.b - 8, "text-anchor": "middle", text: "n/e" }, fr.svg);
  $("#p2-legend").innerHTML = series.map((s) => legendItem(s.label, s.color, s)).join("")
    + (ref ? legendItem("Reference-session attack (cross-day), defended", PAL.head, { dash: "2 4" }) : "");
  const def = pick("defended"), und = pick("undefended");
  table($("#p2-table"), ["attacker", "test trials", "n per user", "defended", "undefended", "status"],
    N_ORDER.map((n, i) => [ATTACKERS[p2Attacker] || p2Attacker, String(p2K), n, pctf(def[i]?.accuracy, 2), pctf(und[i]?.accuracy, 2), def[i]?.status === "evaluable" ? "evaluable" : `not evaluable: ${def[i]?.reason || "missing"}`]));
  $("#p2-note").textContent = "Within-day: the attacker trains on the first n labelled defended trials per user from one half of the deployment session and is tested on the other half; "
    + `both directions averaged, chance ${pctf(CHANCE)}. n/e: not evaluable, n is below the attacker's inner cross-validation folds. `
    + `Sources: ${P2.enrolment.source}; reference-session line ${P2.scenarios.source}.`;
}
drawP2();
onResize($("#p2-chart"), drawP2);
{
  const box = $("#p2-repro");
  const dirLabel = { S1_to_S2: "session 1 → 2", S2_to_S1: "session 2 → 1" };
  for (const r of REPRO.filter((x) => x.comparable)) {
    const row = el("div", { class: "repro-row", title: `Source: ${P2.reproduction.source}` }, box);
    el("span", { class: "repro-label", text: `${r.condition === "defended" ? "Defended" : "Undefended"}, ${dirLabel[r.direction] || r.direction}` }, row);
    el("span", { class: "repro-vals", html: `paper <b>${r.paper_accuracy_percent.toFixed(1)}%</b> · ours <b>${r.observed_accuracy_percent.toFixed(2)}%</b>` }, row);
  }
  const other = REPRO.filter((x) => !x.comparable && x.observed_accuracy_percent != null).length;
  el("p", { class: "hint", text: `${other} other attacker rows are not compared: the paper's settings for them are underspecified or tuned on the evaluation session (${P2.reproduction.source}).` }, box);
  const dl = $("#p2-stats");
  const stat = (label, key, fmt) => { const d = el("div", { title: srcText(H[key].source) }, dl); el("dt", { text: label }, d); el("dd", { text: FORMATS[fmt](H[key].value) }, d); };
  stat("Reference-session attackers (cross-day), defended data (1 trial)", "g_reference_defended", "rangepct");
  stat("Deployment-period attackers (within-day), defended data (1 trial)", "g_post_defended", "rangepct");
  stat("1 labelled trial per user (within-day), logistic regression", "g_enrol_n1", "pct2");
  stat("…same attacker on undefended data", "g_enrol_n1_undefended", "pct2");
  stat("5 labelled trials per user (within-day), logistic regression", "g_enrol_n5", "pct2");
  const attackers = [...new Set(SCEN.map((r) => r.attacker))];
  const v = (a, c, s) => pctf(SCEN.find((r) => r.attacker === a && r.condition === c && r.scenario === s && r.k === 1)?.accuracy);
  table($("#p2-scen"), ["attacker", "reference (cross-day), undefended", "reference (cross-day), defended", "deployment (within-day), undefended", "deployment (within-day), defended"],
    attackers.map((a) => [ATTACKERS[a] || a, v(a, "undefended", "reference_session"), v(a, "defended", "reference_session"), v(a, "undefended", "post_deployment"), v(a, "defended", "post_deployment")]));
  el("p", { class: "hint", text: `Single test trial (k = 1), both directions averaged. Source: ${P2.scenarios.source}.` }, $("#p2-scen"));
}

// ------------------------------------------------------------------ replication (experiment L)
{
  const L = DATA.lee2019;
  const pending = L.status === "pending";
  $("#repl-lede").textContent = `The four claims below were derived from the 2a results above and sealed before any Lee2019 outcome was analysed `
    + `(seal ${L.seal_commit.slice(0, 7)}). ${L.users} participants recorded on two days; identity chance is 1 in ${L.users} (${pctf(L.chance, 2)}). `
    + `Holm's procedure at α = ${L.alpha} decides all four claims together.`;
  $("#repl-status").innerHTML = pending
    ? `<p class="pending-box"><b>Results pending.</b> The sealed final run has not been exported yet. This section fills in when <code>scripts/export_demo_data.py</code> is re-run after <code>results/exp_l/decision.json</code> exists; no page change is needed.</p>`
    : `<p class="done-box"><b>Results available.</b> Decisions from <code>${L.source.file}</code>, reported whether or not they support the claims.</p>`;
  const DECISION = {
    supported: ["Supported", "ok"], not_supported: ["Not supported", "bad"],
    not_evaluable_missing_cells: ["Not evaluable: missing cells", "na"], not_evaluable_benchmark_gate: ["Not evaluable: benchmark gate failed", "na"],
    not_evaluable_computational_failure: ["Not evaluable: computational failure", "na"],
  };
  const list = $("#repl-claims");
  for (const c of L.claims) {
    const li = el("li", { class: "claim" }, list);
    const head = el("div", { class: "claim-head" }, li);
    el("span", { class: "claim-id", text: c.id }, head);
    const r = c.result;
    const [label, cls] = r ? DECISION[r.decision] || [r.decision, "na"] : ["Pending", "pending"];
    el("span", { class: `decision ${cls}`, text: label }, head);
    const D = L.disclosures;
    if (c.id === "C4" && D) el("span", { class: "decision na", text: "Adapted replication" }, head);
    el("p", { text: c.statement }, li);
    if (c.id === "C4" && D)
      el("p", { class: "claim-note", text: `Adapted replication. ${D.c4_caveat.reproduction_caveat}. ${D.c4_caveat.text} `
        + `Sources: ${D.c4_caveat.source.map((s) => s.file).join(", ")}.` }, li);
    if ((c.id === "C1" || c.id === "C2") && D && D.nonconvergence.strongest_c1_c2)
      el("p", { class: "claim-note", text: `The strongest attacker here is an MLP whose fits did not converge within ${D.nonconvergence.budget} iterations; see the convergence disclosure below.` }, li);
    if (r) {
      const dl = el("dl", { class: "claim-stats" }, li);
      const add = (k, v) => { if (v == null) return; const d = el("div", {}, dl); el("dt", { text: k }, d); el("dd", { text: v }, d); };
      add("Observed difference", r.observed_difference != null ? `${(r.observed_difference * 100).toFixed(1)} pp` : null);
      add("Effect requirement met", r.effect_requirement == null ? null : r.effect_requirement ? "yes" : "no");
      add("Benchmark gate", r.benchmark_gate == null ? null : r.benchmark_gate ? "passed" : "failed");
      add("Reproduced benchmark", r.reproduction_accuracy != null ? pctf(r.reproduction_accuracy, 2) : null);
      add("Claim p", pval(r.p));
      add("Holm-adjusted p", pval(r.holm_p));
    }
    if (c.quantities) {
      const tbl = el("table", { class: "claim-q" }, li);
      el("caption", { text: `${c.id}: tested quantities` }, tbl);
      el("thead", { html: `<tr><th scope="col">Quantity</th><th scope="col">Value</th><th scope="col">Interval</th></tr>` }, tbl);
      const tb = el("tbody", {}, tbl);
      for (const q of c.quantities) {
        const tr = el("tr", { title: srcText(q.source) }, tb);
        const th = el("th", { scope: "row", text: q.label }, tr);
        if (q.note) el("span", { class: "q-note", text: q.note }, th);
        el("td", { class: "num", text: qfmt(q.value, q.fmt) }, tr);
        const ci = !q.ci ? null : q.ci[1] == null ? `≥ ${qfmt(q.ci[0], q.fmt)} (${q.ci_kind})` : `${qfmt(q.ci[0], q.fmt)} to ${qfmt(q.ci[1], q.fmt)} (${q.ci_kind})`;
        el("td", { text: [ci, q.seeds ? `${q.seeds} seeds` : null].filter(Boolean).join("; ") }, tr);
      }
      if (c.id === "C3" && r && r.decision === "not_supported")
        el("p", { class: "claim-note", text: `The k = 20 mean is above twice chance, but the claim is not supported after Holm's correction (adjusted p = ${pval(r.holm_p)}).` }, li);
      el("p", { class: "hint", text: `Sources: ${[...new Set(c.quantities.map((q) => q.source.file))].join(", ")}. Hover a row for its exact cell.` }, li);
    }
  }
  if (L.disclosures) {
    const n = L.disclosures.nonconvergence;
    const box = el("div", { class: "pending-box", id: "repl-convergence" });
    list.after(box);
    el("p", { html: `<b>MLP attacker non-convergence.</b> ${n.failed} of ${n.total} MLP attacker fits (${n.confirmatory_failed} of ${n.confirmatory_total} that enter confirmatory tests) `
      + `reached the fixed ${n.budget}-iteration budget without converging, including the strongest attackers for C1 and C2. `
      + `Their fixed-budget predictions are retained under the sealed rule <code>${n.rule}</code>; identity values from these attackers may differ from fully converged ones in either direction.` }, box);
    el("p", { class: "hint", text: `Sources: ${n.source.map((s) => s.file).join(", ")}.` }, box);
  }
  $("#repl-note").textContent = `Claim definitions: ${L.definition_source.map((s) => s.file).join(", ")}. `
    + (pending ? "Status: pending; nothing from the replication is shown until its decision file exists." : `Results: ${L.source.file}.`);
}

// ------------------------------------------------------------------ audit checklist
{
  const LINKS = [[/^Independent|^Sanity/, "#fooled", "Pitfall 1"], [/Deployment-period|scenario/i, "#pitfall2", "Pitfall 2"],
    [/baseline|Strongest|Utility/i, "#frontier", "Frontier"], [/Observation length/i, "#commands", "Commands"],
    [/Chance and population/i, "#limits-title", "Limits"], [/Pre-register/i, "#timeline-title", "Timeline"]];
  const list = $("#checklist-list");
  const items = DATA.checklist.items;
  const count = () => { $("#check-count").textContent = `${list.querySelectorAll("input:checked").length} of ${items.length} covered`; };
  for (const it of items) {
    const li = el("li", { class: "check-item" }, list);
    const id = `chk-${it.n}`;
    const input = el("input", { type: "checkbox", id }, li);
    input.addEventListener("change", count);
    const body = el("div", {}, li);
    el("label", { for: id, html: `<span class="check-n">${it.n}.</span> ${it.title}` }, body);
    el("p", { text: it.text }, body);
    const link = LINKS.find(([re]) => re.test(it.title));
    if (link) el("a", { href: link[1], text: `Evidence: ${link[2]}`, "aria-label": `Evidence for checklist item ${it.n}: ${link[2]}` }, body);
  }
  count();
  $("#checklist-source").textContent = `Source: ${DATA.checklist.source.file} §${DATA.checklist.source.where.section}.`;
}

// ------------------------------------------------------------------ study M: sealed cross-day replication
// Verdict wording and fixed claim phrases are the pre-committed text of docs/decision_log.md (7b14cee).
const M = DATA.m_study;
const MC = M.claims;
const MIDS = Object.keys(MC);
const M_PHRASE = {
  M1: "원 특징의 다른 날 식별(사전등록상 저위험 재현)",
  M2: "독립 공격자와 공동 학습 head의 격차가 0보다 크고 관측값이 0.10 이상",
  M3: "본인의 순위가 우연보다 높고 명령 1개일 때보다 높다",
};
const mVerdict = (c) => (MC[c].supported ? "지지" : "지지 안 됨");
// Holm p as in the write-ups: three decimals, scientific below 0.001.
const mp = (v) => {
  if (v === 1) return "1";
  if (v >= 1e-3) return v.toFixed(3);
  const [m, e] = v.toExponential(1).split("e");
  return `${m}×10${[...String(Number(e))].map((ch) => SUP[ch]).join("")}`;
};
const ms3 = (v) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(3)}`;
const mci = ([m, lo, hi]) => `${ms3(m)} [${ms3(lo)}, ${ms3(hi)}]`;
const M_TIMELINE = (() => {
  const gaps = MC.M2.gaps;
  return MIDS.map((c) => `${c} <span lang="ko">${mVerdict(c)}</span> (${c === "M2"
    ? `gaps ${f3(Math.min(...gaps))}–${f3(Math.max(...gaps))} below the ${MC.M2.margin.toFixed(2)} floor; p = 1 by rule`
    : `Holm p ${mp(MC[c].p_holm)}`})`).join(" · ");
})();
{
  const nSup = MIDS.filter((c) => MC[c].supported).length, nAll = MIDS.length;
  const title = `새 사용자 ${M.users}명, 다른 날: 사전등록한 주장 ${nAll}개${nSup === nAll ? "가 모두 지지됐다"
    : nSup === 0 ? "가 모두 지지되지 않았다" : ` 중 ${nSup}개가 지지됐다`}`;
  const sup = MIDS.filter((c) => MC[c].supported), rest = MIDS.filter((c) => !MC[c].supported);
  const closing = sup.length && rest.length ? ` 다른 날에는 ${sup.join("·")}만 재현됐다. ${rest.join("·")}는 같은 날 조건의 결과로 남는다.` : "";
  const verdicts = `${MIDS.map((c) => `${c} ${M_PHRASE[c]}: ${mVerdict(c)}`).join("; ")}.${closing}`;
  document.querySelectorAll("[data-m=title]").forEach((n) => { n.textContent = `${title}.`; n.title = `Source: ${M.source.file}`; });
  document.querySelectorAll("[data-m=verdicts]").forEach((n) => { n.textContent = verdicts; });
  const reportM = `../${M.report_source.file}#${M.report_anchor}`;
  document.querySelectorAll("[data-m=report-link]").forEach((n) => { n.href = reportM; });
  $("#m-title").textContent = title;
  $("#m-lede").textContent = `보정일(1일차)에 등록한 공격자를 배포일(2·3일차)에 시험한 다른 날(cross-day) 조건이다. `
    + `판정은 봉인 4에서 고정한 규칙(Holm, α = ${M.alpha})으로 내렸고, 독립 재계산 감사를 통과했다.${closing}`;

  const A = { logistic: "로지스틱", rbf_svm: "RBF SVM", mlp: "MLP" };
  const m1 = MC.M1, m2 = MC.M2, m3 = MC.M3;
  const [best, ...others] = Object.entries(m1.attackers).sort((a, b) => b[1] - a[1]);
  const observed = {
    M1: `최강 공격자 평균 재현율 ${f3(best[1])} (${A[best[0]]}; ${others.map(([a, v]) => `${A[a]} ${f3(v)}`).join(", ")}), 기준 ${m1.threshold_factor}/${M.users} = ${f3(m1.threshold)}`,
    M2: `격차 ${f3(Math.min(...m2.gaps))}–${f3(Math.max(...m2.gaps))} (독립 공격자 ${f3(Math.min(...m2.battery))}–${f3(Math.max(...m2.battery))}, head ${f3(m2.head)}), 봉인된 하한 ${m2.margin.toFixed(2)} 미만이라 검정하지 않음`,
    M3: `명령 ${m3.k_star}개 백분위 순위 ${mci(m3.rank_k_star)}, 명령 ${m3.k_base}개 ${mci(m3.rank_k_base)} (τ* = ${m3.tau.toFixed(2)}, ${m3.attacker})`,
  };
  table($("#m-table"), ["주장", "사전등록한 기준", "관측값 (다른 날)", "Holm p", "판정"],
    MIDS.map((c) => [c, M_PHRASE[c], observed[c], c === "M2" ? "1 (규칙상)" : mp(MC[c].p_holm), mVerdict(c)]));
  $("#m-table").querySelectorAll("tbody tr").forEach((tr, i) => tr.lastChild.classList.add(MC[MIDS[i]].supported ? "m-yes" : "m-no"));

  // EQ: the reading rule over every pre-registered cell; no single cell is "the" EQ result.
  const E = M.eq, [tStar, tSec] = E.taus, kLo = Math.min(...E.ks), kHi = Math.max(...E.ks), nA = E.attackers.length;
  const cells = (tau, k) => E.grid.filter((c) => c.tau === tau && (k == null || c.k === k));
  const incl0 = ([, lo, hi]) => lo <= 0 && 0 <= hi;
  const same = cells(tStar, kHi).filter((c) => c.reading === "same_direction").length;
  const red0 = cells(tStar, kLo).filter((c) => incl0(c.reduction)).length;
  const sec = cells(tSec), secNo = sec.filter((c) => c.reading === "no_exposure").length;
  $("#m-eq").textContent = `EQ·RR은 사전등록한 추정치, 판정 없음. EQ 판독 규칙을 사전등록한 격자 전체(g = ${E.g}, Bayes 공격자 ${nA}종 × 명령 ${E.ks.join("·")}개 × τ = ${tStar.toFixed(2)}(τ*)·${tSec})에 적용하면, `
    + `τ*·명령 ${kHi}개에서는 공격자 ${nA}종 중 ${same}종이 'K와 같은 방향'이었다. `
    + `τ*·명령 ${kLo}개에서는 ${red0 === nA ? "모든 공격자에서" : `공격자 ${nA}종 중 ${red0}종에서`} EQ의 감소가 0과 구별되지 않았다. `
    + `τ = ${tSec}에서는 ${secNo === sec.length ? `${sec.length}칸 모두` : `${sec.length}칸 중 ${secNo}칸에서`} '막을 노출이 구별되지 않아 EQ 효과를 말할 수 없다'.`;
  const READ = { same_direction: "K와 같은 방향", no_reduction: "막을 노출은 구별됐지만 EQ의 감소는 0과 구별되지 않았다", no_exposure: "막을 노출이 구별되지 않아 EQ 효과를 말할 수 없다" };
  table($("#m-grid"), ["τ", "명령 수", "공격자", `방어 없음 그룹 내 초과`, "EQ 그룹 내 초과", "감소", "판독"],
    E.grid.map((c) => [c.tau === tStar ? `${tStar.toFixed(2)} (τ*)` : String(c.tau), String(c.k), c.attacker,
      mci(c.undefended), mci(c.defended), mci(c.reduction), READ[c.reading]]));

  $("#m-note").innerHTML = `표와 격자의 값은 다른 날(보정일 등록 → 배포일 시험) 값이며, 구간은 평균 [95% 구간]이다. 그룹 내 초과는 g = ${E.g}이다. 'K와 같은 방향'에는 `
    + "배포일끼리(2·3일차 사이) 또는 같은 날 절반 사이에서 EQ 자체의 초과가 0과 구별된다는 조건도 붙는다. "
    + `출처: 판정·p·M1–M3 관측값 <code>${M.source.file}</code>; 사용자 수·기준 10/N·하한 <code>${M.config_source.file}</code>; `
    + `top-1 <code>${M.top1_source.file}</code>; EQ 격자 <code>${M.eq_source.file}</code>; 보정일 격차 <code>${M.gate_source.file}</code>. `
    + `<a href="../${M.source.file}">${M.source.file}</a> · <a href="${reportM}">기술 보고서 §${M.report_source.where.section}</a>`;

  const caveats = [
    `M1은 로지스틱(${f3(m1.attackers.logistic)})과 RBF SVM(${f3(m1.attackers.rbf_svm)}) 공격자에 기대고 있다. 독립 감사에 따르면 MLP(${f3(m1.attackers.mlp)}) 공격자 하나만으로는 ×3 Holm 보정 뒤 통과하지 못한다.`,
    `M3의 순위는 식별률이 아니다. 명령 ${m3.k_star}개에서 본인을 1순위로 맞힌 비율(top-1, 사전등록한 추정치)은 ${mci(m3.top1_k_star)}이고, 우연 수준은 1/${M.users} = ${f3(M.chance)}이다.`,
    `M2의 '지지 안 됨'은 '격차 없음'이 아니다. 다른 날 격차 ${m2.gaps.length}개는 모두 양수(${f3(Math.min(...m2.gaps))}–${f3(Math.max(...m2.gaps))})였지만 봉인된 하한 ${m2.margin.toFixed(2)}에 못 미쳐 규칙상 검정하지 않았다(p = 1). 보정일 같은 날 격차: ${f3(m2.day1_within_day_gap)}.`,
  ];
  const ul = $("#m-caveats");
  caveats.forEach((t) => el("li", { text: t }, ul));
}

// ------------------------------------------------------------------ pre-registration timeline
{
  const list = $("#timeline");
  for (const ev of DATA.timeline.events) {
    const done = ev.kind !== "pending";
    const li = el("li", { class: `tl tl-${ev.kind}` }, list);
    el("span", { class: "tl-dot", "aria-hidden": "true" }, li);
    el("p", { class: "tl-label", text: ev.label }, li);
    el("h4", { text: ev.title }, li);
    el("p", { class: "tl-meta", html: `${tierBadge(ev.evidence)} ${ev.experiments.map((e) => EXP_LETTER[e]).join(", ")}${done ? "" : " · <span class=\"status pending\">Pending</span>"}` }, li);
    el("p", { text: ev.text }, li);
    if (ev.result === "exp_m") el("p", { class: "tl-result", html: M_TIMELINE }, li);
    if (ev.commit) {
      const p = el("p", { class: "tl-commit" }, li);
      const code = el("code", { text: ev.commit.slice(0, 7), title: ev.commit }, p);
      code.setAttribute("aria-label", `commit ${ev.commit}`);
      if (ev.tests) p.append(` · ${ev.tests} tests passed`);
      if (ev.date) p.append(` · ${ev.date}`);
    }
  }
  $("#timeline-source").textContent = `Commits, dates and test counts: ${DATA.timeline.source.file}.`;
}

// ------------------------------------------------------------------ K: randomized response vs group-equalized decoding
{
  const K = DATA.k_frontier, S = K.setting;
  const KR = Object.fromEntries(K.rows.map((r) => [r.defence, r]));
  const DAY = { within_day: "같은 날 (within-day)", cross_day: "다른 날 (cross-day)" };
  const GCOL = { 2: PAL.grl, 3: PAL.head, 5: PAL.task };
  const INK = "#161a26", GREY = "#4a5163";
  const EQ = [["eqc", "EQ-confusion", "down", "그룹 내 P(출력 | 목표) 균등화"], ["eqm", "EQ-marginal", "triangle", "그룹 내 P(출력) 균등화"]];
  const PATH = ["none", ...K.rr];
  const qOf = (d) => d.slice(4);
  const gOf = (d) => d.split("_g")[1];
  const name = (d) => (d === "none" ? "방어 없음" : d.startsWith("rr_") ? `무작위 응답 RR, q = ${qOf(d)}`
    : `${d.startsWith("eqc") ? "EQ-confusion" : "EQ-marginal"}, g = ${gOf(d) === "all" ? "전체" : gOf(d)}`);
  const col = (metric, day, g) => (metric === "excess" ? `g${g}_excess_${day}_k${S.k}` : `nway_ba_${day}_k${S.k}`);
  // Signed values with a true minus sign; utilities and excesses can be negative.
  const m3 = (v) => (v == null ? "–" : `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(3)}`);
  const arrow = (a, b) => `${m3(a)} → ${m3(b)}`;
  const otherFirst = (g) => [...K.groups.filter((x) => x !== g), g];
  // make_groups: equal groups of g, leftover users join the last group.
  const sizeText = (g) => {
    const s = K.group_sizes[g], last = s.at(-1);
    return last === g ? `g = ${g}: 정확히 ${g}명씩 ${s.length}개 그룹` : `g = ${g}: ${g}명 그룹 ${s.length - 1}개와 ${last}명인 마지막 그룹`;
  };
  let kMetric = "excess", kG = S.g;
  const on = new Map([["rr", true], ["eqc", true], ["eqm", true]]);

  const eqDefs = EQ.flatMap(([v]) => K.groups.map((g) => `${v}_g${g}`));
  const util = [...PATH, ...eqDefs].map((d) => KR[d].weighted_utility);
  const X_STEP = 0.2;
  const xI = [Math.floor((Math.min(...util) - 0.04) / X_STEP), Math.ceil((Math.max(...util) + 0.04) / X_STEP)];
  const xTicks = []; for (let i = xI[0]; i <= xI[1]; i += 1) xTicks.push(Math.round(i * X_STEP * 10) / 10);
  // One y-range per metric for both days and every g, so switching g never rescales the axes.
  const yRange = (metric) => {
    const vals = K.days.flatMap((day) => (metric === "excess"
      ? K.groups.flatMap((g) => [...PATH, ...EQ.map(([v]) => `${v}_g${g}`)].map((d) => KR[d][col(metric, day, g)]))
      : [...PATH, ...eqDefs].map((d) => KR[d][col(metric, day)])));
    const lo = Math.min(0, Math.floor(Math.min(...vals) / 0.05) * 0.05), hi = Math.ceil((Math.max(...vals) * 1.1) / 0.05) * 0.05;
    const step = hi - lo > 0.26 ? 0.1 : 0.05;
    const ticks = []; for (let i = Math.ceil(lo / step - 1e-9); i * step <= hi + 1e-9; i += 1) ticks.push(Math.round(i * step * 100) / 100);
    return { domain: [lo, hi], ticks };
  };
  const tipHtml = (d, day, g) => {
    const r = KR[d], gg = d.includes("_g") ? Number(gOf(d)) : g;
    return `<b>${name(d)}</b><br>${DAY[day]}, 명령 ${S.k}개<br>가중 유용성 ${m3(r.weighted_utility)} · 정지 비율 ${f3(r.hold_rate)}<br>`
      + (kMetric === "excess" ? `그룹 내 초과 (g = ${gg}): ${m3(r[col("excess", day, gg)])}` : `전체 ${K.users}명 중 식별: ${f3(r[col("nway", day)])}`);
  };

  radios($("#k-metric"), "k-metric", [{ value: "excess", label: "그룹 내 초과 식별" }, { value: "nway", label: `전체 ${K.users}명 중 식별` }], kMetric,
    (v) => { kMetric = v; drawK(); });
  radios($("#k-g"), "k-g", K.groups.map((g) => ({ value: g, label: g === S.g ? `g = ${g} (g*)` : `g = ${g}`, title: g === S.g ? "선언된 대표 설정 g*" : null })), kG,
    (v) => { kG = Number(v); drawK(); });
  chips($("#k-toggles"), [{ key: "rr", label: "무작위 응답 RR", color: GREY, on: true }, ...EQ.map(([v, label]) => ({ key: v, label, color: GREY, on: true }))],
    (key, value) => { on.set(key, value); drawK(); });
  $("#k-badge").innerHTML = `${badge("exp_k")} 탐색적: g = ${S.g} 강조`;
  $("#k-lede").textContent = `Lee2019 ${K.users}명, 재중심화 병목 P2-RC (d = 8), τ = ${S.tau}, 명령 ${S.k}개(k = ${S.k}). `
    + "두 방어 모두 해독된 왼쪽·오른쪽·정지 명령을 전송 전에 바꾼다. 무작위 응답 RR(q)는 각 명령을 확률 q로 무작위 명령으로 바꾼다. "
    + "EQ는 사용자를 보정일 유용성 순으로 g명씩 묶고(남는 사용자는 마지막 그룹에 들어간다), 보정일 데이터로 사용자별 확률 행렬을 적합해 "
    + "그룹 안의 모든 사용자가 같은 출력 분포를 갖게 한다. "
    + "오른쪽일수록 조작에 유리하고, 아래일수록 신원이 덜 드러난다. "
    + `g = ${S.g}은 K가 미리 선언한 대표 설정(g*)이라 굵게 표시했고, 다른 g는 흐리게 그렸다. 모든 값은 탐색적이다.`;

  function drawK() {
    const yr = yRange(kMetric);
    for (const day of K.days) {
      const fr = frame($(`#k-chart-${day}`), {
        xDomain: [xTicks[0], xTicks.at(-1)], yDomain: yr.domain, xTicks, yTicks: yr.ticks, xFmt: (v) => m3(v).slice(0, -2), yFmt: (v) => m3(v).slice(0, -1),
        xLabel: `가중 유용성 P(정답) − ${S.wrong_weight}·P(오답)`, yLabel: kMetric === "excess" ? "그룹 내 초과 식별" : `${K.users}명 중 식별 (BA)`,
      });
      // Chance label under the line at the left: the undefended and EQ points crowd the right end.
      const yRef = fr.y(kMetric === "excess" ? 0 : K.chance);
      el("line", { class: "ref-line", x1: fr.m.l, x2: fr.W - fr.m.r, y1: yRef, y2: yRef }, fr.svg);
      el("text", { class: "ref-label", x: fr.m.l + 4, y: yRef + 13, text: kMetric === "excess" ? "그룹 내 우연 수준 (초과 0)" : `우연 수준 1/${K.users}` }, fr.svg);
      const pts = [];
      const point = (layer, d, shape, x, y, r, color, g) => {
        const gg = el("g", { class: "pt" }, layer);
        marker(gg, shape, x, y, r, color);
        pts.push({ g: gg, x, y, html: tipHtml(d, day, g) });
      };
      // Undefended and RR: one line per g for the within-group excess, a single black line for N-way.
      for (const g of kMetric === "excess" ? otherFirst(kG) : [null]) {
        const hot = g == null || g === kG, color = g == null ? INK : GCOL[g];
        const layer = el("g", { opacity: hot ? 1 : 0.35 }, fr.svg);
        const seq = on.get("rr") ? PATH : ["none"];
        const c = col(kMetric, day, g);
        const xy = (d) => [fr.x(KR[d].weighted_utility), fr.y(KR[d][c])];
        if (seq.length > 1) el("polyline", { points: seq.map((d) => xy(d).join(",")).join(" "), fill: "none", stroke: color, "stroke-width": hot ? 2.5 : 1.5 }, layer);
        for (const d of seq) {
          const [x, y] = xy(d);
          point(layer, d, d === "none" ? "star" : "circle", x, y, d === "none" ? (hot ? 7.5 : 6) : (hot ? 4.5 : 3.5), color, g);
          if (hot && d !== "none") el("text", { class: "pt-label", x, y: y - 9, "text-anchor": "middle", text: `q=${qOf(d)}` }, layer);
        }
      }
      for (const [v, , shape] of EQ) {
        if (!on.get(v)) continue;
        for (const g of otherFirst(kG)) {
          const d = `${v}_g${g}`, hot = g === kG;
          const layer = el("g", { opacity: hot ? 1 : 0.4 }, fr.svg);
          point(layer, d, shape, fr.x(KR[d].weighted_utility), fr.y(KR[d][col(kMetric, day, g)]), hot ? 7 : 5, GCOL[g], g);
        }
      }
      plotPoints(fr, pts);
    }

    $("#k-legend").innerHTML = [
      legendItem("방어 없음", GREY, { shape: "star", dash: "0 24" }),
      on.get("rr") ? legendItem(`무작위 응답 RR (q = ${K.rr.map(qOf).join(", ")})`, GREY, { shape: "circle" }) : "",
      ...EQ.filter(([v]) => on.get(v)).map(([, label, shape, what]) => legendItem(`${label}: ${what}`, GREY, { shape, dash: "0 24" })),
      ...K.groups.map((g) => legendItem(`g = ${g}${g === S.g ? " (선언된 g*)" : ""}`, GCOL[g], { shape: "square", dash: "0 24" })),
      kMetric === "nway" ? legendItem("검정: 그룹 없음 (방어 없음·RR)", INK, {}) : "",
    ].join("");

    const eq = `eqc_g${kG}`, none = KR.none, e = KR[eq];
    $("#k-head").textContent = `방어 없음 → EQ-confusion, g = ${kG}${kG === S.g ? " (선언된 g*)" : ""}, 명령 ${S.k}개`;
    const dl = $("#k-stats");
    dl.innerHTML = "";
    const stat = (label, value, column) => {
      const source = `Source: ${K.source.file} (defence = none, ${eq}; ${column})`;
      const d = el("div", { title: source, tabindex: "0" }, dl);
      el("dt", { text: label }, d); el("dd", { text: value }, d);
      el("span", { class: "sr", text: source }, d);
      focusTipOn(d, source);
    };
    for (const day of K.days) stat(`그룹 내 초과, ${DAY[day]}`, arrow(none[col("excess", day, kG)], e[col("excess", day, kG)]), col("excess", day, kG));
    for (const day of K.days) stat(`${K.users}명 중 식별, ${DAY[day]}`, arrow(none[col("nway", day)], e[col("nway", day)]), col("nway", day));
    stat("가중 유용성", arrow(none.weighted_utility, e.weighted_utility), "weighted_utility");
    stat("정지 비율", arrow(none.hold_rate, e.hold_rate), "hold_rate");

    const defs = [...PATH, ...EQ.map(([v]) => `${v}_g${kG}`)];
    table($("#k-table"), ["방어", "가중 유용성", "유용성 손실", "정지 비율",
      ...K.days.map((day) => `그룹 내 초과 g = ${kG}, ${DAY[day]}`), ...K.days.map((day) => `${K.users}명 중 식별, ${DAY[day]}`)],
    defs.map((d) => { const r = KR[d]; return [name(d), m3(r.weighted_utility), m3(r.weighted_utility_loss), f3(r.hold_rate),
      ...K.days.map((day) => m3(r[col("excess", day, kG)])), ...K.days.map((day) => f3(r[col("nway", day)]))]; }));
  }
  drawK();
  K.days.forEach((day) => onResize($(`#k-chart-${day}`), drawK));

  const all = KR.eqc_gall, e3 = KR[`eqc_g${S.g}`];
  $("#k-note").innerHTML = `가중 유용성 = P(정답 이동) − w·P(오답 이동), w = τ/(1 − τ) = ${S.wrong_weight}. `
    + "그룹 내 초과 = 그룹이 주어졌을 때 자기 그룹 안에서의 식별 정확도 − 1/(자기 그룹 크기)이며 0이 우연 수준이다. "
    + `${K.users}명의 그룹 구성: ${K.groups.map(sizeText).join("; ")} (<code>${K.group_sizes_source.file}</code> ${K.group_sizes_source.where.function}). `
    + `전체 식별은 ${K.users}명 중 균형 정확도이며 우연 수준은 1/${K.users} = ${f3(K.chance)}이다. `
    + `각 점은 Bayes 공격자 ${K.attackers}종(범주형·Markov, 목표 무시·목표 인지) 중 최강값의 seed ${K.seeds}개 평균이다. `
    + `다른 날 공격자는 교차 적합(cross-fitted)한 보정일 명령으로 등록한다. 출처: <code>${K.source.file}</code> `
    + `(dataset = ${S.dataset}, condition = ${S.condition}, tau = ${S.tau}, k = ${S.k} 열). 사용자 수·우연 수준·seed·공격자 수: <code>${K.chance_source.file}</code>. `
    + `같은 비교의 정적 그림: <a href="../${K.figure}">${K.figure}</a>.`;
  const limits = [
    `EQ의 균등성은 적합에 쓴 보정일 혼동 행렬에 대해서만 정확하다. 배포일에 등록한 같은 날 공격자에게는 g = ${S.g}에서도 그룹 내 초과가 남는다(${m3(e3[col("excess", "within_day", S.g)])}).`,
    "EQ의 유용성이 방어 없음보다 높은 것은 공짜 이득이 아니다. EQ의 선형계획이 각 사용자의 명령을 그 사용자의 유용성 최적점으로 재보정하므로, 비교에 재보정 효과가 섞여 있다.",
    "K의 RR에는 재보정이 없다. 따라서 이 그래프는 같은 유용성에서 EQ와 RR을 비교하지 않는다.",
    `EQ는 자기 그룹 안에서만 신원을 감추고(${sizeText(S.g)}), 그룹 소속은 드러난다. ${K.users}명 전원을 한 그룹(g = 전체)으로 묶으면 EQ-confusion의 최선은 '항상 정지'다(가중 유용성 ${m3(all.weighted_utility)}, 정지 비율 ${f3(all.hold_rate)}).`,
    "K는 이미 분석한 BCI IV 2a·Lee2019 데이터를 재사용하므로 모든 K 수치는 탐색적이며, 적합된 시스템에 조건부다.",
  ];
  const ul = $("#k-limits");
  limits.forEach((t) => el("li", { text: t }, ul));
}
