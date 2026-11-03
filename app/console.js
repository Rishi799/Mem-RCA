/* Mem-RCA console front end. Talks to scripts/serve_console.py; no external libraries. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : `${Math.round(100 * v)}%`);
const SVGNS = "http://www.w3.org/2000/svg";
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
/* plain-language names for fault types; names an engineer enrols are shown as typed */
const FAULT_NAMES = { cpu: "CPU overload", mem: "Memory pressure", disk: "Disk slowdown", delay: "Network delay", loss: "Packet loss",
  socket: "Connection exhaustion", f1: "Code bug F1", f2: "Code bug F2", f3: "Code bug F3", f4: "Code bug F4", f5: "Code bug F5", unknown: "Unknown" };
const SHORT_NAMES = { cpu: "CPU", mem: "Memory", disk: "Disk", delay: "Delay", loss: "Loss", socket: "Sockets" };
const nice = (type) => FAULT_NAMES[type] || type;
const short = (type) => SHORT_NAMES[type] || type;
const METRIC_NAMES = { cpu: "CPU", mem: "memory", diskio: "disk I/O", socket: "sockets", workload: "requests", "latency-50": "median latency",
  "latency-90": "p90 latency", latency: "latency", load: "requests", error: "errors" };
const niceMetric = (m) => METRIC_NAMES[m] || m;
const niceText = (text) => text.replace(/\b(cpu|mem|disk|delay|loss|socket)\b/g, (m) => nice(m));

const app = {
  scenarios: [], state: null, incident: null, timer: null,
  showTruth: false, colors: new Map(), busy: false, propView: innerWidth < 600 ? "list" : "graph", propLive: false, propBase: 0,
  mode: "live", live: null,
};

/* one replay session per browser tab */
const SESSION = (() => {
  let id = null;
  try { id = sessionStorage.getItem("memrca-session"); } catch (_) { /* storage may be unavailable */ }
  if (!id) {
    id = Math.random().toString(36).slice(2) + Date.now().toString(36);
    try { sessionStorage.setItem("memrca-session", id); } catch (_) { /* ignore */ }
  }
  return id;
})();

async function api(path, body) {
  const headers = { "X-Session": SESSION };
  const res = await fetch(path, body === undefined ? { headers } : {
    method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

/* ---------- colour per failure type: fixed order of first appearance, never cycled ---------- */
function typeColor(type) {
  if (!app.colors.has(type)) {
    const slot = app.colors.size;
    app.colors.set(type, slot < SERIES.length ? `var(${SERIES[slot]})` : "var(--ink-3)");
  }
  return app.colors.get(type);
}

/* ---------- tooltip: any element with data-tip ---------- */
const tip = $("tooltip");
function showTip(text, x, y) {
  tip.textContent = text;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = `${Math.min(window.innerWidth - w - 8, x + 14)}px`;
  tip.style.top = `${Math.max(8, y - h - 10)}px`;
}
document.addEventListener("mousemove", (e) => {
  const el = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (el) showTip(el.getAttribute("data-tip"), e.clientX, e.clientY);
  else if (!e.target.closest || !e.target.closest(".crosshair-zone")) tip.hidden = true;
});

/* ---------- small SVG helpers ---------- */
function svg(w, h, cls = "") {
  const s = document.createElementNS(SVGNS, "svg");
  s.setAttribute("viewBox", `0 0 ${w} ${h}`);
  s.setAttribute("preserveAspectRatio", "xMidYMid meet");
  if (cls) s.setAttribute("class", cls);
  return s;
}
function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}
/* bar with a 4px rounded data end and a square baseline end */
function barPath(x0, x1, y, h, r = 4) {
  if (x1 < x0) [x0, x1] = [x1, x0];
  const rr = Math.min(r, (x1 - x0), h / 2);
  return `M${x0},${y} H${x1 - rr} Q${x1},${y} ${x1},${y + rr} V${y + h - rr} Q${x1},${y + h} ${x1 - rr},${y + h} H${x0} Z`;
}
function barPathLeft(x0, x1, y, h, r = 4) { /* rounded end on the left (negative values) */
  const rr = Math.min(r, (x1 - x0), h / 2);
  return `M${x1},${y} H${x0 + rr} Q${x0},${y} ${x0},${y + rr} V${y + h - rr} Q${x0},${y + h} ${x0 + rr},${y + h} H${x1} Z`;
}
function columnPath(x, y0, y1, w, r = 4) { /* vertical bar from baseline y0 up to y1, rounded top */
  const rr = Math.min(r, (y0 - y1), w / 2);
  return `M${x},${y0} V${y1 + rr} Q${x},${y1} ${x + rr},${y1} H${x + w - rr} Q${x + w},${y1} ${x + w},${y1 + rr} V${y0} Z`;
}

/* ---------- bar list (HTML) ---------- */
function barList(container, rows, { max = 1, format = (v) => fmt(v), color = "var(--accent)", highlight = null, tipFor = null } = {}) {
  container.innerHTML = rows.map((r) => {
    const w = Math.max(0, Math.min(100, (100 * r.value) / (max || 1)));
    const c = typeof color === "function" ? color(r) : color;
    const tipText = tipFor ? tipFor(r) : `${r.name}: ${format(r.value)}`;
    const name = r.swatch ? `<span class="swatch" style="background:${r.swatch}"></span>${esc(r.name)}` : esc(r.name);
    return `<div class="bar-row" data-tip="${esc(tipText)}"><span class="name ${highlight && r.name === highlight ? "anchor" : ""}">${name}</span>` +
      `<span class="track"><i style="width:${w}%;background:${c}"></i></span><span class="val">${format(r.value)}</span></div>`;
  }).join("");
}

/* ======================= CONSOLE ======================= */
async function loadScenarios() {
  app.scenarios = await api("/api/scenarios");
  const select = $("scenario");
  const groups = {};
  for (const s of app.scenarios) (groups[s.group] ||= []).push(s);
  select.innerHTML = Object.entries(groups).map(([g, list]) =>
    `<optgroup label="${esc(g)}">${list.map((s) => `<option value="${esc(s.id)}">${esc(s.title.replace(/\b(cpu|mem|disk|delay|loss|socket)\b/g, (m) => nice(m)))}</option>`).join("")}</optgroup>`).join("");
  const preferred = app.scenarios.find((s) => s.id === "re2:tt:fold1");
  if (preferred) select.value = preferred.id;
  describeScenario();
  if (!app.scenarios.length) $("scenario-desc").textContent = "No scenarios found: run the experiment scripts first (see README).";
}

function describeConfig(config) {
  const [space, rest] = config.split(":");
  const [pooling, memory] = rest.split("+");
  const root = pooling.endsWith("_pred") ? "learned localiser" : pooling.endsWith("_shift") ? "largest metric shift" : null;
  const view = pooling.replace(/_(pred|shift|oracle)$/, "");
  const viewName = { anchor: "root-anchored", global_max: "strongest symptom (global max)", global_mean: "average over services", hybrid: "anchored + global max" }[view] || view;
  const memoryName = memory.startsWith("split") ? `k-NN familiarity (k=${memory.slice(5)}) + prototype naming` : memory;
  return { viewName, root, memoryName, space };
}

function describeScenario() {
  const s = app.scenarios.find((x) => x.id === $("scenario").value);
  if (!s) return;
  $("scenario-desc").textContent = niceText(s.description);
  const c = describeConfig(s.config);
  const signature = { "root-anchored": "centred on the origin", "strongest symptom (global max)": "strongest symptoms",
    "anchored + global max": "origin + strongest symptoms", "average over services": "service average" }[c.viewName] || c.viewName;
  $("config").innerHTML = [
    `<span class="chip">Origin detection: <b>${s.id.startsWith("re3") ? "largest metric deviation" : "deep localiser (LSTM + GAT)"}</b></span>`,
    `<span class="chip">Pattern signature: <b>${esc(signature)}</b></span>`,
    `<span class="chip">Blast radius: <b>PropagationNet</b></span>`,
  ].join("");
}

const systemName = () => (app.state ? app.state.title.split(" · ")[0] : "");

function setStatus(text, mode = "") {
  $("status-pill").className = `status-pill ${mode}`;
  $("status-pill").querySelector("span").textContent = text;
}

async function startScenario() {
  stopPlay();
  app.busy = true;
  $("start").disabled = true;
  try {
    const { state } = await api("/api/session", { scenario: $("scenario").value });
    app.colors = new Map();
    state.types.forEach((t) => typeColor(t.type));
    app.state = state;
    app.incident = null;
    $("incident-body").hidden = true; $("incident-empty").hidden = false;
    stopPropPlay(); app.prop = null; $("prop-body").hidden = true; $("prop-empty").hidden = false;
    $("verdict-body").hidden = true; $("verdict-empty").hidden = false;
    $("incident-id").textContent = app.mode === "live" ? "No active incident" : `${state.total} incidents in this replay`;
    $("situation").hidden = true;
    setStatus(`${app.mode === "live" ? "Monitoring" : "Replaying"} · ${systemName()}`, "live");
    $("verdict-step").textContent = "";
    $("next").disabled = false; $("play").disabled = false;
    $("start").textContent = app.mode === "live" ? "Restart monitoring" : "Restart replay";
    renderState();
  } catch (err) {
    alert(err.message);
  } finally {
    app.busy = false;
    $("start").disabled = false;
  }
}

async function nextIncident() {
  if (app.busy) return;
  app.busy = true;
  try {
    const { incident, state } = await api("/api/next", {});
    app.state = state;
    if (!incident) {
      stopPlay();
      $("next").disabled = true; $("play").disabled = true;
      $("incident-id").textContent = "Replay complete";
      setStatus("Replay complete", "done");
    } else {
      app.incident = incident;
      renderIncident();
      loadPropagation();
    }
    renderState();
  } finally {
    app.busy = false;
  }
}

function togglePlay() {
  if (app.timer) return stopPlay();
  $("play").textContent = "❚❚ Pause";
  app.timer = setInterval(nextIncident, 6000);
  nextIncident();
}
function stopPlay() {
  clearInterval(app.timer);
  app.timer = null;
  $("play").textContent = "▶ Auto";
}

async function enrol(candidateId, name) {
  const { state } = await api("/api/enrol", { candidate: candidateId, name });
  app.state = state;
  state.types.forEach((t) => typeColor(t.type));
  renderState();
}

/* ---------- state panels ---------- */
function renderState() {
  const s = app.state;
  if (!s) return;
  const k = s.kpis;
  $("progress").style.width = `${(100 * s.position) / Math.max(1, s.total)}%`;
  const tiles = [
    ["Incidents", `${k.processed}`, `of ${s.total} in this replay`],
    ["Known pattern", `${k.recognised}`, "matched a known failure", "var(--good)"],
    ["New pattern", `${k.novel}`, "no known pattern close enough", "var(--warn)"],
    ["Patterns learned", `${k.enrolled}`, "added during this replay", "var(--violet)"],
    ["Pattern library", `${k.types}`, `match threshold ${fmt(s.tau)}`],
    ["Decision accuracy", app.showTruth ? pct(k.open_set_accuracy) : "—", app.showTruth ? "evaluation mode" : "enable evaluation mode"],
  ];
  $("kpis").innerHTML = tiles.map(([label, value, hint, dot]) =>
    `<div class="kpi"><div class="label">${dot ? `<span class="status-dot" style="background:${dot}"></span>` : ""}${label}</div>` +
    `<div class="value">${value}</div><div class="hint">${hint}</div></div>`).join("");

  $("memory-sub").textContent = `${s.types.length} patterns · ${s.buffer.length} unclassified`;
  $("types").innerHTML = s.types.map((t) =>
    `<span class="type-pill ${t.enrolled ? "new" : ""}" data-tip="${esc(`${t.type}: ${t.stored} stored incidents${t.enrolled ? " (learned during this replay)" : ""}`)}">` +
    `<span class="swatch" style="background:${typeColor(t.type)}"></span>${esc(nice(t.type))}<span class="count">${t.stored}</span>` +
    `${t.enrolled ? '<span class="badge new">new</span>' : ""}</span>`).join("");

  renderCandidates();
  renderMap();
  renderTimeline();
}

function renderCandidates() {
  const s = app.state;
  const box = $("candidates");
  if (!s.candidates.length) {
    box.innerHTML = `<p class="muted small">${s.buffer.length ? `${s.buffer.length} unclassified incident${s.buffer.length > 1 ? "s" : ""}. A group is proposed once three or more look alike.` :
      "Incidents that match no known pattern collect here. Once three or more look alike, they are proposed as a new pattern for review."}</p>`;
    return;
  }
  box.innerHTML = s.candidates.map((c) => {
    const truth = app.showTruth ? ` · ground truth mostly <b>${esc(nice(c.truth_majority))}</b>` : "";
    return `<div class="candidate"><div><b>Proposed new pattern</b> · ${c.size} similar incidents</div>` +
      `<div class="meta">Strongest deviations at the origin: ${c.signature.map((m) => `<b>${esc(niceMetric(m))}</b>`).join(", ")}${truth}</div>` +
      `<div class="row"><input type="text" id="name-${c.id}" placeholder="Name this pattern, e.g. ${esc(niceMetric(c.signature[0]))} saturation" aria-label="Name for the new pattern">` +
      `<button class="btn small primary" data-enrol="${c.id}">Create pattern</button></div></div>`;
  }).join("");
  box.querySelectorAll("[data-enrol]").forEach((b) => b.addEventListener("click", () => {
    const id = Number(b.dataset.enrol);
    enrol(id, $(`name-${id}`).value);
  }));
}

function renderTimeline() {
  const h = app.state.history;
  if (!h.length) { $("timeline").innerHTML = '<span class="muted small">No incidents yet.</span>'; return; }
  $("timeline").innerHTML = h.map((r) => {
    const novel = r.verdict === "NOVEL";
    const wrong = app.showTruth && !r.truth.correct;
    const label = `#${r.step}: ${novel ? "new failure (NOVEL)" : nice(r.verdict)}` + (app.showTruth ? `\nground truth: ${nice(r.truth.type)} at ${r.truth.root}${wrong ? "\n✗ wrong" : "\n✓ correct"}` : "");
    return `<span class="tick ${novel ? "novel" : "known"} ${wrong ? "wrong" : ""}" data-tip="${esc(label)}">${novel ? "?" : "✓"}</span>`;
  }).join("");
}

/* ---------- memory map (PCA of the memory space) ---------- */
function renderMap() {
  const s = app.state, box = $("map");
  const W = Math.max(280, box.clientWidth || 460), H = Math.round(W * 0.62), P = 18;
  const pts = [...s.map.stored.map((p) => p.xy), ...s.map.prototypes.map((p) => p.xy), ...s.map.buffer.map((p) => p.xy)];
  if (app.incident) pts.push(app.incident.map);
  s.candidates.forEach((c) => pts.push(c.map));
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x) => P + ((x - x0) / ((x1 - x0) || 1)) * (W - 2 * P);
  const sy = (y) => H - P - ((y - y0) / ((y1 - y0) || 1)) * (H - 2 * P);
  const root = svg(W, H);
  for (const p of s.map.stored) {
    el("circle", { cx: sx(p.xy[0]), cy: sy(p.xy[1]), r: 3.5, fill: typeColor(p.type), "fill-opacity": 0.55, "data-tip": `past incident · ${p.type}` }, root);
  }
  for (const p of s.map.buffer) {
    el("circle", { cx: sx(p.xy[0]), cy: sy(p.xy[1]), r: 6, fill: "none", stroke: "var(--warn)", "stroke-width": 2, "data-tip": "unclassified incident" }, root);
  }
  for (const c of s.candidates) {
    el("circle", { cx: sx(c.map[0]), cy: sy(c.map[1]), r: 16, fill: "var(--warn)", "fill-opacity": 0.12, stroke: "var(--warn)", "stroke-dasharray": "3 3", "stroke-width": 1.5, "data-tip": `proposed new pattern · ${c.size} incidents` }, root);
  }
  for (const p of s.map.prototypes) {
    const cx = sx(p.xy[0]), cy = sy(p.xy[1]);
    el("circle", { cx, cy, r: 7, fill: typeColor(p.type), stroke: "var(--surface-2)", "stroke-width": 2, "data-tip": `pattern centre · ${p.type}` }, root);
    text(root, cx + 10, cy + 4, short(p.type), { class: "label-text", "font-weight": 600 });
  }
  if (app.incident) {
    const [cx, cy] = [sx(app.incident.map[0]), sy(app.incident.map[1])];
    el("circle", { cx, cy, r: 11, fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, root);
    el("circle", { cx, cy, r: 4, fill: "var(--ink)", "data-tip": `current incident · ${app.incident.verdict === "NOVEL" ? "new pattern" : app.incident.verdict}` }, root);
  }
  box.replaceChildren(root);
  $("map-legend").innerHTML = `<span><span class="swatch" style="background:var(--ink-3)"></span>past incident (colour = pattern)</span>` +
    `<span><span class="swatch" style="background:var(--ink-2);width:12px;height:12px"></span>pattern centre</span>` +
    `<span><span class="ring"></span>unclassified</span><span>◎ current incident</span>`;
}

/* ---------- incident panels ---------- */
const incidentId = (step) => `INC-${String(step).padStart(4, "0")}`;
function renderIncident() {
  const inc = app.incident;
  $("incident-empty").hidden = true; $("incident-body").hidden = false;
  $("verdict-empty").hidden = true; $("verdict-body").hidden = false;
  $("incident-id").textContent = app.showTruth ? `${incidentId(inc.step)} · ${inc.case}` : incidentId(inc.step);
  $("verdict-step").textContent = incidentId(inc.step);

  const learned = inc.ranking.learned;
  $("rank-learned").innerHTML = learned ? "" : '<p class="muted small">No trained localiser for this suite; the anchor uses the largest metric shift.</p>';
  if (learned) barList($("rank-learned"), learned.map((r) => ({ name: r.service, value: r.score })), {
    max: 1, format: (v) => pct(v), highlight: inc.anchor, tipFor: (r) => `${r.name}: ${pct(r.value)} root probability`,
  });
  const shift = inc.ranking.metric_shift;
  barList($("rank-shift"), shift.map((r) => ({ name: r.service, value: r.score })), {
    max: Math.max(...shift.map((r) => r.score), 1e-6), format: (v) => fmt(v, 1), color: "var(--ink-3)", highlight: inc.anchor,
    tipFor: (r) => `${r.name}: largest shift ${fmt(r.value, 2)} (log10 robust z)`,
  });

  if (inc.series) {
    const [svc, ...metric] = inc.series.metric.split("_");
    $("series-title").textContent = `${shortName(svc)} service: ${niceMetric(metric.join("_"))} before and after the fault`;
  } else $("series-title").textContent = "Telemetry";
  lineChart($("series"), clipSeries(inc.series));

  const t = inc.truth;
  $("incident-truth").hidden = !app.showTruth;
  $("incident-truth").innerHTML = `Ground truth: <b>${esc(nice(t.type))}</b> fault injected at <b>${esc(t.root)}</b>` +
    `${t.known_at_start ? "" : " · a type the memory did not know at the start"} · verdict <b>${t.correct ? "correct ✓" : "wrong ✗"}</b>`;

  renderVerdict(inc);
}

function renderVerdict(inc) {
  const novel = inc.verdict === "NOVEL";
  $("verdict").className = `verdict ${novel ? "novel" : "known"}`;
  $("verdict").innerHTML = `<div class="icon">${novel ? "?" : "✓"}</div><div>` +
    `<div class="title">${novel ? "New failure pattern" : `Known pattern: ${esc(nice(inc.verdict))}`}</div>` +
    `<div class="detail">${novel ? `No known pattern is close enough (best match ${fmt(inc.familiarity)}, threshold ${fmt(inc.tau)}). The incident is held as unclassified, and similar incidents will be grouped for review.` :
      `Matches “${esc(nice(inc.verdict))}” with similarity ${fmt(inc.familiarity)}, above the threshold ${fmt(inc.tau)}.`}${inc.anchor ? ` Signature centred on <b>${esc(inc.anchor)}</b>.` : ""}</div></div>`;
  gauge($("gauge"), inc.familiarity, inc.tau, novel);
  const sims = inc.similar_types;
  barList($("similar"), sims.map((r) => ({ name: nice(r.type), code: r.type, value: Math.max(0, r.similarity), raw: r.similarity, swatch: typeColor(r.type) })), {
    max: 1, format: (v) => fmt(v), color: (r) => typeColor(r.code), tipFor: (r) => `similarity to ${r.name}: ${fmt(r.raw, 3)}`,
  });
  fingerprintPanels($("fingerprint"), inc.segments, inc.metric_types);
  renderSituation();
}

/* incident header: id, title, status badges, and the three answers an on-call engineer needs */
function renderSituation() {
  const inc = app.incident;
  if (!inc) { $("situation").hidden = true; return; }
  $("situation").hidden = false;
  const novel = inc.verdict === "NOVEL";
  const learned = inc.ranking.learned;
  const root = inc.anchor || (learned || inc.ranking.metric_shift)[0].service;
  const rootScore = learned ? (learned.find((r) => r.service === root) || learned[0]).score : null;
  const values = inc.segments[0].values;
  const strongest = values.reduce((best, v, i) => (Math.abs(v) > Math.abs(values[best]) ? i : best), 0);
  const symptom = niceMetric(inc.metric_types[strongest]);
  let blast = null;
  if (app.prop && app.prop.case === inc.case) {
    const t = Number($("prop-time").value);
    const others = propState(t).filter((n) => !n.root);
    blast = { t, hit: others.filter((n) => n.hit),
      soon: others.filter((n) => serviceStatus(n, t).cls === "soon").sort((a, b) => (a.q50 ?? 999) - (b.q50 ?? 999)) };
  }
  $("ih-id").textContent = incidentId(inc.step);
  $("ih-title").textContent = novel ? `Unknown failure on ${root} · strongest signal: ${symptom}` : `${nice(inc.verdict)} on ${root}`;
  const badges = [];
  if (app.mode === "live" && app.live && app.live.incStep === inc.step) badges.push(app.live.closed ? ["muted", "Closed"] : ["crit", "Open"]);
  badges.push(novel ? ["warn", "New pattern"] : ["good", "Known pattern"]);
  if (blast !== null) {
    const n = blast.soon.length;
    badges.push(n ? ["crit", `${n} service${n === 1 ? "" : "s"} at risk`] : blast.hit.length ? ["muted", "No further spread expected"] : ["muted", "Contained"]);
  }
  if (app.showTruth) {
    badges.push(root === inc.truth.root ? ["good", "Origin correct"] : ["crit", `Origin missed · actual ${inc.truth.root}`]);
    badges.push(inc.truth.correct ? ["good", "Pattern correct"] : ["crit", "Pattern wrong"]);
  }
  $("ih-badges").innerHTML = badges.map(([c, t]) => `<span class="badge-pill ${c}">${esc(t)}</span>`).join("");
  $("ih-root").textContent = root;
  $("ih-root-sub").textContent = rootScore !== null ? `${pct(rootScore)} confidence · deep localiser` : "largest metric deviation";
  $("ih-diag").textContent = novel ? "New failure pattern" : nice(inc.verdict);
  $("ih-diag-sub").textContent = novel ? `no known pattern above threshold ${fmt(inc.tau)}` : `similarity ${fmt(inc.familiarity)} · threshold ${fmt(inc.tau)}`;
  $("ih-blast").textContent = blast === null ? "Forecasting…" : `${blast.hit.length} affected · ${blast.soon.length} at risk`;
  $("ih-blast-sub").textContent = blast === null ? "" : blast.soon.length ? `next: ${shortName(blast.soon[0].service)} ${eta(blast.soon[0], blast.t)}` : "no further impact expected";
}

function gauge(container, value, tau, novel) {
  const W = Math.max(260, container.clientWidth || 520), H = 52, L = 8, R = 8;
  const lo = Math.min(-0.2, value - 0.05), hi = 1;
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const root = svg(W, H);
  el("rect", { x: L, y: 16, width: W - L - R, height: 10, rx: 5, fill: "var(--surface-2)" }, root);
  el("path", { d: barPath(x(Math.max(lo, Math.min(0, value))), x(value), 16, 10, 5), fill: novel ? "var(--warn)" : "var(--good)" }, root);
  el("line", { x1: x(tau), x2: x(tau), y1: 8, y2: 34, stroke: "var(--ink)", "stroke-width": 2 }, root);
  text(root, x(tau), 46, `τ = ${fmt(tau)}`, { class: "axis-text", "text-anchor": "middle", "font-weight": 600 });
  text(root, L, 8, "familiarity", { class: "axis-text" });
  text(root, Math.min(W - R - 2, Math.max(L + 40, x(value))), 8, fmt(value), { class: "axis-text", "text-anchor": "end", "font-weight": 600 });
  root.setAttribute("data-tip", `familiarity ${fmt(value, 3)} vs threshold τ ${fmt(tau, 3)} (5th percentile of known validation incidents)`);
  container.replaceChildren(root);
}

function fingerprintPanels(container, segments, types) {
  const span = Math.max(1, ...segments.flatMap((s) => s.values.map((v) => Math.abs(v))));
  container.innerHTML = segments.map((seg) => `<div class="fp-panel"><div class="fp-title">${esc(seg.name)}</div>` +
    types.map((t, i) => {
      const v = seg.values[i], w = (50 * Math.abs(v)) / span;
      const bar = Math.abs(v) > 1e-9 ? `<i class="${v > 0 ? "pos" : "neg"}" style="width:${w}%"></i>` : "";
      const tipText = `${seg.name} · ${t}: ${v > 0 ? "+" : ""}${fmt(v)} (log10 robust z, ${v > 0 ? "increase" : v < 0 ? "decrease" : "no change"})`;
      return `<div class="fp-row" data-tip="${esc(tipText)}"><span>${esc(niceMetric(t))}</span><span class="fp-track">${bar}</span>` +
        `<span class="val">${Math.abs(v) > 1e-9 ? `${v > 0 ? "+" : ""}${fmt(v, 1)}` : "0"}</span></div>`;
    }).join("") + "</div>").join("");
}

/* in live monitoring nothing after the replay clock may be drawn */
function clipSeries(series) {
  const L = app.live;
  if (!series || app.mode !== "live" || !L || !L.tele || !L.opened) return series;
  const keep = series.t.map((t) => t <= L.tau);
  return { ...series, t: series.t.filter((_, i) => keep[i]), v: series.v.filter((_, i) => keep[i]) };
}

function lineChart(container, series) {
  if (!series || !series.t.length) { container.innerHTML = '<p class="muted small">Raw telemetry not available for this incident.</p>'; return; }
  const W = Math.max(300, container.clientWidth || 520), H = 170, L = 44, R = 10, T = 12, B = 24;
  const ts = series.t, vs = series.v;
  const [t0, t1] = [Math.min(...ts), Math.max(...ts)];
  let [v0, v1] = [Math.min(...vs), Math.max(...vs)];
  if (v1 - v0 < 1e-9) { v0 -= 1; v1 += 1; }
  const pad = (v1 - v0) * 0.08; v0 -= pad; v1 += pad;
  const x = (t) => L + ((t - t0) / ((t1 - t0) || 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - v0) / (v1 - v0)) * (H - T - B);
  const root = svg(W, H, "crosshair-zone");
  for (let k = 0; k <= 3; k++) {
    const v = v0 + ((v1 - v0) * k) / 3;
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid-line" }, root);
    text(root, L - 6, y(v) + 3, Math.abs(v) >= 1000 ? v.toExponential(1) : Number(v.toPrecision(3)), { class: "axis-text", "text-anchor": "end" });
  }
  [t0, 0, t1].forEach((t) => text(root, x(t), H - 6, `${t > 0 ? "+" : ""}${t}s`, { class: "axis-text", "text-anchor": "middle" }));
  el("rect", { x: x(0), y: T, width: W - R - x(0), height: H - T - B, fill: "var(--warn)", "fill-opacity": 0.08 }, root);
  el("line", { x1: x(0), x2: x(0), y1: T, y2: H - B, stroke: "var(--warn-ink)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, root);
  text(root, x(0) + 6, T + 10, "fault injected", { class: "axis-text", fill: "var(--warn-ink)" });
  el("path", { d: ts.map((t, i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(vs[i]).toFixed(1)}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round" }, root);
  const cross = el("line", { y1: T, y2: H - B, stroke: "var(--ink-3)", "stroke-width": 1, visibility: "hidden" }, root);
  const dot = el("circle", { r: 4, fill: "var(--accent)", stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, root);
  const zone = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, root);
  zone.addEventListener("mousemove", (e) => {
    const box = root.getBoundingClientRect();
    const t = t0 + ((e.clientX - box.left) / box.width * W - L) / (W - L - R) * (t1 - t0);
    let i = 0, best = Infinity;
    ts.forEach((tt, k) => { const d = Math.abs(tt - t); if (d < best) { best = d; i = k; } });
    cross.setAttribute("x1", x(ts[i])); cross.setAttribute("x2", x(ts[i])); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(ts[i])); dot.setAttribute("cy", y(vs[i])); dot.setAttribute("visibility", "visible");
    showTip(`${ts[i] > 0 ? "+" : ""}${ts[i]}s after injection\n${series.metric}: ${Number(vs[i].toPrecision(4))}`, e.clientX, e.clientY);
  });
  zone.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; });
  container.replaceChildren(root);
}

/* ---------- propagation forecast (PropagationNet) ---------- */
const shortName = (s) => s.replace(/^ts-/, "").replace(/-service$/, "").replace(/service$/, "");

async function loadPropagation() {
  try {
    app.prop = await api("/api/propagation", {});
  } catch (err) {
    app.prop = null;
    $("prop-empty").hidden = false; $("prop-body").hidden = true;
    $("prop-empty").innerHTML = `<p class="muted">${esc(err.message)}</p>`;
    return;
  }
  stopPropPlay();
  $("prop-empty").hidden = true; $("prop-body").hidden = false;
  $("prop-time").max = app.prop.horizon_s - app.prop.bin_s;
  $("prop-time").step = app.prop.bin_s;
  $("prop-time").value = 0;
  app.propAll = false;
  app.propBase = 0;
  renderPropagation();
  renderSituation();
}

/* index of the forecast on screen: the one made at detection, or the latest one when live updates are on */
const forecastIndex = (t) => {
  const last = app.prop.forecasts.length - 1, base = Math.min(app.propBase, last);
  return app.propLive ? Math.max(base, Math.min(Math.floor(t / app.prop.bin_s), last)) : base;
};

function propState(t) {
  const P = app.prop;
  const k = forecastIndex(t);
  const f = P.forecasts[k], f0 = P.forecasts[0];
  return P.services.map((s, i) => {
    const actual = P.actual_s[i];
    return {
      service: s, i, root: s === P.root, inGraph: P.in_graph[i] || s === P.root,
      hit: actual !== null && actual <= t, actual, p: f.p_hit[i], q10: f.q10[i], q50: f.q50[i], q90: f.q90[i],
      p0: f0.p_hit[i], q10_0: f0.q10[i], q50_0: f0.q50[i], q90_0: f0.q90[i],
      up: P.hop_up[i], down: P.hop_down[i], any: P.hop_any[i],
    };
  });
}

/* where a microservice sits relative to the broken one: [lane key, label] */
function relation(n) {
  if (n.root) return ["root", "This is where the fault is"];
  if (!n.inGraph) return ["off", "No traced link"];
  const caller = n.up !== null && (n.down === null || n.up <= n.down);
  if (caller) return n.up === 1 ? ["up1", "Calls the broken service"] : ["up2", `Calls it indirectly (${n.up} steps)`];
  if (n.down !== null) return n.down === 1 ? ["down1", "Called by the broken service"] : ["down2", `Further downstream (${n.down} steps)`];
  return ["side", "Linked through other services"];
}
const LANES = [
  ["root", "Where it started", "The microservice with the fault."],
  ["up1", "Calls the broken service", "Usually hit first: they wait for answers that are now slow or failing."],
  ["up2", "Calls it indirectly", "Two or more calls up the chain."],
  ["down1", "Called by the broken service", "Hit less often: they just receive fewer or odd requests."],
  ["down2", "Further downstream", "Two or more calls down the chain."],
  ["side", "Linked through other services", "Connected, but not through a direct call chain."],
  ["off", "No traced link to it", "Only the ones at risk or already affected are shown."],
];
const STATUS_ORDER = { root: 0, hit: 1, soon: 2, late: 3, maybe: 4, calm: 5 };

function eta(n, t) {
  if (n.q50 === null) return "likely";
  return n.q50 > t ? `in ~${n.q50 - t} s` : "due now";
}

function serviceStatus(n, t) {
  if (n.root) return { cls: "root", text: "fault starts here" };
  if (n.hit) return { cls: "hit", text: n.actual === 0 ? "affected from the start" : `affected at +${n.actual} s` };
  /* the expected time has passed by more than one step without an impact: the prediction has not come true (yet) */
  if (n.p >= 0.5 && n.q50 !== null && t > n.q50 + app.prop.bin_s) return { cls: "late", text: `predicted ~+${n.q50} s · not affected yet` };
  if (n.p >= 0.5) return { cls: "soon", text: `${n.q50 !== null && n.q50 > t ? "expected " : ""}${eta(n, t)} · ${pct(n.p)} chance` };
  if (n.p >= 0.3) return { cls: "maybe", text: `possible · ${pct(n.p)} chance` };
  return { cls: "calm", text: `unlikely · ${pct(n.p)} chance` };
}

function byStatus(t) {
  return (a, b) => {
    const sa = serviceStatus(a, t).cls, sb = serviceStatus(b, t).cls;
    if (sa !== sb) return STATUS_ORDER[sa] - STATUS_ORDER[sb];
    if (sa === "hit") return a.actual - b.actual;
    if (sa === "soon") return (a.q50 ?? 999) - (b.q50 ?? 999);
    return b.p - a.p;
  };
}

/* layout of every service, computed once per incident. Traced calls and documented dependencies form
   the links; the component that holds the origin is drawn radially around it (one ring per hop). If the
   origin has (almost) no links, the largest component is drawn around its best-connected service instead.
   Services in no drawn component sit in a band below, so no service is ever left out. */
function graphLayout(P) {
  if (P.layout) return P.layout;
  const n = P.services.length, ri = P.services.indexOf(P.root);
  const key = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  const nbr = P.services.map(() => new Set()), traced = new Set(), links = [];
  const add = (a, b, kind) => {
    if (a === b || nbr[a].has(b)) return;
    nbr[a].add(b); nbr[b].add(a);
    if (kind === "traced") traced.add(key(a, b));
    links.push([a, b]);
  };
  P.edges.forEach(([a, b]) => add(a, b, "traced"));
  (P.doc_edges || []).forEach(([a, b]) => add(a, b, "doc"));

  const comp = new Array(n).fill(-1), comps = [];
  for (let s0 = 0; s0 < n; s0++) {
    if (comp[s0] >= 0) continue;
    const members = [], queue = [s0];
    comp[s0] = comps.length;
    while (queue.length) {
      const u = queue.shift();
      members.push(u);
      nbr[u].forEach((v) => { if (comp[v] < 0) { comp[v] = comps.length; queue.push(v); } });
    }
    comps.push(members);
  }
  const largest = comps.reduce((a, b) => (b.length > a.length ? b : a));
  const own = comps[comp[ri]];
  const main = own.length >= Math.min(5, largest.length) ? own : largest;
  const centre = main === own ? ri : main.reduce((a, b) => (nbr[b].size > nbr[a].size ? b : a));

  const depth = new Array(n).fill(null), parent = new Array(n).fill(null), kids = P.services.map(() => []);
  depth[centre] = 0;
  for (const queue = [centre]; queue.length;) {
    const u = queue.shift();
    [...nbr[u]].sort((a, b) => a - b).forEach((v) => {
      if (depth[v] === null) { depth[v] = depth[u] + 1; parent[v] = u; kids[u].push(v); queue.push(v); }
    });
  }
  const leaves = new Array(n).fill(1);
  const count = (u) => (leaves[u] = kids[u].length ? kids[u].reduce((sum, v) => sum + count(v), 0) : 1);
  count(centre);
  const maxDepth = Math.max(1, ...depth.filter((d) => d !== null));
  /* ring d sits at (d + 0.6) units, which leaves the crowded first ring more room around the centre */
  const ring = (d) => (d ? d + 0.6 : 0), units = ring(maxDepth);
  const W = 1300, ry = Math.min(85, 330 / units), rx = Math.min(200, (W / 2 - 150) / units);
  const radialH = Math.round(2 * (ry * units + 52)), cx = W / 2, cy = radialH / 2;
  const pos = new Array(n).fill(null);
  pos[centre] = [cx, cy];
  /* each subtree gets an angle proportional to its number of leaves */
  const place = (u, a0, a1) => {
    let a = a0;
    kids[u].forEach((v) => {
      const span = (a1 - a0) * leaves[v] / leaves[u], mid = a + span / 2;
      pos[v] = [cx + rx * ring(depth[v]) * Math.cos(mid), cy + ry * ring(depth[v]) * Math.sin(mid)];
      place(v, a, a + span);
      a += span;
    });
  };
  place(centre, -Math.PI / 2, 1.5 * Math.PI);

  /* the band: the origin's own small component first, then larger groups, each kept together */
  const name = (i) => shortName(P.services[i]);
  const rest = comps.filter((c) => c !== main)
    .sort((a, b) => (b.includes(ri) - a.includes(ri)) || (b.length - a.length) || name(a[0]).localeCompare(name(b[0])));
  const band = new Set(), STEP = 122, GAP = 26, ROW = 64;
  let x = 60, row = 0;
  const bandTop = radialH + 34;
  rest.forEach((group) => {
    const head = group.includes(ri) ? ri : group.reduce((a, b) => (nbr[b].size > nbr[a].size ? b : a));
    const ordered = [head, ...group.filter((g) => g !== head).sort((a, b) => name(a).localeCompare(name(b)))];
    if (x + STEP * (ordered.length - 1) > W - 60 && x > 60) { x = 60; row += 1; }
    ordered.forEach((g) => {
      if (x > W - 60) { x = 60; row += 1; }
      pos[g] = [x, bandTop + 14 + row * ROW];
      band.add(g);
      x += STEP;
    });
    x += GAP;
  });
  const H = band.size ? bandTop + 14 + row * ROW + 44 : radialH;
  return (P.layout = { pos, nbr, traced, links, depth, parent, maxDepth, ring, W, H, cx, cy, rx, ry, centre, band, bandTop, key });
}

/* seconds after the fault at which a service was (or is expected to be) affected; null if not expected */
function impactTime(n, t) {
  if (n.root) return 0;
  if (n.hit) return n.actual;
  return serviceStatus(n, t).cls === "soon" && n.q50 !== null ? n.q50 : null;
}

function renderPropGraph(nodes, t) {
  const P = app.prop, L = graphLayout(P);
  const box = $("prop-graph");
  box.innerHTML = "";
  const root = svg(L.W, L.H, "dep-graph");
  for (let d = 1; d <= L.maxDepth; d++) el("ellipse", { cx: L.cx, cy: L.cy, rx: L.rx * L.ring(d), ry: L.ry * L.ring(d), class: "g-ring" }, root);
  if (L.centre !== P.services.indexOf(P.root)) {
    text(root, 16, 22, `${shortName(P.root)} is not linked to the main call graph (shown below); the graph is centred on ${shortName(P.services[L.centre])}`, { class: "g-note" });
  }
  if (L.band.size) {
    el("line", { x1: 16, x2: L.W - 16, y1: L.bandTop - 8, y2: L.bandTop - 8, class: "g-band-rule" }, root);
    text(root, 16, L.bandTop + 2, "Not linked to the call graph above", { class: "g-note" });
  }
  L.links.forEach(([a, b]) => {
    /* traced calls on the shortest-path tree solid, other traced calls faint, documented-only links dashed */
    const tree = L.parent[a] === b || L.parent[b] === a;
    const kind = L.traced.has(L.key(a, b)) ? (tree ? "tree" : "extra") : "doc";
    el("line", { x1: L.pos[a][0], y1: L.pos[a][1], x2: L.pos[b][0], y2: L.pos[b][1], class: `g-edge ${kind}` }, root);
  });

  /* spread path: each affected (or expected) service is linked to the neighbour it most likely got it from,
     i.e. the latest neighbour that was affected before it; the label is the delay along that link */
  const times = nodes.map((n) => impactTime(n, t));
  const depthOf = (i) => L.depth[i] ?? 99;
  const paths = el("g", {}, root), pills = el("g", {}, root);
  nodes.forEach((n, v) => {
    if (n.root || times[v] === null) return;
    let from = null;
    L.nbr[v].forEach((u) => {
      if (times[u] === null || times[u] > times[v]) return;
      if (from === null || times[u] > times[from] || (times[u] === times[from] && depthOf(u) < depthOf(from))) from = u;
    });
    /* no affected neighbour: follow the call tree back to the first service affected before it */
    const route = [v];
    if (from === null) {
      for (let u = L.parent[v]; u !== null; u = L.parent[u]) {
        route.push(u);
        if (times[u] !== null && times[u] <= times[v]) { from = u; break; }
      }
    } else route.push(from);
    if (from === null) return;
    const actual = n.hit && (nodes[from].hit || nodes[from].root);
    for (let k = 0; k + 1 < route.length; k++) {
      const [a, b] = [L.pos[route[k]], L.pos[route[k + 1]]];
      el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: `g-edge ${actual ? "spread" : "expected"}` }, paths);
    }
    const delay = times[v] - times[from];
    if (delay === 0 || nodes[from].root || !actual) return;
    const [x1, y1] = L.pos[route[1]], [x2, y2] = L.pos[v];
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, label = `+${delay} s`, w = 10 + 6.4 * label.length;
    const pill = el("g", { class: "g-pill spread", "data-tip": `${shortName(nodes[from].service)} → ${shortName(n.service)}\naffected ${delay} s later` }, pills);
    el("rect", { x: mx - w / 2, y: my - 10, width: w, height: 20, rx: 10 }, pill);
    text(pill, mx, my + 4, label, { "text-anchor": "middle" });
  });

  /* nodes, with the name and the time of impact next to them; labels of quiet services are dropped where
     they would overlap another label (the name stays in the tooltip) */
  const layer = el("g", {}, root), boxes = [];
  const fits = (b) => !boxes.some((o) => b[0] < o[2] && o[0] < b[2] && b[1] < o[3] && o[1] < b[3]);
  const order = nodes.map((_, v) => v).sort((a, b) => (serviceStatus(nodes[a], t).cls === "calm") - (serviceStatus(nodes[b], t).cls === "calm"));
  order.forEach((v) => {
    const n = nodes[v];
    const st = serviceStatus(n, t), [x, y] = L.pos[v];
    const tip = `${n.service}\n${relation(n)[1]}\n${st.text}` + (n.hit ? `\npredicted at the start: ${pct(n.p0)} chance${n.q50_0 !== null ? `, ~${n.q50_0} s` : ""}` : "");
    const g = el("g", { class: `g-node ${st.cls}`, "data-tip": tip }, layer);
    if (n.hit && n.actual > t - P.bin_s) el("circle", { cx: x, cy: y, r: 10, class: "g-pulse" }, g);
    el("circle", { cx: x, cy: y, r: n.root ? 13 : st.cls === "calm" ? 6 : 9 }, g);
    const sub = n.root ? "fault origin" : st.cls === "hit" ? (n.actual === 0 ? "hit at start" : `hit at +${n.actual} s`)
      : st.cls === "soon" ? `${eta(n, t)} · ${pct(n.p)}` : st.cls === "late" ? `not hit · was ~+${n.q50} s` : st.cls === "maybe" ? pct(n.p) : "";
    let ax = x, ay = y + (n.root ? 30 : 23), anchor = "middle";
    if (!L.band.has(v) && v !== L.centre) {
      const dx = x - L.cx, dy = y - L.cy;
      if (Math.abs(dx) > 40) { ax = x + Math.sign(dx) * 14; ay = y + (sub ? -1 : 4); anchor = dx > 0 ? "start" : "end"; }
      else if (dy < 0) ay = y - (sub ? 28 : 14);
    }
    const label = shortName(n.service), w = 7 * Math.max(label.length, sub.length * 0.9) + 4;
    const x0 = anchor === "start" ? ax : anchor === "end" ? ax - w : ax - w / 2;
    const b = [x0, ay - 12, x0 + w, ay + (sub ? 18 : 4)];
    if (st.cls === "calm" && !fits(b)) return;
    boxes.push(b);
    text(g, ax, ay, label, { class: "g-name", "text-anchor": anchor });
    if (sub) text(g, ax, ay + 14, sub, { class: "g-sub", "text-anchor": anchor });
  });
  box.appendChild(root);
}

function renderPropagation() {
  const P = app.prop;
  if (!P) return;
  const t = Number($("prop-time").value);
  const nodes = propState(t);
  const forecastAt = forecastIndex(t) * P.bin_s;
  const rootName = shortName(P.root);
  $("prop-clock").textContent = `+${t} s`;
  $("prop-sub").textContent = `Fault at ${P.root} · ${P.fault_used === "unknown" ? "type unknown (the memory said it is new)" : nice(P.fault_used)}`;

  /* one-sentence story of the moment */
  const others = nodes.filter((n) => !n.root);
  const hit = others.filter((n) => n.hit).sort((a, b) => a.actual - b.actual);
  const soon = others.filter((n) => serviceStatus(n, t).cls === "soon").sort((a, b) => (a.q50 ?? 999) - (b.q50 ?? 999));
  const names = (list) => list.slice(0, 3).map((n) => `<b>${esc(shortName(n.service))}</b>`).join(", ") + (list.length > 3 ? ` and ${list.length - 3} more` : "");
  $("prop-story").innerHTML =
    `<div class="story-time">${t === 0 ? "The fault has just been confirmed" : `${t} seconds after the fault started`} at <b>${esc(rootName)}</b></div>` +
    `<div class="story-line"><span class="story-dot hit"></span>${hit.length ? `${hit.length} other service${hit.length === 1 ? " is" : "s are"} affected so far: ${names(hit)}.` : "No other microservice is affected yet."}</div>` +
    `<div class="story-line"><span class="story-dot soon"></span>${soon.length ? `Expected next: ${soon.slice(0, 3).map((n) => `<b>${esc(shortName(n.service))}</b> ${eta(n, t)}`).join(", ")}${soon.length > 3 ? ` and ${soon.length - 3} more` : ""}.` : "No further microservice is expected to be affected."}</div>`;
  const graph = app.propView === "graph";
  $("prop-key").innerHTML = [["root", "where it started"], ["hit", "affected"], ["soon", "expected (50 %+ chance)"], ["late", "predicted, not hit yet"], ["maybe", "possible (30–50 %)"], ["calm", "unlikely"]]
    .map(([c, l]) => `<span><i class="svc-dot ${c}"></i>${l}</span>`).join("") +
    (graph ? `<span><i class="key-line spread"></i>spread path · +s = delay after the previous service</span><span><i class="key-line expected"></i>expected path</span>` +
      `<span><i class="key-line traced"></i>traced call</span><span><i class="key-line doc"></i>documented link (not traced)</span>` : "");
  document.querySelectorAll("[data-pview]").forEach((b) => b.classList.toggle("active", b.dataset.pview === app.propView));
  $("prop-graph").hidden = !graph;
  $("prop-lanes").hidden = graph;
  if (graph) renderPropGraph(nodes, t);

  /* grouped list: every microservice as a box, grouped by its link to the broken service */
  const groups = new Map(LANES.map(([key]) => [key, []]));
  nodes.forEach((n) => groups.get(relation(n)[0]).push(n));
  $("prop-lanes").innerHTML = LANES.map(([key, title, hint]) => {
    let list = groups.get(key);
    let quiet = 0;
    /* in the outer lanes, services that are unlikely to be hit are only counted, unless "show all" is on */
    if (!app.propAll && ["up2", "down2", "side", "off"].includes(key)) {
      const shown = list.filter((n) => n.hit || n.p >= 0.3);
      quiet = list.length - shown.length;
      list = shown;
    }
    if (!list.length && !quiet) return "";
    const boxes = list.sort(byStatus(t)).map((n) => {
      const st = serviceStatus(n, t);
      const tip = `${n.service}\n${relation(n)[1]}\n${st.text}` + (n.hit ? `\npredicted at the start: ${pct(n.p0)} chance${n.q50_0 !== null ? `, ~${n.q50_0} s` : ""}` : "");
      return `<div class="svc ${st.cls}" data-tip="${esc(tip)}"><b>${esc(shortName(n.service))}</b><span>${esc(st.text)}</span></div>`;
    }).join("");
    const more = quiet ? `<div class="svc-more">+ ${quiet} more microservice${quiet === 1 ? "" : "s"}, unlikely to be affected</div>` : "";
    return `<div class="lane lane-${key}"><div class="lane-label"><b>${title}</b><span>${hint}</span></div><div class="lane-items">${boxes}${more}</div></div>`;
  }).join("");

  /* table: one row per microservice */
  const rows = others.slice().sort(byStatus(t));
  const shownRows = app.propAll ? rows : rows.slice(0, 12);
  $("prop-table").innerHTML = `<div class="ptable"><div class="prow phead"><span>Microservice</span><span>Link to the broken service</span><span>Chance of being affected</span><span>Expected</span><span>Actually affected</span><span>Now</span></div>` +
    shownRows.map((n) => {
      const st = serviceStatus(n, t);
      const p = n.hit ? n.p0 : n.p;
      const expected = n.hit ? (n.q50_0 !== null ? `predicted +${n.q50_0} s` : "not predicted") : (n.q50 !== null ? `+${n.q50} s` : "—");
      const actual = n.actual !== null && (n.actual <= t || app.showTruth) ? `+${n.actual} s` : (app.showTruth ? "not affected" : "—");
      return `<div class="prow"><span class="pname">${esc(shortName(n.service))}</span><span class="muted">${esc(relation(n)[1])}</span>` +
        `<span class="pchance"><span class="track"><i style="width:${Math.round(100 * p)}%"></i></span><span>${pct(p)}${n.hit ? " (at start)" : ""}</span></span>` +
        `<span>${esc(expected)}</span><span>${esc(actual)}</span><span><i class="svc-dot ${st.cls}"></i>${esc(st.cls === "soon" ? "expected soon" : st.cls === "late" ? "not hit yet" : st.cls === "maybe" ? "possible" : st.cls === "calm" ? "unlikely" : "affected")}</span></div>`;
    }).join("") + "</div>";
  $("prop-all").hidden = false;
  $("prop-all").textContent = app.propAll ? "Show fewer" : `Show all ${rows.length} services`;

  /* how good was the forecast made at the start? (end of the replay, or with ground truth on) */
  const scored = others.filter((n) => n.actual !== 0);
  const done = t >= Number($("prop-time").max) || app.showTruth;
  if (done) {
    const real = scored.filter((n) => n.actual !== null);
    const predicted = scored.filter((n) => n.p0 >= 0.5);
    const right = real.filter((n) => n.p0 >= 0.5);
    const missed = real.length - right.length, alarms = predicted.length - right.length;
    const errors = right.filter((n) => n.q50_0 !== null).map((n) => Math.abs(n.q50_0 - n.actual));
    $("prop-score").hidden = false;
    $("prop-score").innerHTML = `<b>Forecast check:</b> ` +
      (real.length ? `${real.length} microservice${real.length === 1 ? " was" : "s were"} really affected within ${P.horizon_s / 60} minutes; <b>${right.length}</b> of them ${right.length === 1 ? "was" : "were"} predicted in advance` +
        `${missed ? `, ${missed} missed` : ""}.` : `No other microservice was really affected within ${P.horizon_s / 60} minutes.`) +
      ` ${alarms} false alarm${alarms === 1 ? "" : "s"}.` +
      (errors.length ? ` Timing was off by about ${Math.round(errors.reduce((a, b) => a + b, 0) / errors.length)} s on average.` : "");
  } else {
    $("prop-score").hidden = true;
  }
  $("prop-note").textContent = `Predictions on screen were made at +${forecastAt} s using only the data seen up to then; ${app.propLive ? `they are refreshed every ${P.bin_s} s.` :
    "they stay fixed, so you can compare them with what really happens."} Time 0 is when the fault was confirmed at ${P.root} (${P.t0} s after it was injected). Actual times appear once the replay reaches them, or all at once with ground truth on.`;
}

function togglePropPlay() {
  if (app.propTimer) return stopPropPlay();
  stopPlay();
  pauseLive();
  $("prop-play").textContent = "❚❚ Pause";
  if (Number($("prop-time").value) >= Number($("prop-time").max)) $("prop-time").value = 0;
  app.propTimer = setInterval(() => {
    const slider = $("prop-time");
    if (Number(slider.value) >= Number(slider.max)) return stopPropPlay();
    slider.value = Number(slider.value) + Number(slider.step);
    renderPropagation();
  }, 900);
}
function stopPropPlay() {
  clearInterval(app.propTimer);
  app.propTimer = null;
  $("prop-play").textContent = "▶ Play";
}

/* ======================= LIVE MONITORING ======================= */
/* The recorded telemetry of each incident is streamed at a chosen speed. The incident opens when
   the first service's anomaly is confirmed, root cause and pattern follow once the analysis window
   has passed, and the blast radius then runs on the replay clock. */
const TICK_MS = 250, PRE_S = 150, ANALYSIS_S = 120;
const EMPTY = {};

function setMode(mode) {
  stopLive(); stopPlay(); stopPropPlay();
  app.mode = mode;
  document.body.classList.toggle("mode-live", mode === "live");
  document.body.classList.toggle("mode-step", mode === "step");
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("start").textContent = app.state ? (mode === "live" ? "Restart monitoring" : "Restart replay") : (mode === "live" ? "Start monitoring" : "Start replay");
  if (mode === "step" && !app.incident) ["incident", "verdict", "prop"].forEach((k) => { $(`${k}-empty`).innerHTML = EMPTY[k]; });
}

/* one continuous replay clock: it starts at the first recording's own time and runs on across recordings */
function liveClock(tau) {
  const L = app.live;
  return new Date((L.tele.inject_unix + L.shift + tau) * 1000).toLocaleTimeString([], { hour12: false });
}
function listNames(services) {
  const names = services.map((x) => `<b>${esc(shortName(x))}</b>`);
  return names.length > 3 ? `${names.slice(0, 3).join(", ")} and ${names.length - 3} more` : names.join(", ");
}
function feedPush(text, cls = "info") {
  const L = app.live;
  L.feed.unshift({ time: L.tele ? liveClock(L.tau) : "", text, cls, fresh: true });
  L.feed.length = Math.min(L.feed.length, 40);
  $("feed").innerHTML = L.feed.map((f) => `<li class="${f.cls}${f.fresh ? " new" : ""}"><time>${f.time}</time><span>${f.text}</span></li>`).join("");
  L.feed.forEach((f) => { f.fresh = false; });
}
function livePanels(message) {
  $("incident-body").hidden = true; $("incident-empty").hidden = false;
  $("verdict-body").hidden = true; $("verdict-empty").hidden = false;
  $("prop-body").hidden = true; $("prop-empty").hidden = false;
  ["incident", "verdict", "prop"].forEach((k) => { $(`${k}-empty`).innerHTML = `<p>${message}</p>`; });
  $("verdict-step").textContent = "";
}

async function startLive() {
  stopLive();
  app.live = { timer: null, feed: [], tele: null };
  $("feed").innerHTML = "";
  app.incident = null; app.prop = null;
  livePanels("No active incident. All services are within their normal range.");
  $("live-pause").disabled = false; $("live-skip").disabled = false;
  if (await loadRecording()) resumeLive();
}
function resumeLive() {
  const L = app.live;
  if (!L || L.done) return;
  clearInterval(L.timer);
  L.timer = setInterval(liveTick, TICK_MS);
  $("live-pause").textContent = "❚❚ Pause";
}
function pauseLive() {
  if (!app.live || !app.live.timer) return;
  clearInterval(app.live.timer);
  app.live.timer = null;
  $("live-pause").textContent = "▶ Resume";
}
function stopLive() {
  if (app.live) { clearInterval(app.live.timer); app.live.timer = null; }
  $("live-pause").textContent = "❚❚ Pause";
}

/* the next recording: normal traffic first, the incident later */
async function loadRecording() {
  const L = app.live;
  L.loading = true;
  const tele = await api("/api/live/peek", {});
  L.loading = false;
  if (tele.done) {
    L.done = true;
    stopLive();
    $("live-pause").disabled = true; $("live-skip").disabled = true;
    feedPush("All recorded incidents of this environment have been replayed.", "good");
    setStatus("Replay complete", "done");
    return false;
  }
  const n = tele.services.length;
  const shift = L.tele ? L.tele.inject_unix + L.shift + L.tau + 30 - (tele.inject_unix - PRE_S) : 0;
  Object.assign(L, {
    tele, shift, tau: -PRE_S, opened: false, analysing: false, analysed: false, closed: false, endAt: null, lastBin: null,
    seen: new Array(n).fill(false), first: [], incStep: null, pending: null,
    analysisAt: Math.max(ANALYSIS_S, tele.detect_s ?? ANALYSIS_S),
    worst: tele.t.map((_, k) => Math.max(0, ...tele.ratio.map((r) => r[k] ?? 0))),
  });
  buildTiles();
  feedPush(`Recording ${tele.recording} of ${tele.total}: monitoring ${n} services`, "info");
  setStatus(`Monitoring · ${systemName()}`, "live");
  renderLive();
  return true;
}

function liveTick() {
  const L = app.live;
  if (!L || !L.tele || L.loading) return;
  L.tau = Math.min(L.tau + (Number($("speed").value) * TICK_MS) / 1000, L.tele.end_s);
  advanceLive();
}

function advanceLive() {
  const L = app.live, T = L.tele;
  const fresh = [];
  T.confirmed.forEach((c, i) => { if (c !== null && c <= L.tau && !L.seen[i]) { L.seen[i] = true; fresh.push(i); } });
  if (fresh.length) {
    const names = fresh.map((i) => T.services[i]);
    if (!L.opened) openIncident(names);
    else feedPush(`${listNames(names)} ${names.length === 1 ? "is" : "are"} now anomalous`, "warn");
  }
  if (!L.analysing && !L.analysed && L.tau >= L.analysisAt) {
    if (!L.opened) openIncident([]);
    L.analysing = true;
    L.pending = analyseLive().catch((e) => { L.analysing = false; feedPush(`Analysis failed: ${esc(e.message)}`, "crit"); });
  }
  if (L.analysed) followPropagation();
  if (L.analysed && !L.closed && L.tau >= L.endAt) closeIncident();
  if (!L.analysed && !L.analysing && L.tau >= T.end_s) closeIncident();
  renderLive();
}

function openIncident(services) {
  const L = app.live;
  L.opened = true;
  L.first = services;
  L.incStep = (app.state ? app.state.position : 0) + 1;
  stopPropPlay();
  app.incident = null; app.prop = null;
  livePanels("Collecting telemetry for root-cause analysis…");
  $("incident-id").textContent = incidentId(L.incStep);
  feedPush(`<b>${incidentId(L.incStep)}</b> opened: anomaly on ${services.length ? listNames(services) : "the monitored services"}`, "crit");
  setStatus(`Incident open · ${incidentId(L.incStep)}`, "alert");
}

async function analyseLive() {
  const L = app.live;
  const { incident, state } = await api("/api/next", {});
  if (app.live !== L) return;  /* restarted while the analysis was running */
  app.state = state;
  if (!incident) { L.analysing = false; return; }
  app.incident = incident;
  renderState();
  renderIncident();
  await loadPropagation();
  if (app.live !== L) return;
  const P = app.prop;
  if (P) app.propBase = Math.max(0, Math.min(Math.floor((L.analysisAt - P.t0) / P.bin_s), P.forecasts.length - 1));
  L.analysed = true; L.analysing = false;
  L.endAt = Math.min(L.tele.end_s, Math.max(L.analysisAt + 60, P ? P.t0 + P.horizon_s : L.analysisAt + 120));
  followPropagation(true);
  const learned = incident.ranking.learned;
  const root = incident.anchor || (learned || incident.ranking.metric_shift)[0].service;
  const score = learned ? (learned.find((r) => r.service === root) || learned[0]).score : null;
  feedPush(`Root cause: <b>${esc(root)}</b>${score !== null ? ` (${pct(score)} confidence)` : ""} · ${incident.verdict === "NOVEL" ? "<b>new failure pattern</b>, held for review" : `matches <b>${esc(nice(incident.verdict))}</b>`}`, "good");
  if (P) {
    const t = Number($("prop-time").value);
    const soon = propState(t).filter((n) => !n.root && serviceStatus(n, t).cls === "soon").sort((a, b) => (a.q50 ?? 999) - (b.q50 ?? 999));
    feedPush(`Blast radius: ${soon.length ? `${soon.length} service${soon.length === 1 ? "" : "s"} at risk, next ${listNames([soon[0].service])} ${eta(soon[0], t)}` : "no further impact expected"}`, "info");
  }
}

/* the blast radius runs on the replay clock (unless someone is scrubbing or playing it by hand) */
function followPropagation(force = false) {
  const L = app.live, P = app.prop;
  if (!P || app.propTimer) return;
  const t = Math.max(0, Math.min(Math.floor((L.tau - P.t0) / P.bin_s) * P.bin_s, Number($("prop-time").max)));
  if (!force && t === L.lastBin) return;
  L.lastBin = t;
  $("prop-time").value = t;
  renderPropagation();
  renderSituation();
  if (app.incident) lineChart($("series"), clipSeries(app.incident.series));
}

function closeIncident() {
  const L = app.live;
  L.closed = true;
  const affected = L.seen.filter(Boolean).length;
  feedPush(`<b>${incidentId(L.incStep)}</b> closed after the 5-minute forecast window · ${affected} service${affected === 1 ? "" : "s"} affected in total`, "info");
  setStatus(`Monitoring · ${systemName()}`, "live");
  if (app.incident) renderSituation();
  L.loading = true;
  setTimeout(() => { if (app.live === L) loadRecording(); }, 2500);
}

/* presenter shortcut: jump to the alert, then to the analysis, then to the end of the incident */
async function skipLive() {
  const L = app.live;
  if (!L || !L.tele || L.loading || L.analysing) return;
  const alert = L.tele.detect_s ?? L.analysisAt;
  if (!L.opened) L.tau = L.tau < alert - 10 ? alert - 10 : alert;
  else if (!L.analysed) L.tau = Math.max(L.tau, L.analysisAt);
  else if (!L.closed) L.tau = Math.max(L.tau, L.endAt);
  advanceLive();
  if (L.pending) await L.pending;
}

/* ---------- live views ---------- */
function sampleIndex(T, tau) {
  let lo = 0, hi = T.t.length - 1;
  if (tau <= T.t[0]) return 0;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (T.t[mid] <= tau) lo = mid; else hi = mid - 1; }
  return lo;
}

function buildTiles() {
  const L = app.live, T = L.tele;
  const order = T.services.map((_, i) => i).sort((a, b) => shortName(T.services[a]).localeCompare(shortName(T.services[b])));
  $("tiles").innerHTML = order.map((i) => `<div class="tile ok" data-i="${i}"><b>${esc(shortName(T.services[i]))}</b><i></i></div>`).join("");
  L.tiles = [];
  $("tiles").querySelectorAll(".tile").forEach((tile) => { L.tiles[Number(tile.dataset.i)] = { el: tile, bar: tile.querySelector("i"), cls: "ok" }; });
}

function renderLive() {
  const L = app.live;
  if (!L || !L.tele) return;
  const T = L.tele, k = sampleIndex(T, L.tau);
  let anomalous = 0;
  T.services.forEach((service, i) => {
    const tile = L.tiles[i], r = T.ratio[i][k];
    const bad = T.confirmed[i] !== null && T.confirmed[i] <= L.tau;
    anomalous += bad;
    const cls = r === null ? "nodata" : bad ? "bad" : r >= 0.8 ? "warm" : "ok";
    if (tile.cls !== cls) { tile.el.className = `tile ${cls}`; tile.cls = cls; }
    tile.bar.style.width = `${Math.min(100, ((r ?? 0) / 1.5) * 100)}%`;
    tile.el.dataset.tip = `${service}\nanomaly level ${r === null ? "no data" : fmt(r)} (1.0 = alert)` +
      (bad ? `\nanomalous since ${liveClock(T.confirmed[i])}` : "");
  });
  $("health-sub").textContent = `${liveClock(L.tau)} · ${$("speed").value}× replay · ${T.services.length} services · ` +
    `${anomalous ? `${anomalous} anomalous` : "all healthy"}`;
  healthChart(L, k);
  if (L.opened && !L.analysed) renderLiveHeader();
}

function healthChart(L, k) {
  const T = L.tele, box = $("health-chart");
  const W = Math.max(300, box.clientWidth || 700), H = 120, Lp = 34, R = 10, Tp = 8, B = 20;
  const x1 = Math.max(L.tau, T.t[0] + 60), x0 = x1 - 300;
  const pts = [];
  for (let j = 0; j <= k; j++) if (T.t[j] >= x0) pts.push([T.t[j], Math.min(L.worst[j], 8)]);
  const top = Math.max(1.5, ...pts.map((p) => p[1])) * 1.1;
  const x = (t) => Lp + ((t - x0) / (x1 - x0)) * (W - Lp - R);
  const y = (v) => Tp + (1 - v / top) * (H - Tp - B);
  const root = svg(W, H);
  [0, 1, Math.floor(top)].filter((v, i, a) => a.indexOf(v) === i && v <= top).forEach((v) => {
    el("line", { x1: Lp, x2: W - R, y1: y(v), y2: y(v), class: "grid-line" }, root);
    text(root, Lp - 6, y(v) + 3, v, { class: "axis-text", "text-anchor": "end" });
  });
  el("line", { x1: Lp, x2: W - R, y1: y(1), y2: y(1), stroke: "var(--critical)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, root);
  text(root, W - R - 4, y(1) - 5, "alert threshold", { class: "axis-text", fill: "var(--critical)", "text-anchor": "end" });
  if (pts.length > 1) {
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
    el("path", { d: `${line} L${x(pts[pts.length - 1][0]).toFixed(1)},${y(0)} L${x(pts[0][0]).toFixed(1)},${y(0)} Z`, fill: "var(--accent)", "fill-opacity": 0.1 }, root);
    el("path", { d: line, fill: "none", stroke: "var(--accent)", "stroke-width": 2, "stroke-linejoin": "round" }, root);
  }
  if (L.opened && T.detect_s !== null && T.detect_s >= x0) {
    el("line", { x1: x(T.detect_s), x2: x(T.detect_s), y1: Tp, y2: H - B, stroke: "var(--critical)", "stroke-width": 1.5 }, root);
    text(root, x(T.detect_s) - 5, H - B - 6, `${incidentId(L.incStep)} opened`, { class: "axis-text", fill: "var(--critical)", "text-anchor": "end" });
  }
  [x0, (x0 + x1) / 2, x1].forEach((t, i) => text(root, x(t), H - 5, liveClock(Math.round(t)), { class: "axis-text", "text-anchor": ["start", "middle", "end"][i] }));
  box.replaceChildren(root);
}

/* incident header while the analysis window is still filling */
function renderLiveHeader() {
  const L = app.live;
  const affected = L.seen.filter(Boolean).length;
  const start = L.tele.detect_s ?? L.tau, done = Math.min(1, Math.max(0, (L.tau - start) / Math.max(1, L.analysisAt - start)));
  $("situation").hidden = false;
  $("ih-id").textContent = incidentId(L.incStep);
  $("ih-title").textContent = `Anomaly detected on ${L.first.length ? L.first.map(shortName).slice(0, 2).join(", ") : "the monitored services"}`;
  $("ih-badges").innerHTML = `<span class="badge-pill crit">Open</span><span class="badge-pill accent">Analysing · ${Math.round(100 * done)}%</span>`;
  $("ih-root").textContent = "Analysing…";
  $("ih-root-sub").textContent = `collecting telemetry · ${Math.max(0, Math.ceil(L.analysisAt - L.tau))} s of data to go`;
  $("ih-diag").textContent = "Analysing…";
  $("ih-diag-sub").textContent = "pattern match runs with the root-cause analysis";
  $("ih-blast").textContent = `${affected} service${affected === 1 ? "" : "s"} anomalous`;
  $("ih-blast-sub").textContent = "so far · forecast follows the analysis";
}

/* ======================= EVIDENCE ======================= */
const SYSTEM_NAMES = { ob: "Online Boutique", ss: "Sock Shop", tt: "Train Ticket" };
const SYSTEM_SHORT = { ob: "Online Boutique (12 svc)", ss: "Sock Shop", tt: "Train Ticket (~70 svc)" };

function groupedColumns(container, groups, series, { domain = [0, 1], format = (v) => fmt(v), refLine = null, refLabel = "", labelSeries = null } = {}) {
  const W = Math.max(320, container.clientWidth || 560), H = 250, L = 34, R = 8, T = 16, B = 30;
  const y = (v) => T + (1 - (v - domain[0]) / (domain[1] - domain[0])) * (H - T - B);
  const root = svg(W, H);
  for (let k = 0; k <= 4; k++) {
    const v = domain[0] + ((domain[1] - domain[0]) * k) / 4;
    el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid-line" }, root);
    text(root, L - 6, y(v) + 3, format(v), { class: "axis-text", "text-anchor": "end" });
  }
  const gw = (W - L - R) / groups.length, bw = Math.min(26, (gw - 24) / series.length - 2);
  groups.forEach((g, gi) => {
    const start = L + gi * gw + (gw - series.length * (bw + 2)) / 2;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (v === null || v === undefined || Number.isNaN(v)) return;
      const x = start + si * (bw + 2);
      el("path", { d: columnPath(x, y(domain[0]), y(Math.max(domain[0], v)), bw), fill: s.color, "data-tip": `${g.label}\n${s.name}: ${format(v)}` }, root);
      if (labelSeries === s.name) text(root, x + bw / 2, y(v) - 5, format(v), { class: "label-text", "text-anchor": "middle", "font-weight": 600 });
    });
    text(root, L + gi * gw + gw / 2, H - 10, g.label, { class: "axis-text", "text-anchor": "middle" });
  });
  if (refLine !== null) {
    el("line", { x1: L, x2: W - R, y1: y(refLine), y2: y(refLine), stroke: "var(--ink-3)", "stroke-dasharray": "4 4", "stroke-width": 1 }, root);
    text(root, W - R, y(refLine) - 4, refLabel, { class: "axis-text", "text-anchor": "end" });
  }
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = series.map((s) => `<span><span class="swatch" style="background:${s.color}"></span>${esc(s.name)}</span>`).join("");
  container.replaceChildren(legend, root);
}

function tableHtml(columns, rows) {
  return `<table><thead><tr>${columns.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${columns.map((c) => `<td class="${c.num ? "num" : ""}">${esc(c.get(r))}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
}

function evidenceCard(title, takeaway) {
  const card = document.createElement("article");
  card.className = "card ev-card";
  card.innerHTML = `<h2>${esc(title)}</h2><p class="takeaway">${takeaway}</p><div class="chart"></div>`;
  $("evidence").appendChild(card);
  return card;
}

function pendingCard(title, what) {
  const card = evidenceCard(title, `<span class="pending">${esc(what)}</span>`);
  card.querySelector(".chart").remove();
}

/* categorical slots in fixed order (validated adjacent pairs); Mem-RCA always slot 1 */
const METHOD_COLORS = { "Mem-RCA": "var(--s1)", anchor: "var(--s2)", "global max": "var(--s3)", "global mean": "var(--s4)" };

async function loadEvidence() {
  const ev = await api("/api/evidence");
  $("evidence").replaceChildren();
  const systems = ["ob", "ss", "tt"];
  const groups = systems.map((s) => ({ key: s, label: SYSTEM_SHORT[s] }));

  const roleSeries = (rows, mapping, metric = "auroc") => Object.entries(mapping).map(([name, role]) => ({
    name, color: METHOD_COLORS[name], values: systems.map((s) => (rows.find((r) => r.system === s && r.role === role) || {})[metric] ?? null),
  }));

  if (ev.re1) {
    const series = roleSeries(ev.re1, { "Mem-RCA": "Mem-RCA", anchor: "baseline anchor", "global max": "baseline global_max", "global mean": "baseline global_mean" });
    const mem = series[0].values, best = systems.map((_, i) => Math.max(series[1].values[i], series[2].values[i], series[3].values[i]));
    const card = evidenceCard("RE1 · fault types held out on an independent suite",
      `Confirmatory. Five leave-one-fault-type-out folds per system, learned root localiser retrained per fold. Mem-RCA novelty AUROC ${systems.map((s, i) => `${SYSTEM_NAMES[s]} ${fmt(mem[i])}`).join(", ")} (best fixed baseline ${best.map((b) => fmt(b)).join(" / ")}).` +
      (ev.hypotheses ? ` <b>H1</b> (root anchoring beats global pooling on Train Ticket): ${ev.hypotheses.h1 ? "supported" : "not supported"}. <b>H2</b> (Mem-RCA beats every fixed fingerprint on average): ${ev.hypotheses.h2 ? "supported" : "not supported"}.` : ""));
    groupedColumns(card.querySelector(".chart"), groups, series, { refLine: 0.5, refLabel: "chance", labelSeries: "Mem-RCA" });
    card.insertAdjacentHTML("beforeend", `<details><summary>Table view</summary>${tableHtml([
      { label: "System", get: (r) => SYSTEM_NAMES[r.system] }, { label: "Method", get: (r) => r.role },
      { label: "AUROC", num: true, get: (r) => fmt(r.auroc) }, { label: "OSCR", num: true, get: (r) => fmt(r.oscr) },
      { label: "Known-type acc.", num: true, get: (r) => pct(r.closed_set_acc) }, { label: "Unseen flagged", num: true, get: (r) => pct(r.unseen_flagged_novel) },
    ], ev.re1)}</details>`);
  } else pendingCard("RE1 · fault types held out on an independent suite", "Not run yet: python scripts/train_external_localisers.py, then python scripts/evaluate_external.py");

  if (ev.re3) {
    const series = roleSeries(ev.re3, { "Mem-RCA": "Mem-RCA", anchor: "baseline anchor", "global max": "baseline global_max", "global mean": "baseline global_mean" });
    const rows = ev.re3.filter((r) => r.role === "Mem-RCA");
    const card = evidenceCard("RE3 · code-level faults the memory has never seen",
      `Confirmatory. A memory built from RE2 resource faults rates RE3 code-level incidents as unfamiliar: AUROC ${rows.map((r) => `${SYSTEM_NAMES[r.system]} ${fmt(r.auroc)}`).join(", ")}, ` +
      `with ${rows.map((r) => pct(r.unseen_flagged_novel)).join(" / ")} of them flagged NOVEL while ${rows.map((r) => pct(r.known_accepted)).join(" / ")} of known incidents are accepted. Averaging over services (global mean) gets close to chance.`);
    groupedColumns(card.querySelector(".chart"), groups, series, { refLine: 0.5, refLabel: "chance", labelSeries: "Mem-RCA" });
    card.insertAdjacentHTML("beforeend", `<details><summary>Table view</summary>${tableHtml([
      { label: "System", get: (r) => SYSTEM_NAMES[r.system] }, { label: "Method", get: (r) => r.role },
      { label: "AUROC", num: true, get: (r) => fmt(r.auroc) }, { label: "Flagged NOVEL", num: true, get: (r) => pct(r.unseen_flagged_novel) },
      { label: "Known kept", num: true, get: (r) => pct(r.known_accepted) }, { label: "Mann–Whitney p", num: true, get: (r) => Number(r.mann_whitney_p).toExponential(1) },
    ], ev.re3)}</details>`);
  }

  if (ev.re2_memrca) {
    const series = roleSeries(ev.re2_memrca, { "Mem-RCA": "Mem-RCA", anchor: "E2 anchor", "global max": "E2 global_max", "global mean": "E2 global_mean" });
    const card = evidenceCard("RE2 · two fault types arrive unannounced (development)",
      "Mean over 3 seeds × 3 folds. Averaging over services (global mean) hides the root's signal on Train Ticket. Anchoring on the root fixes this, and Mem-RCA picks the view per system on validation incidents. Optimistic: RE2 was used to design the method.");
    groupedColumns(card.querySelector(".chart"), groups, series, { refLine: 0.5, refLabel: "chance", labelSeries: "Mem-RCA" });
    card.insertAdjacentHTML("beforeend", `<details><summary>Table view</summary>${tableHtml([
      { label: "System", get: (r) => SYSTEM_NAMES[r.system] }, { label: "Method", get: (r) => r.role }, { label: "Configuration", get: (r) => r.config },
      { label: "AUROC", num: true, get: (r) => fmt(r.auroc) }, { label: "OSCR", num: true, get: (r) => fmt(r.oscr) },
      { label: "Unseen flagged", num: true, get: (r) => pct(r.unseen_flagged_novel) },
    ], ev.re2_memrca)}</details>`);
  }

  if (ev.shortcut) {
    const methods = { "Learned localiser": "root_head", "Metric shift, 5 training roots": "shift_training_roots_only", "Metric shift, all services": "shift_all_services", "No telemetry (prior)": "no_telemetry_prior" };
    const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
    const series = Object.entries(methods).map(([name, m], i) => ({ name, color: colors[i], values: systems.map((s) => (ev.shortcut.find((r) => r.system === s && r.method === m) || {}).hit_at_1 ?? null) }));
    const card = evidenceCard("Root-cause ranking and the benchmark shortcut (C4)",
      "Hit@1 on RE2 test incidents. RE2 has only five root services per system. A rule with no learning that only looks at those five nearly matches the learned localiser on Train Ticket, so a high Hit@1 here does not by itself show real localisation skill.");
    groupedColumns(card.querySelector(".chart"), groups, series, { format: (v) => fmt(v), labelSeries: "Learned localiser" });
  }

  for (const [suite, title, note] of [
    ["re3", "Time-to-impact forecast on RE3 (confirmatory)", "Trained on RE2, tested once on RE3 code-level faults with the fault type unknown (protocol: docs/PROPAGATION_PROTOCOL.md)."],
    ["re2", "Time-to-impact forecast on RE2 (development)", "Each repetition forecast by a model trained on the other two."]]) {
    const rows = ev[`propagation_${suite}`];
    if (!rows) continue;
    const split = suite === "re2" ? "rep-CV" : "RE3 (confirmatory)";
    const pick = (m, s, key) => (rows.find((r) => r.split === split && r.method === m && r.system === s) || {})[key] ?? null;
    const methods = { PropagationNet: "PropagationNet", memory: "memory", "graph table": "graph table", "all at once": "all at once" };
    const series = Object.entries(methods).map(([name, m], i) => ({ name, color: `var(${SERIES[i]})`, values: systems.map((s) => pick(m, s, "auroc")) }));
    const gain = (ev[`propagation_gain_${suite}`] || []).find((g) => g.system === "all" && g.update_at_s === 60);
    const card = evidenceCard(title, `${note} AUROC of “will this service be hit within 300 s?”, forecast at the root's onset. ` +
      `Pooled: PropagationNet ${fmt(pick("PropagationNet", "all", "auroc"))}, graph table ${fmt(pick("graph table", "all", "auroc"))}, memory ${fmt(pick("memory", "all", "auroc"))}. ` +
      `Order of impact (C-index): PropagationNet ${fmt(pick("PropagationNet", "all", "c_index"))} vs graph table ${fmt(pick("graph table", "all", "c_index"))}.` +
      (gain ? ` Updating the forecast 60 s later lifts AUROC on the services still not hit from ${fmt(gain.auroc_t0_forecast)} to ${fmt(gain.auroc_updated)}.` : ""));
    groupedColumns(card.querySelector(".chart"), groups, series, { refLine: 0.5, refLabel: "chance", labelSeries: "PropagationNet" });
    card.insertAdjacentHTML("beforeend", `<details><summary>Table view</summary>${tableHtml([
      { label: "System", get: (r) => SYSTEM_NAMES[r.system] || "pooled" }, { label: "Method", get: (r) => r.method },
      { label: "AUROC", num: true, get: (r) => fmt(r.auroc) }, { label: "C-index", num: true, get: (r) => fmt(r.c_index) },
      { label: "F1", num: true, get: (r) => fmt(r.f1) }, { label: "Onset MAE (s)", num: true, get: (r) => fmt(r.mae_s, 0) },
    ], rows.filter((r) => r.split === split))}</details>`);
  }

  if (ev.service_holdout) {
    const methods = { "Learned localiser": "learned_head", "Training roots first": "shift_training_roots_first", "Metric shift, all services": "shift_all_services", "No telemetry (prior)": "no_telemetry_prior" };
    const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
    const unseen = ev.service_holdout.filter((r) => r.status === "unseen_root");
    const series = Object.entries(methods).map(([name, m], i) => ({ name, color: colors[i], values: systems.map((s) => (unseen.find((r) => r.system === s && r.method === m) || {}).hit_at_1 ?? null) }));
    const card = evidenceCard("Service hold-out: roots never seen as roots in training",
      "Hit@1 when the root service was excluded from training (mean over the five held-out services). Candidate-based shortcuts score zero by construction. This is the honest measure of localisation on new services.");
    groupedColumns(card.querySelector(".chart"), groups, series, { format: (v) => fmt(v), labelSeries: "Learned localiser" });
    card.insertAdjacentHTML("beforeend", `<details><summary>Table view (seen and unseen roots)</summary>${tableHtml([
      { label: "System", get: (r) => SYSTEM_NAMES[r.system] }, { label: "Root", get: (r) => r.status.replace("_", " ") },
      { label: "Method", get: (r) => r.method }, { label: "Hit@1", num: true, get: (r) => fmt(r.hit_at_1) },
    ], ev.service_holdout)}</details>`);
  } else pendingCard("Service hold-out: roots never seen as roots in training", "Not run yet: python scripts/run_service_holdout.py");
}

/* ======================= METHOD ======================= */
function renderPipeline() {
  const steps = [
    ["Telemetry", "Per-service metrics (CPU, memory, disk, sockets, load, latency, errors) around the alert."],
    ["Localise the root", "LSTM encoders + directed graph attention score every service as the likely root."],
    ["Root-anchored fingerprint", "The root's own metric shifts plus the symptoms it caused elsewhere, as one vector."],
    ["Incident memory", "Similarity to stored incidents of each known type. Below τ, the incident is NOVEL."],
    ["Discover & enrol", "Similar NOVEL incidents are grouped, named by an engineer and added to memory. No retraining."],
    ["Forecast impact", "PropagationNet (graph attention + hazard head) predicts which services the fault reaches next, and when."],
  ];
  $("pipeline").innerHTML = steps.map(([title, body], i) => `<div class="step"><div class="num">STEP ${i + 1}</div><b>${title}</b><p>${body}</p></div>`).join("");
}

/* ======================= wiring ======================= */
function switchView(view) {
  document.querySelectorAll(".tab").forEach((t) => { const on = t.dataset.view === view; t.classList.toggle("active", on); t.setAttribute("aria-selected", on); });
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  if (view === "evidence") loadEvidence().catch((e) => { $("evidence").innerHTML = `<p class="pending">${esc(e.message)}</p>`; });
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("memrca-theme"); } catch (_) { /* storage may be unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("memrca-theme", next); } catch (_) { /* ignore */ }
    if (app.state) renderState();
    if (app.incident) renderIncident();
    renderLive();
  });
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
$("scenario").addEventListener("change", describeScenario);
$("start").addEventListener("click", async () => {
  stopLive();
  await startScenario();
  if (app.mode === "live" && app.state) await startLive();
});
document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => { if (b.dataset.mode !== app.mode) setMode(b.dataset.mode); }));
$("live-pause").addEventListener("click", () => (app.live && app.live.timer ? pauseLive() : resumeLive()));
$("live-skip").addEventListener("click", skipLive);
$("speed").addEventListener("change", renderLive);
$("next").addEventListener("click", nextIncident);
$("play").addEventListener("click", togglePlay);
$("prop-play").addEventListener("click", togglePropPlay);
$("prop-time").addEventListener("input", () => { stopPropPlay(); stopPlay(); pauseLive(); renderPropagation(); renderSituation(); });
$("prop-live").addEventListener("change", (e) => { app.propLive = e.target.checked; renderPropagation(); renderSituation(); });
$("prop-all").addEventListener("click", () => { app.propAll = !app.propAll; renderPropagation(); });
document.querySelectorAll("[data-pview]").forEach((b) => b.addEventListener("click", () => { app.propView = b.dataset.pview; renderPropagation(); }));
$("truth").addEventListener("change", (e) => {
  app.showTruth = e.target.checked;
  if (app.state) renderState();
  if (app.incident) renderIncident();
  if (app.prop) renderPropagation();
});
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (app.state) renderMap();
    if (app.incident) renderIncident();
    if (app.prop) renderPropagation();
    renderLive();
    if ($("view-evidence").classList.contains("active")) loadEvidence();
  }, 200);
});
/* demo links: ?scenario=re2:tt:fold1&steps=20&enrol=1&truth=1&view=evidence */
async function autostart() {
  const q = new URLSearchParams(location.search);
  if (q.get("theme")) document.documentElement.dataset.theme = q.get("theme");
  if (q.get("view")) switchView(q.get("view"));
  if (!q.get("scenario")) return;
  if (!q.get("view")) switchView("console");
  $("scenario").value = q.get("scenario");
  describeScenario();
  if (q.get("live")) {
    if (q.get("speed")) $("speed").value = q.get("speed");
    await startScenario();
    await startLive();
    for (let i = 0; i < Number(q.get("skip") || 0); i++) await skipLive();
    if (q.get("pause")) pauseLive();
    return;
  }
  setMode("step");
  await startScenario();
  for (let i = 0; i < Number(q.get("steps") || 0); i++) {
    await nextIncident();
    if (q.get("enrol") && app.state.candidates.length) {
      const c = app.state.candidates[0];
      await enrol(c.id, `${c.signature[0]}-fault`);
    }
  }
  if (q.get("truth")) { $("truth").checked = true; $("truth").dispatchEvent(new Event("change")); }
  if (q.get("plive")) { $("prop-live").checked = true; app.propLive = true; }
  if (q.get("ptime")) {
    await loadPropagation();
    $("prop-time").value = q.get("ptime");
    renderPropagation();
  }
}

/* ======================= OVERVIEW ======================= */
async function loadOverview() {
  let ev;
  try { ev = await api("/api/evidence"); } catch (_) { $("bignums").innerHTML = ""; return; }
  const cards = [];
  const tt = (ev.re3 || []).find((r) => r.system === "tt" && r.role === "Mem-RCA");
  if (tt) {
    cards.push([fmt(tt.auroc), "Spots failures it has never seen", "Code bugs that were never in memory, on the 69-service app. Score: 0.5 = guessing, 1 = perfect."]);
    cards.push([pct(tt.unseen_flagged_novel), "of brand-new failures caught", `flagged as new, while ${pct(tt.known_accepted)} of familiar failures are still recognised.`]);
  }
  const prop = ev.propagation_re3 || [];
  const pn = prop.find((r) => r.method === "PropagationNet" && r.system === "all");
  const gt = prop.find((r) => r.method === "graph table" && r.system === "all");
  if (pn && gt) cards.push([fmt(pn.c_index), "Predicts who is hit first", `how well it orders the next services to fail, against ${fmt(gt.c_index)} for a rule of thumb (0.5 = guessing).`]);
  const hold = ev.service_holdout || [];
  const mean = (status, method) => {
    const rows = hold.filter((r) => r.status === status && r.method === method);
    return rows.length ? rows.reduce((a, r) => a + r.hit_at_1, 0) / rows.length : null;
  };
  const seen = mean("seen_root", "learned_head"), unseen = mean("unseen_root", "learned_head");
  if (seen !== null && unseen !== null) {
    cards.push([`${pct(seen)} → ${pct(unseen)}`, "Honest finding: a hidden shortcut", "A standard deep root-cause model's accuracy when the faulty service was never a root in training. We test for this and report it.", true]);
  }
  $("bignums").innerHTML = cards.map(([num, what, how, warn]) =>
    `<div class="bignum"><div class="num ${warn ? "warnc" : ""}">${esc(num)}</div><div class="what">${esc(what)}</div><div class="how">${esc(how)}</div></div>`).join("");
}

["incident", "verdict", "prop"].forEach((k) => { EMPTY[k] = $(`${k}-empty`).innerHTML; });
initTheme();
loadOverview();
renderPipeline();
loadScenarios().then(autostart).catch((e) => { $("scenario-desc").textContent = `Could not reach the console server: ${e.message}`; });
