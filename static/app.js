"use strict";

// ------------------------------------------------------------------- util --

const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtTs = (t) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric",
  hour: "2-digit", minute: "2-digit" });
const fmtMin = (m) => m == null ? "—" : m < 60 ? `${Math.round(m)}m`
  : m < 1440 ? `${(m / 60).toFixed(1)}h` : `${(m / 1440).toFixed(1)}d`;
const ago = (t) => { const s = Date.now() / 1000 - t;
  return s < 60 ? `${Math.round(s)}s ago` : fmtMin(s / 60) + " ago"; };

/** A 404 body is still valid JSON, so an unchecked fetch paints its fields as
    "undefined" instead of failing. Throw on the status rather than the parse. */
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status} ${r.statusText}`);
  return r.json();
}

/** "NVIDIA RTX PRO 6000 Blackwell Server Edition" -> "RTX PRO 6000". The full
    name goes nowhere useful in a table column that narrow. */
function shortGpu(name) {
  if (!name) return "";
  return String(name).replace(/^NVIDIA\s+/, "")
    .replace(/\s+(Server Edition|Workstation Edition|Blackwell)\b/g, "")
    .trim();
}

function utilClass(p) {
  if (p == null) return "faint";
  return p >= 60 ? "good" : p >= 20 ? "warn" : "bad";
}

function table(cols, rows, opts = {}) {
  const t = el("table"), thead = el("thead"), tr = el("tr");
  cols.forEach(c => { const th = el("th", c.num ? "num" : null, c.label); tr.appendChild(th); });
  thead.appendChild(tr); t.appendChild(thead);
  const tb = el("tbody");
  rows.forEach(r => {
    const row = el("tr", opts.rowClass ? opts.rowClass(r) : null);
    cols.forEach(c => {
      const td = el("td", [c.num ? "num" : "", c.cls ? c.cls(r) : ""].join(" ").trim());
      const v = c.get(r);
      if (v instanceof Node) td.appendChild(v); else td.innerHTML = v ?? "";
      row.appendChild(td);
    });
    if (opts.onClick) { row.classList.add("clickable"); row.onclick = () => opts.onClick(r); }
    tb.appendChild(row);
  });
  t.appendChild(tb);
  return t;
}

/** Inline SVG trend line. Shows shape only, so it carries no axes or ticks. */
function spark(vals, color, w = 70, h = 16) {
  if (!vals || vals.length < 2) return "";
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const pts = vals.map((v, i) =>
    `${(i / (vals.length - 1) * w).toFixed(1)},${(h - (v - lo) / span * h).toFixed(1)}`);
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`
       + `<polyline points="${pts.join(" ")}" fill="none" stroke="${color}" `
       + `stroke-width="1.2" vector-effect="non-scaling-stroke"/></svg>`;
}

function bar(pct, color) {
  const b = el("div", "bar"), i = el("i");
  i.style.width = Math.max(0, Math.min(100, pct || 0)) + "%";
  i.style.background = color;
  b.appendChild(i); return b;
}

// ------------------------------------------------------------------ state --

const S = { live: null, sel: null, charts: {}, utilDays: 1, tab: "overview",
            filter: "", cmp: new Set(), cmpFilter: "", seriesPending: null,
            tail: null, clDays: 1, cfg: null };

// ----------------------------------------------------------------- charts --

function chartOpts(title, series, height) {
  return {
    title, width: 100, height: height || 190,
    class: "uplot-dark",
    cursor: { y: false },
    axes: [
      // space:90 keeps time labels from stacking on top of each other when the
      // window is short and uPlot wants a tick every bucket.
      { stroke: cssVar("--dim"), grid: { stroke: cssVar("--line"), width: 1 },
        ticks: { stroke: cssVar("--line") }, space: 90 },
      { stroke: cssVar("--dim"), grid: { stroke: cssVar("--line"), width: 1 },
        ticks: { stroke: cssVar("--line") }, size: 55 },
    ],
    legend: { live: true },
    series,
  };
}

/** Create-or-update so a 20s refresh does not tear the chart down and rebuild it. */
function draw(key, node, opts, data) {
  const w = node.clientWidth || 600;
  opts.width = w;
  if (S.charts[key] && S.charts[key].node === node) {
    const u = S.charts[key].u;
    if (u.series.length === opts.series.length) { u.setSize({ width: w, height: opts.height });
      u.setData(data); return; }
  }
  // Another key may own this node (the loss panel swaps between an indexed and
  // a time-scaled chart); destroy whatever is there before reclaiming it.
  for (const [k, c] of Object.entries(S.charts))
    if (c.node === node) { c.u.destroy(); delete S.charts[k]; }
  node.innerHTML = "";
  S.charts[key] = { u: new uPlot(opts, data, node), node };
}

const PALETTE = () => [cssVar("--blue"), cssVar("--mag"), cssVar("--cyan"),
                       cssVar("--green"), cssVar("--yellow"), cssVar("--red")];

// --------------------------------------------------------------- overview --

function renderHeader(d) {
  $("clusters").innerHTML = "";
  d.clusters.forEach(c => {
    const p = el("span", "pill " + (c.ok ? "ok" : "bad"));
    p.textContent = c.ok ? `${c.name} ${c.running}R / ${c.total}` : `${c.name} down`;
    if (!c.ok) p.title = c.error;
    if (c.warn) p.title = c.warn;
    $("clusters").appendChild(p);
    $("clusters").appendChild(document.createTextNode(" "));
  });
  $("updated").textContent = "updated " + ago(d.ts);
  $("dot").className = (Date.now() / 1000 - d.ts) < 90 ? "live" : "stale";
}

/** Case-insensitive substring over the fields worth searching by. */
function matchesFilter(j, q) {
  if (!q) return true;
  return [j.name, j.jid, j.partition, j.node, j.cluster, j.reason]
    .some(v => String(v ?? "").toLowerCase().includes(q));
}

function renderJobs(d) {
  const order = { R: 0, CG: 1, PD: 2 };
  const q = S.filter.trim().toLowerCase();
  const jobs = d.jobs.filter(j => matchesFilter(j, q)).sort((a, b) =>
    (order[a.state] ?? 9) - (order[b.state] ?? 9) ||
    a.cluster.localeCompare(b.cluster) || a.jid.localeCompare(b.jid));

  const cols = [
    { label: "host", get: r => `<span class="dim">${esc(r.cluster)}</span>` },
    { label: "jobid", get: r => esc(r.jid) },
    { label: "name", get: r => `<span class="trunc" style="display:inline-block">${esc(r.name)}</span>` },
    { label: "st", get: r => `<span class="${r.state}">${esc(r.state)}</span>` },
    { label: "elapsed", get: r => `<span class="dim">${esc(r.elapsed)}</span>` },
    { label: "left", cls: r => r.minutes_left != null && r.minutes_left < 30 ? "bad" : "dim",
      get: r => esc(r.timeleft) },
    { label: "gpu", get: r => `<span class="dim trunc" style="display:inline-block">`
        + `${esc(shortGpu(r.gpu_name))}</span>` },
    { label: "gpu mem", num: true, get: r => r.gpu
        ? `<span class="dim">${r.gpu[0]}/${r.gpu[1]}</span>` : "" },
    { label: "util", num: true, cls: r => r.gpu ? utilClass(r.gpu[2]) : "faint",
      get: r => r.gpu ? r.gpu[2] + "%" : (r.state === "R" ? "n/a" : "") },
    { label: "", get: r => r.gpu ? bar(r.gpu[2], cssVar(r.gpu[2] >= 60 ? "--green"
        : r.gpu[2] >= 20 ? "--yellow" : "--red")) : "" },
    { label: "watts", num: true, cls: r => r.watts == null ? "faint" : "dim",
      get: r => r.watts != null ? Math.round(r.watts) + " W"
              : r.est_watts != null ? `<span title="estimated from TDP x util">~`
                                      + `${Math.round(r.est_watts)} W</span>` : "" },
    { label: "loss", get: r => spark(r.series, cssVar("--mag")) },
    { label: "log age", num: true, cls: r => r.log_age_min > 20 ? "bad" : "dim",
      get: r => r.log_age_min == null ? "" : fmtMin(r.log_age_min) },
    { label: "node / reason", get: r => `<span class="dim">${esc(r.node || r.reason)}</span>` },
  ];

  $("job-count").textContent = q ? `(${jobs.length} of ${d.jobs.length})` : "";
  const node = $("jobtable");
  node.innerHTML = "";
  if (!jobs.length) {
    node.appendChild(el("div", "empty",
      q ? `nothing matches "${S.filter}"` : "nothing queued or running"));
    return;
  }
  node.appendChild(table(cols, jobs, {
    rowClass: r => (S.sel && r.cluster === S.sel.cluster && r.jid === S.sel.jid) ? "sel" : "",
    onClick: r => selectJob(r.cluster, r.jid),
  }));
}

function renderAlerts(list, node, withAck) {
  node.innerHTML = "";
  if (!list.length) { node.appendChild(el("div", "empty", "no active alerts")); return; }
  list.forEach(a => {
    const d = el("div", `alert sev-${a.kind}` + (a.acked ? " acked" : ""));
    d.appendChild(el("span", "k", a.kind));
    d.appendChild(el("span", "m", a.msg));
    d.appendChild(el("span", "faint", a.cleared_at ? "cleared " + ago(a.cleared_at)
                                                   : "since " + ago(a.first_seen)));
    if (withAck && !a.acked && !a.cleared_at) {
      const b = el("button", null, "ack");
      b.onclick = async () => { await fetch(`/api/alerts/${encodeURIComponent(a.key)}/ack`,
        { method: "POST" }); a.acked = 1; d.classList.add("acked"); b.remove(); };
      d.appendChild(b);
    }
    node.appendChild(d);
  });
}

// ------------------------------------------------------------ job detail ---

async function selectJob(cluster, jid) {
  S.sel = { cluster, jid };
  writeHash();
  renderJobs(S.live);
  $("detail-card").hidden = false;
  const j = S.live.jobs.find(x => x.cluster === cluster && x.jid === jid);
  $("detail-title").textContent = `${cluster} / ${jid} ${j ? j.name : ""}`;

  const stats = $("detail-stats");
  stats.innerHTML = "";
  const put = (label, value, cls) => {
    const w = el("div"); const s = el("div", "stat");
    s.appendChild(el("b", cls, value)); w.appendChild(s);
    w.appendChild(el("div", "dim", label)); stats.appendChild(w);
  };
  if (j) {
    put("state", j.state, j.state);
    put("elapsed", j.elapsed);
    put("walltime left", j.timeleft, j.minutes_left != null && j.minutes_left < 30 ? "bad" : "");
    put("gpu util", j.gpu ? j.gpu[2] + "%" : "n/a", utilClass(j.gpu ? j.gpu[2] : null));
    put("gpus", j.gpu ? j.gpu[3] : "—");
    put("log errors", j.errors ?? 0, j.errors ? "bad" : "");
    $("detail-log").textContent = (j.log ? j.log + "\n" : "") + (j.lastline || "(no output yet)");
  } else {
    $("detail-log").textContent = "(job is no longer in the queue — showing stored history)";
  }
  // The tail costs a round-trip to the cluster, so it is a click, not a tick.
  // A fetched tail survives the ticks that follow it — otherwise the next
  // 20-second refresh would wipe what you just asked for.
  $("tail-btn").hidden = !j || j.state !== "R";
  showTail();

  // A tick can land while the previous fetch is still open, and both would draw
  // the same series. One in flight per job is enough.
  const key = `${cluster}/${jid}`;
  if (S.seriesPending === key) return;
  S.seriesPending = key;
  let s;
  try {
    s = await (await fetch(`/api/series/${cluster}/${jid}?days=14`)).json();
  } finally {
    S.seriesPending = null;
  }
  if (!S.sel || S.sel.cluster !== cluster || S.sel.jid !== jid) return;
  const ts = s.ts;

  // A running job carries the full metric history grepped out of its log, which
  // is far denser than one stored point per poll. Prefer it, indexed by sample
  // number; fall back to the stored per-tick series once the job is gone.
  const liveSeries = j && j.series && j.series.length > 1 ? j.series : null;
  if (liveSeries) {
    const opts = chartOpts("loss (last " + liveSeries.length + " log samples)", [
      { label: "sample" },
      { label: "loss", stroke: PALETTE()[1], width: 1.5 },
    ]);
    opts.scales = { x: { time: false } };
    draw("loss-idx", $("chart-loss"), opts, [liveSeries.map((_, i) => i), liveSeries]);
  } else if (ts.some((_, i) => s.loss[i] != null)) {
    draw("loss-ts", $("chart-loss"), chartOpts("loss (stored history)", [
      {}, { label: "loss", stroke: PALETTE()[1], width: 1.5, spanGaps: true },
    ]), [ts, s.loss]);
  } else { $("chart-loss").innerHTML = '<div class="empty">no loss samples yet</div>'; }

  if (ts.length > 1) {
    draw("jobgpu", $("chart-job-gpu"), chartOpts("gpu util % / mem GiB", [
      {},
      { label: "util %", stroke: PALETTE()[0], width: 1.5, spanGaps: true },
      { label: "mem GiB", stroke: PALETTE()[3], width: 1.5, spanGaps: true },
    ]), [ts, s.util, s.mem.map(v => v == null ? null : +(v / 1024).toFixed(1))]);
  } else { $("chart-job-gpu").innerHTML = '<div class="empty">not enough samples yet</div>'; }
}

// -------------------------------------------------------------- usage tab --

async function loadUsage() {
  const days = S.utilDays;
  const bucket = days <= 1 ? 300 : days <= 7 ? 1800 : 7200;
  $("util-range-lbl").textContent = `(last ${days < 1 ? days * 24 + "h" : days + "d"})`;

  const u = await (await fetch(`/api/cluster_util?days=${days}&bucket=${bucket}`)).json();
  const names = Object.keys(u);
  if (!names.length) {
    $("chart-util").innerHTML = '<div class="empty">no samples yet — the poller needs a few ticks</div>';
  } else {
    // Union of every cluster's bucket timestamps so uPlot gets one shared x axis.
    const allTs = [...new Set(names.flatMap(n => u[n].ts))].sort((a, b) => a - b);
    const idx = new Map(allTs.map((t, i) => [t, i]));
    const cols = names.map(n => {
      const arr = new Array(allTs.length).fill(null);
      u[n].ts.forEach((t, i) => { arr[idx.get(t)] = u[n].util[i]; });
      return arr;
    });
    const pal = PALETTE();
    draw("util", $("chart-util"), chartOpts("mean gpu util %", [
      {}, ...names.map((n, i) => ({ label: n, stroke: pal[i % pal.length],
        width: 1.5, spanGaps: true })),
    ], 230), [allTs, ...cols]);
  }

  renderMemEff();

  const gh = await (await fetch("/api/gpu_hours?days=30")).json();
  const tot = gh.reduce((a, r) => a + r.gpu_h, 0);
  const wast = gh.reduce((a, r) => a + r.wasted_h, 0);
  $("waste-stats").innerHTML = "";
  const stat = (label, value, cls) => {
    const w = el("div"), s = el("div", "stat");
    s.appendChild(el("b", cls, value)); w.appendChild(s);
    w.appendChild(el("div", "dim", label)); $("waste-stats").appendChild(w);
  };
  stat("gpu-hours held", tot.toFixed(1));
  stat("gpu-hours idle", wast.toFixed(1), wast / (tot || 1) > 0.4 ? "bad" : "warn");
  stat("efficiency", tot ? (100 - 100 * wast / tot).toFixed(0) + "%" : "—",
       utilClass(tot ? 100 - 100 * wast / tot : null));
  stat("jobs measured", gh.length);

  const wt = $("waste-table"); wt.innerHTML = "";
  if (!gh.length) wt.appendChild(el("div", "empty",
    "needs at least two samples of a running job — check back after a couple of refreshes"));
  else wt.appendChild(table([
    { label: "host", get: r => `<span class="dim">${esc(r.cluster)}</span>` },
    { label: "jobid", get: r => esc(r.jid) },
    { label: "name", get: r => `<span class="trunc" style="display:inline-block">${esc(r.name)}</span>` },
    { label: "gpu-h", num: true, get: r => r.gpu_h.toFixed(2) },
    { label: "idle gpu-h", num: true, cls: r => r.wasted_h / (r.gpu_h || 1) > 0.5 ? "bad" : "warn",
      get: r => r.wasted_h.toFixed(2) },
    { label: "mean util", num: true, cls: r => utilClass(r.mean_util),
      get: r => Math.round(r.mean_util) + "%" },
    { label: "last seen", num: true, get: r => `<span class="dim">${ago(r.last_ts)}</span>` },
  ], gh.slice(0, 40), { onClick: r => { switchTab("overview"); selectJob(r.cluster, r.jid); } }));

  const qw = await (await fetch("/api/queue_waits?days=30")).json();
  const qn = $("waits"); qn.innerHTML = "";
  if (!qw.length) qn.appendChild(el("div", "empty",
    "no queue transitions observed yet — a job must be seen pending and then running"));
  else qn.appendChild(table([
    { label: "host", get: r => `<span class="dim">${esc(r.cluster)}</span>` },
    { label: "partition", get: r => esc(r.partition) },
    { label: "n", num: true, get: r => r.n },
    { label: "median", num: true, get: r => fmtMin(r.median_min) },
    { label: "p90", num: true, get: r => fmtMin(r.p90_min) },
    { label: "max", num: true, get: r => fmtMin(r.max_min) },
  ], qw));

  const f = await (await fetch("/api/failures?days=30")).json();
  const fn = $("failures"); fn.innerHTML = "";
  const total = f.states.reduce((a, s) => a + s.n, 0) || 1;
  f.states.forEach(s => {
    const bad = /FAIL|CANCEL|TIMEOUT|OUT_OF|NODE_FAIL/.test(s.state);
    const row = el("div", "alert" + (bad ? " sev-ERR" : ""));
    row.style.borderLeftColor = cssVar(bad ? "--red" : s.state === "COMPLETED" ? "--green" : "--yellow");
    row.appendChild(el("span", "k", s.n));
    row.appendChild(el("span", "m", s.state));
    row.appendChild(el("span", "faint", Math.round(100 * s.n / total) + "%"));
    fn.appendChild(row);
  });
  fn.appendChild(el("div", "dim", `${f.jobs_with_log_errors} job(s) logged OOM/CUDA/traceback while running`));
}

// ------------------------------------------------------------ compare tab --

/** Every job worth offering: live queue first, then anything with stored loss. */
async function comparePool() {
  const seen = new Map();
  const add = (cluster, jid, name, live) => {
    const k = `${cluster}/${jid}`;
    if (!seen.has(k)) seen.set(k, { cluster, jid, name: name || "", live });
  };
  (S.live?.jobs || []).forEach(j => add(j.cluster, j.jid, j.name, true));
  const fin = await (await fetch("/api/finished?days=30")).json();
  fin.forEach(r => add(r.cluster, r.jid, r.name, false));
  return [...seen.values()];
}

async function loadCompare() {
  const pool = await comparePool();
  const q = S.cmpFilter.trim().toLowerCase();
  const shown = pool.filter(r => !q ||
    `${r.cluster}/${r.jid} ${r.name}`.toLowerCase().includes(q));

  const pick = $("cmp-picker");
  pick.innerHTML = "";
  if (!shown.length) pick.appendChild(el("div", "empty", "no runs to compare"));
  shown.slice(0, 200).forEach(r => {
    const k = `${r.cluster}/${r.jid}`;
    const b = el("button", "chip" + (S.cmp.has(k) ? " active" : ""),
                 `${r.jid} ${r.name}`.trim());
    b.title = k + (r.live ? " (in the queue now)" : "");
    if (r.live) b.classList.add("live");
    b.onclick = () => {
      // Twelve curves is where the legend stops being readable and the API
      // stops accepting more.
      if (S.cmp.has(k)) S.cmp.delete(k);
      else if (S.cmp.size < 12) S.cmp.add(k);
      loadCompare();
    };
    pick.appendChild(b);
  });

  const keys = [...S.cmp];
  $("cmp-note").textContent = keys.length ? `${keys.length} selected` : "pick two or more runs";
  const node = $("chart-compare");
  if (keys.length < 1) { node.innerHTML = '<div class="empty">nothing selected</div>'; return; }

  const qs = keys.map(k => "job=" + encodeURIComponent(k)).join("&");
  const data = await (await fetch(`/api/compare?${qs}&days=30`)).json();
  const withLoss = data.filter(r => r.loss.length > 1);
  if (!withLoss.length) {
    node.innerHTML = '<div class="empty">no loss samples stored for these runs</div>';
    return;
  }

  const n = Math.max(...withLoss.map(r => r.loss.length));
  const pal = PALETTE();
  const opts = chartOpts("loss by step", [
    { label: "step" },
    ...withLoss.map((r, i) => ({ label: `${r.jid} ${r.name}`.trim(),
      stroke: pal[i % pal.length], width: 1.5, spanGaps: true })),
  ], 300);
  opts.scales = { x: { time: false } };
  draw("compare", node, opts, [
    Array.from({ length: n }, (_, i) => i),
    ...withLoss.map(r => Array.from({ length: n }, (_, i) => r.loss[i] ?? null)),
  ]);
}

// ------------------------------------------------------------ history tab --

async function loadHistory() {
  const tl = await (await fetch("/api/timeline?days=7")).json();
  const g = $("gantt"); g.innerHTML = "";
  if (!tl.length) { g.appendChild(el("div", "empty", "no jobs observed yet")); return; }

  const now = Date.now() / 1000;
  const t0 = Math.min(...tl.map(r => r.first_seen));
  const span = Math.max(1, now - t0);
  const pc = (t) => ((t - t0) / span) * 100;

  tl.slice().reverse().forEach(r => {
    const row = el("div", "row");
    row.appendChild(el("div", "lbl", `${r.cluster}/${r.jid} ${r.name || ""}`));
    const track = el("div", "track");
    const end = r.end_ts || r.last_seen;
    const runStart = r.run_start || end;
    if (r.pend_start && r.pend_start < runStart) {
      const q = el("i", "seg");
      q.style.left = pc(r.pend_start) + "%";
      q.style.width = Math.max(0.4, pc(runStart) - pc(r.pend_start)) + "%";
      q.style.background = cssVar("--yellow"); q.style.opacity = ".5";
      q.title = `queued ${fmtMin((runStart - r.pend_start) / 60)}`;
      track.appendChild(q);
    }
    if (r.run_start) {
      const s = el("i", "seg");
      s.style.left = pc(r.run_start) + "%";
      s.style.width = Math.max(0.4, pc(end) - pc(r.run_start)) + "%";
      const bad = /FAIL|CANCEL|TIMEOUT|OUT_OF|NODE_FAIL/.test(r.final_state || "");
      s.style.background = cssVar(bad ? "--red" : r.last_state === "R" ? "--green" : "--blue");
      s.title = `${r.final_state || r.last_state} — ran ${fmtMin((end - r.run_start) / 60)}`;
      track.appendChild(s);
    }
    row.appendChild(track);
    row.appendChild(el("span", "faint", fmtMin((end - (r.run_start || end)) / 60)));
    g.appendChild(row);
  });

  const fin = await (await fetch("/api/finished?days=7")).json();
  const fn = $("finished"); fn.innerHTML = "";
  if (!fin.length) { fn.appendChild(el("div", "empty", "nothing in the window")); return; }
  fn.appendChild(table([
    { label: "host", get: r => `<span class="dim">${esc(r.cluster)}</span>` },
    { label: "jobid", get: r => esc(r.jid) },
    { label: "name", get: r => `<span class="trunc" style="display:inline-block">${esc(r.name)}</span>` },
    { label: "state", cls: r => /COMPLETED/.test(r.state) ? "good"
        : /FAIL|CANCEL|TIMEOUT|OUT_OF|NODE_FAIL/.test(r.state) ? "bad" : "warn",
      get: r => esc(r.state) },
    { label: "elapsed", num: true, get: r => `<span class="dim">${esc(r.elapsed)}</span>` },
    { label: "exit", num: true, get: r => `<span class="dim">${esc(r.exitcode)}</span>` },
    { label: "maxrss", num: true, get: r => `<span class="dim">${esc(r.maxrss)}</span>` },
    { label: "ended", num: true, get: r => `<span class="dim">${r.end_ts ? fmtTs(r.end_ts) : "—"}</span>` },
  ], fin, { onClick: r => { switchTab("overview"); selectJob(r.cluster, r.jid); } }));
}

/** Asked vs peak used, worst-fitting first. sacct only reports MaxRSS on the
    batch step, so jobs that finished before that was collected are absent. */
async function renderMemEff() {
  const rows = await (await fetch("/api/mem_efficiency?days=30")).json();
  const node = $("memeff");
  node.innerHTML = "";
  if (!rows.length) {
    node.appendChild(el("div", "empty",
      "no memory samples yet — this fills in as jobs finish"));
    return;
  }
  const gib = (m) => (m / 1024).toFixed(1) + "G";
  node.appendChild(table([
    { label: "host", get: r => `<span class="dim">${esc(r.cluster)}</span>` },
    { label: "jobid", get: r => esc(r.jid) },
    { label: "name", get: r => `<span class="trunc" style="display:inline-block">${esc(r.name)}</span>` },
    { label: "state", cls: r => r.state === "COMPLETED" ? "dim" : "bad",
      get: r => esc(r.state) },
    { label: "asked", num: true, get: r => gib(r.req_mib) },
    { label: "used", num: true, get: r => gib(r.used_mib) },
    { label: "fit", num: true, cls: r => r.pct >= 60 ? "good" : r.pct >= 25 ? "warn" : "bad",
      get: r => r.pct.toFixed(0) + "%" },
    { label: "", get: r => bar(r.pct, cssVar(r.pct >= 60 ? "--green"
        : r.pct >= 25 ? "--yellow" : "--red")) },
  ], rows.slice(0, 40)));
}

async function loadAlertHistory() {
  const a = await (await fetch("/api/alerts?days=14")).json();
  renderAlerts(a, $("alerts-all"), true);
}

// ----------------------------------------------------------- clusters tab --

const fmtW = (w) => w == null ? "n/a" : w >= 1000 ? (w / 1000).toFixed(2) + " kW"
                                                 : Math.round(w) + " W";

/** A labelled meter. `used` and `total` are counts, not percentages. */
function meter(label, used, total, color) {
  const w = el("div", "meter");
  const head = el("div", "meter-head");
  head.appendChild(el("span", null, label));
  head.appendChild(el("span", "dim", total ? `${used} / ${total}` : "—"));
  w.appendChild(head);
  w.appendChild(bar(total ? used / total * 100 : 0, color));
  return w;
}

function clusterCapacity(t) {
  const box = el("div");
  const pal = PALETTE();
  box.appendChild(meter("gpus in use", t.gpus_alloc, t.gpus_total, pal[0]));
  box.appendChild(meter("cpus in use", t.cpus_alloc, t.cpus_total, pal[2]));
  box.appendChild(meter("nodes busy", t.nodes_alloc + t.nodes_mixed, t.nodes_total, pal[1]));

  const free = el("div", "stats");
  const stat = (label, value, cls) => {
    const w = el("div"), st = el("div", "stat");
    st.appendChild(el("b", cls, String(value)));
    w.appendChild(st); w.appendChild(el("div", "dim", label));
    free.appendChild(w);
  };
  const idleGpus = Math.max(0, t.gpus_total - t.gpus_alloc);
  stat("gpus free", idleGpus, idleGpus > 0 ? "good" : "bad");
  stat("nodes idle", t.nodes_idle, t.nodes_idle > 0 ? "good" : "faint");
  stat("nodes down", t.nodes_down, t.nodes_down ? "warn" : "faint");
  box.appendChild(free);

  const models = Object.entries(t.gpu_models || {}).sort((a, b) => b[1] - a[1]);
  if (models.length) {
    const m = el("div", "dim", "");
    m.style.marginTop = "8px";
    m.innerHTML = models.map(([k, v]) => `${esc(k)}&nbsp;&times;${v}`).join(" &middot; ");
    box.appendChild(m);
  }
  return box;
}

/** How much of what is running right now is yours. */
function myShare(name, total) {
  const mine = (S.live?.jobs || []).filter(j => j.cluster === name && j.state === "R").length;
  const pct = total ? (mine / total * 100) : 0;
  const w = el("div", "dim");
  w.style.marginTop = "8px";
  w.textContent = total
    ? `${mine} of ${total} running jobs are yours (${pct.toFixed(0)}%)`
    : "nothing running";
  return w;
}

function naCard(title, why) {
  const c = el("div", "card");
  c.appendChild(el("h2", null, title));
  c.appendChild(el("div", "empty", why));
  return c;
}

function powerCard(name, live, kwh) {
  const c = el("div", "card");
  c.appendChild(el("h2", null, "power — " + name));
  const measured = live?.watts, est = live?.est_watts;
  const stats = el("div", "stats");
  const stat = (label, value, cls) => {
    const w = el("div"), st = el("div", "stat");
    st.appendChild(el("b", cls, value));
    w.appendChild(st); w.appendChild(el("div", "dim", label));
    stats.appendChild(w);
  };
  const now = measured != null ? measured : est;
  stat(now == null ? "nothing drawing"
       : measured != null ? "measured now" : "estimated now",
       fmtW(now), now == null ? "faint" : measured != null ? "good" : "warn");
  if (kwh) {
    stat(`energy (last ${S.clDays}d)`,
         (kwh.kwh || kwh.est_kwh || 0).toFixed(2) + " kWh");
    stat("at " + (S.cfg?.price_per_kwh ?? 0.12) + " /kWh", "$" + (kwh.dollars ?? 0).toFixed(2));
  }
  c.appendChild(stats);
  if (measured == null && est != null)
    c.appendChild(el("div", "dim",
      "estimated from board TDP x utilization — reads low on an idle card, "
      + "which still draws 40-60 W"));
  if (kwh && kwh.node_scoped)
    c.appendChild(el("div", "warn",
      `${kwh.node_scoped} job(s) read power off the whole node, not just their `
      + "own cards — those figures are an upper bound"));
  return c;
}

async function loadClusters() {
  try {
    await renderClusters();
  } catch (e) {
    $("cluster-cards").innerHTML = "";
    $("cluster-cards").appendChild(naCard("clusters unavailable", String(e.message)));
    $("cl-note").textContent = "the running service is older than this page — "
      + "restart jobdash and reload";
  }
}

async function renderClusters() {
  if (!S.cfg) S.cfg = await getJSON("/api/config");
  const days = S.clDays;
  const bucket = days <= 1 ? 300 : days <= 7 ? 1800 : 7200;
  const [stats, en] = await Promise.all([
    getJSON(`/api/clusters?days=${days}&bucket=${bucket}`),
    getJSON(`/api/energy?days=${days}`),
  ]);
  const byCluster = Object.fromEntries((en.clusters || []).map(c => [c.cluster, c]));
  const liveBy = Object.fromEntries((S.live?.clusters || []).map(c => [c.name, c]));

  const wrap = $("cluster-cards");
  wrap.innerHTML = "";
  const pal = PALETTE();

  const cls = S.cfg.clusters || [];
  if (cls.length && typeof cls[0] !== "object")
    throw new Error("/api/config returned cluster names without kind or gpu");

  cls.forEach(cl => {
    const name = cl.name, st = stats[name], live = liveBy[name];

    if (!st || !st.total) {
      wrap.appendChild(naCard("queue — " + name, cl.kind === "tmux"
        ? "no scheduler on this host — there is no queue to report"
        : "no cluster samples yet — the poller needs a tick"));
    } else {
      const t = st.total;
      const c = el("div", "card");
      c.appendChild(el("h2", null, "queue — " + name));
      const s = el("div", "stats");
      [["running", t.running, "good"], ["pending", t.pending, t.pending ? "warn" : "faint"],
       ["other", t.other, "faint"]].forEach(([label, v, cls]) => {
        const w = el("div"), b = el("div", "stat");
        b.appendChild(el("b", cls, String(v))); w.appendChild(b);
        w.appendChild(el("div", "dim", label)); s.appendChild(w);
      });
      c.appendChild(s);
      c.appendChild(myShare(name, t.running));
      const wrapT = el("div", "scroll-x");
      wrapT.appendChild(table([
        { label: "partition", get: r => esc(r.partition) },
        { label: "run", num: true, get: r => r.running },
        { label: "pend", num: true, get: r => r.pending },
        { label: "gpus", num: true, get: r => `${r.gpus_alloc}/${r.gpus_total}` },
        { label: "free", num: true, cls: r => r.gpus_total - r.gpus_alloc > 0 ? "good" : "bad",
          get: r => r.gpus_total - r.gpus_alloc },
        { label: "nodes idle", num: true, get: r => `${r.nodes_idle}/${r.nodes_total}` },
      ], st.partitions));
      c.appendChild(wrapT);
      wrap.appendChild(c);

      const cap = el("div", "card");
      cap.appendChild(el("h2", null, "capacity — " + name));
      cap.appendChild(clusterCapacity(t));
      wrap.appendChild(cap);
    }

    if (!cl.gpu) {
      wrap.appendChild(naCard("power — " + name, "gpu polling is off for this cluster"));
    } else {
      wrap.appendChild(powerCard(name, live, byCluster[name]));
    }

    if (st && st.series && st.series.ts.length > 1) {
      const ch = el("div", "card wide");
      ch.appendChild(el("h2", null, `load — ${name}`));
      const node = el("div", "chart");
      node.style.height = "200px";
      ch.appendChild(node);
      wrap.appendChild(ch);
      draw("cl-" + name, node, chartOpts("jobs queued / gpus allocated", [
        {},
        { label: "running", stroke: pal[0], width: 1.5, spanGaps: true },
        { label: "pending", stroke: pal[4], width: 1.5, spanGaps: true },
        { label: "gpus used", stroke: pal[3], width: 1.5, spanGaps: true },
        { label: "gpus total", stroke: cssVar("--dim"), width: 1, dash: [4, 4] },
      ], 200), [st.series.ts, st.series.running, st.series.pending,
                st.series.gpus_alloc, st.series.gpus_total]);
    }
  });

  $("cl-note").textContent = en.clusters?.length
    ? `energy priced at $${en.price_per_kwh}/kWh` : "";
}

// ------------------------------------------------------------------- tabs --

function switchTab(name) {
  S.tab = name;
  document.querySelectorAll("nav button").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  ["overview", "usage", "clusters", "compare", "history", "alerts"].forEach(t =>
    $("tab-" + t).hidden = t !== name);
  writeHash();
  if (name === "usage") loadUsage();
  if (name === "clusters") loadClusters();
  if (name === "compare") loadCompare();
  if (name === "history") loadHistory();
  if (name === "alerts") loadAlertHistory();
}

// ------------------------------------------------------------- permalink ---
// The tab and the selected job live in the hash, so a reload lands where you
// left off and a job view can be pasted to someone else.

let hashLock = false;

function writeHash() {
  const parts = [`tab=${S.tab}`];
  if (S.sel) parts.push(`job=${S.sel.cluster}/${S.sel.jid}`);
  if (S.filter) parts.push(`q=${encodeURIComponent(S.filter)}`);
  const h = "#" + parts.join("&");
  if (h === location.hash) return;
  hashLock = true;
  history.replaceState(null, "", h);
  hashLock = false;
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const q = p.get("q") || "";
  if (q !== S.filter) { S.filter = q; $("filter").value = q; }
  const tab = p.get("tab");
  if (tab && $("tab-" + tab) && tab !== S.tab) switchTab(tab);
  const job = p.get("job");
  if (job && job.includes("/")) {
    const [cluster, ...rest] = job.split("/");
    const jid = rest.join("/");
    if (!S.sel || S.sel.cluster !== cluster || S.sel.jid !== jid) selectJob(cluster, jid);
  }
}

addEventListener("hashchange", () => { if (!hashLock) readHash(); });

document.querySelectorAll("nav button").forEach(b =>
  b.onclick = () => switchTab(b.dataset.tab));

$("filter").oninput = (e) => {
  S.filter = e.target.value;
  writeHash();
  if (S.live) renderJobs(S.live);
};

/** Paint the stored tail if it belongs to the selected job, else leave the
    live last line alone. */
function showTail() {
  const key = S.sel ? `${S.sel.cluster}/${S.sel.jid}` : "";
  const t = S.tail;
  const log = $("detail-log");
  if (!t || t.key !== key) {
    log.classList.remove("tall");
    $("tail-note").textContent = "";
    return;
  }
  log.textContent = t.text || "(empty)";
  log.classList.add("tall");
  log.scrollTop = log.scrollHeight;
  $("tail-note").textContent = `${t.path} — ${fetchedAgo(t.at)}`;
}

const fetchedAgo = (t) => "fetched " + ago(t / 1000);

$("tail-btn").onclick = async () => {
  if (!S.sel) return;
  const { cluster, jid } = S.sel;
  const key = `${cluster}/${jid}`;
  $("tail-note").textContent = "fetching…";
  $("tail-btn").disabled = true;
  try {
    const r = await (await fetch(`/api/logtail/${cluster}/${jid}?lines=120`)).json();
    S.tail = { key, path: r.path, text: r.text, at: Date.now() };
    if (S.sel && `${S.sel.cluster}/${S.sel.jid}` === key) showTail();
  } catch (e) {
    $("tail-note").textContent = "fetch failed";
  } finally {
    $("tail-btn").disabled = false;
  }
};

$("cmp-filter").oninput = (e) => { S.cmpFilter = e.target.value; loadCompare(); };
$("cmp-clear").onclick = () => { S.cmp.clear(); loadCompare(); };

$("refresh").onclick = () => fetch("/api/refresh", { method: "POST" });

[[0.25, "6h"], [1, "24h"], [7, "7d"], [30, "30d"]].forEach(([d, label]) => {
  const b = el("button", d === 1 ? "active" : "", label);
  b.onclick = () => {
    S.utilDays = d;
    document.querySelectorAll("#util-range button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    loadUsage();
  };
  $("util-range").appendChild(b);
});

[[0.25, "6h"], [1, "24h"], [7, "7d"], [30, "30d"]].forEach(([d, label]) => {
  const b = el("button", d === 1 ? "active" : "", label);
  b.onclick = () => {
    S.clDays = d;
    document.querySelectorAll("#cl-range button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    loadClusters();
  };
  $("cl-range").appendChild(b);
});

// ------------------------------------------------------------------ live ---

function apply(d) {
  S.live = d;
  renderHeader(d);
  renderJobs(d);
  renderAlerts(d.alerts, $("alerts-live"), true);
  if (S.sel) selectJob(S.sel.cluster, S.sel.jid);
  if (S.tab === "usage") loadUsage();
  if (S.tab === "clusters") loadClusters();
}

function connect() {
  const es = new EventSource("/events");
  es.onmessage = (e) => apply(JSON.parse(e.data));
  es.onerror = () => { $("dot").className = "stale"; };
}

fetch("/api/live").then(r => r.json()).then(d => { if (d.ts) apply(d); })
  .then(readHash).finally(connect);
setInterval(() => { if (S.live) $("updated").textContent = "updated " + ago(S.live.ts); }, 5000);
addEventListener("resize", () => {
  Object.values(S.charts).forEach(c => c.u.setSize(
    { width: c.node.clientWidth, height: c.u.height }));
});
