/* Combinato-Lite curation GUI */

const state = {
  groups: [],
  activeGid: null,
  selected: new Set(),
  compareGid: null,
};

const typeClass = { 2: "su", 1: "mu", 0: "none", [-1]: "art" };

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

async function loadInfo() {
  const info = await api("/api/info");
  document.getElementById("info").textContent =
    `${info.datafile}  ·  ${info.sorting}  ·  ${info.sign}  ·  ${info.n_spikes} spikes`;
}

function renderList() {
  const ul = document.getElementById("group-list");
  ul.innerHTML = "";
  for (const g of state.groups) {
    const li = document.createElement("li");
    if (g.gid === state.activeGid) li.classList.add("active");
    if (state.selected.has(g.gid)) li.classList.add("selected");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selected.has(g.gid);
    cb.addEventListener("click", (e) => {
      e.stopPropagation();
      if (cb.checked) state.selected.add(g.gid);
      else state.selected.delete(g.gid);
      document.getElementById("btn-merge").disabled = state.selected.size < 2;
      li.classList.toggle("selected", cb.checked);
    });

    const badge = document.createElement("span");
    badge.className = `badge ${typeClass[g.type] || "none"}`;
    badge.textContent = g.type_name;

    const label = document.createElement("span");
    label.textContent = `G${g.gid} · ${g.n_spikes} spk · ${g.n_clusters} cl`;

    li.append(cb, badge, label);
    li.addEventListener("click", () => selectGroup(g.gid));
    ul.appendChild(li);
  }
}

async function refreshGroups() {
  state.groups = await api("/api/groups");
  renderList();
  if (state.activeGid != null) await showGroup(state.activeGid);
}

async function selectGroup(gid) {
  if (state.activeGid != null && state.activeGid !== gid && (event?.shiftKey || event?.metaKey)) {
    state.compareGid = gid;
  } else {
    state.activeGid = gid;
    state.compareGid = null;
  }
  renderList();
  await showGroup(state.activeGid);
  const other = state.compareGid ?? state.activeGid;
  await showXcorr(state.activeGid, other);
}

async function showGroup(gid) {
  const d = await api(`/api/groups/${gid}`);
  const mean = d.mean_waveform;
  const x = mean.map((_, i) => i);

  const traces = [];
  for (const wf of d.waveforms.slice(0, 80)) {
    traces.push({
      x, y: wf, type: "scatter", mode: "lines",
      line: { color: "rgba(61,139,253,0.15)", width: 1 },
      hoverinfo: "skip", showlegend: false,
    });
  }
  traces.push({
    x, y: mean, type: "scatter", mode: "lines",
    line: { color: "#fff", width: 2 },
    name: "mean",
  });

  Plotly.newPlot("wave-plot", traces, {
    title: `Group ${gid} · ${d.n_spikes} spikes · type ${d.type}`,
    paper_bgcolor: "#1a2332",
    plot_bgcolor: "#1a2332",
    font: { color: "#e7ecf3", size: 11 },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { title: "sample" },
    yaxis: { title: "µV" },
    showlegend: false,
  }, { responsive: true });

  // Simple raster / time histogram
  const times = d.times;
  Plotly.newPlot("raster-plot", [{
    x: times, type: "histogram",
    marker: { color: "#3d8bfd" },
    nbinsx: 80,
  }], {
    title: "Spike times",
    paper_bgcolor: "#1a2332",
    plot_bgcolor: "#1a2332",
    font: { color: "#e7ecf3", size: 11 },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { title: "time (ms)" },
    yaxis: { title: "count" },
  }, { responsive: true });
}

async function showXcorr(a, b) {
  const xc = await api(`/api/xcorr?gid_a=${a}&gid_b=${b}&lag_ms=50&bins=100`);
  const centers = [];
  for (let i = 0; i < xc.edges.length - 1; i++) {
    centers.push((xc.edges[i] + xc.edges[i + 1]) / 2);
  }
  Plotly.newPlot("xcorr-plot", [{
    x: centers, y: xc.counts, type: "bar",
    marker: { color: "#2ecc71" },
  }], {
    title: a === b ? `Auto-correlogram G${a}` : `Cross-correlogram G${a} × G${b}`,
    paper_bgcolor: "#1a2332",
    plot_bgcolor: "#1a2332",
    font: { color: "#e7ecf3", size: 11 },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { title: "lag (ms)" },
    yaxis: { title: "count" },
  }, { responsive: true });
}

async function mergeSelected() {
  const gids = [...state.selected];
  if (gids.length < 2) return;
  await api("/api/merge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gids }),
  });
  state.selected.clear();
  document.getElementById("btn-merge").disabled = true;
  await refreshGroups();
}

async function setType(type) {
  if (state.activeGid == null) return;
  await api("/api/set_type", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gid: state.activeGid, type: Number(type) }),
  });
  await refreshGroups();
}

document.getElementById("btn-refresh").addEventListener("click", refreshGroups);
document.getElementById("btn-merge").addEventListener("click", mergeSelected);
document.querySelectorAll(".type-bar button[data-type]").forEach((btn) => {
  btn.addEventListener("click", () => setType(btn.dataset.type));
});

(async function init() {
  await loadInfo();
  await refreshGroups();
  if (state.groups.length) await selectGroup(state.groups[0].gid);
})();
