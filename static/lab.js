// ---------- SECTION 4: CONTROLLED BREAKAGE ----------
async function runExperiment(mode) {
  requireSession();
  const body = { ...baseQuery(), mode, distractor: $("distractor").value };
  const data = await callApi("POST", "/api/experiment", body);
  $("exp-desc").textContent = data.description + (data.note ? ` (${data.note})` : "");
  $("exp-base").textContent = data.baseline.answer;
  $("exp-broken").textContent = data.experiment.answer;
  $("exp-label").textContent = {
    no_context: "No retrieval", distractor: "With distractor", shuffle_middle: "Best chunk buried",
  }[mode];
  $("exp-result").hidden = false;
}

// ---------- SECTION 5: SIDE-BY-SIDE COMPARISON ----------
const CMP_SIZES = [200, 300, 500, 800, 1200];
const CMP_OVERLAPS = [0, 25, 50, 100, 150];
const CMP_KS = [1, 2, 3, 4, 6, 8];
const CMP_FIELDS = ["size", "overlap", "embedder", "strategy", "k"];

function cmpPanel(side) {
  const panel = el("div", "cmp-panel");
  panel.appendChild(el("h3", "", `Setup ${side.toUpperCase()}`));
  const field = (name, label, options) => {
    const wrap = el("label", "", label);
    const select = el("select");
    select.id = `cmp-${side}-${name}`;
    fillSelect(select, options);
    wrap.appendChild(select);
    panel.appendChild(wrap);
  };
  const nums = (arr) => Object.fromEntries(arr.map((n) => [n, String(n)]));
  field("size", "Chunk size", nums(CMP_SIZES));
  field("overlap", "Overlap", nums(CMP_OVERLAPS));
  field("embedder", "Embedding model", state.cfg.embedders);
  field("strategy", "Strategy", state.cfg.strategies);
  field("k", "Top-k", nums(CMP_KS));
  return panel;
}

function setSide(side, values) {
  Object.entries(values).forEach(([name, value]) => ($(`cmp-${side}-${name}`).value = String(value)));
}

const PRESETS = {
  "dense-bm25": [{ strategy: "dense" }, { strategy: "bm25" }],
  chunks: [{ size: 200, overlap: 25 }, { size: 1200, overlap: 100 }],
  models: [{ embedder: "minilm" }, { embedder: "bge-small" }],
};

function sideConfig(side) {
  const v = (name) => $(`cmp-${side}-${name}`).value;
  return {
    chunk_size: +v("size"), chunk_overlap: +v("overlap"), embedder: v("embedder"),
    strategy: v("strategy"), k: +v("k"),
  };
}

function describeConfig(c) {
  return `${c.strategy} · ${c.embedder} · size ${c.chunk_size}/${c.chunk_overlap} · k=${c.k}`;
}

function renderSide(container, side, title) {
  const box = el("div", "cmp-side");
  box.appendChild(el("h3", "", title));
  box.appendChild(el("p", "hint", `${describeConfig(side.config)} · ${side.chunk_count} chunks`));
  if (side.answer !== null) {
    const stage = el("div", "stage");
    stage.appendChild(el("span", "stage-label", "Answer"));
    stage.appendChild(el("p", "answer", side.answer));
    box.appendChild(stage);
  } else if (side.answer_error) {
    box.appendChild(el("p", "diagnosis retrieval_failure", side.answer_error));
  }
  const wf = el("div");
  box.appendChild(wf);
  renderTiming(wf, side.timings_ms);
  const chunks = el("div", "result");
  renderChunkCards(chunks, side.retrieved);
  box.appendChild(chunks);
  container.appendChild(box);
}

async function runCompare() {
  requireSession();
  const data = await callApi("POST", "/api/compare", {
    session_id: state.sessionId,
    question: requireQuestion(),
    a: sideConfig("a"),
    b: sideConfig("b"),
    skip_llm: $("cmp-skip-llm").checked || !state.cfg.llm_configured,
  });
  const out = $("cmp-result");
  out.innerHTML = "";
  renderSide(out, data.a, "Setup A");
  renderSide(out, data.b, "Setup B");
  $("cmp-summary").textContent =
    `${Math.round(data.span_overlap * 100)}% of A's retrieved text overlaps something B retrieved ` +
    `(compared by character span, so it works across different chunk sizes).`;
}

function initCompare() {
  const cfg = $("cmp-config");
  cfg.append(cmpPanel("a"), cmpPanel("b"));
  const d = state.cfg.defaults;
  ["a", "b"].forEach((s) => setSide(s, { size: d.chunk_size, overlap: d.chunk_overlap, embedder: d.embedder, strategy: d.strategy, k: d.k }));
  setSide("b", { strategy: "bm25" });
  document.querySelectorAll("[data-preset]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const [a, b] = PRESETS[btn.dataset.preset];
      ["a", "b"].forEach((s) => setSide(s, { size: d.chunk_size, overlap: d.chunk_overlap, embedder: d.embedder, strategy: d.strategy, k: d.k }));
      setSide("a", a);
      setSide("b", b);
    }));
  guarded($("compare-btn"), runCompare);
  document.querySelectorAll(".exp").forEach((btn) => guarded(btn, () => runExperiment(btn.dataset.mode)));
}
