// ---------- SECTION 1: CHUNKING LAB + BUILD ----------
const MAP_CHAR_LIMIT = 6000;

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function chunkParams() {
  return { chunk_size: +$("chunk-size").value, chunk_overlap: +$("chunk-overlap").value };
}

function syncSliders() {
  const size = +$("chunk-size").value;
  const overlap = $("chunk-overlap");
  overlap.max = Math.min(400, size - 10);
  if (+overlap.value > +overlap.max) overlap.value = overlap.max;
  $("out-size").textContent = size;
  $("out-overlap").textContent = overlap.value;
}

// Slice the source text into segments by how many chunks cover each stretch.
function renderChunkMap(text, preview) {
  const map = $("chunk-map");
  map.innerHTML = "";
  const shown = text.slice(0, MAP_CHAR_LIMIT);
  const cuts = new Set([0, shown.length]);
  preview.chunks.forEach((c) => { cuts.add(Math.min(c.start, shown.length)); cuts.add(Math.min(c.end, shown.length)); });
  const points = [...cuts].sort((a, b) => a - b);
  for (let i = 0; i < points.length - 1; i++) {
    const [a, b] = [points[i], points[i + 1]];
    if (a === b) continue;
    const covering = preview.chunks.filter((c) => c.start <= a && c.end >= b);
    const span = el("span", "seg", shown.slice(a, b));
    if (covering.length >= 2) {
      span.classList.add("overlap");
      span.title = `Overlap: chunks ${covering.map((c) => c.id).join(" and ")}`;
    } else if (covering.length === 1) {
      span.classList.add(`c${covering[0].id % 3}`);
      span.title = `Chunk ${covering[0].id}`;
    }
    map.appendChild(span);
  }
  if (text.length > shown.length) {
    map.appendChild(el("span", "seg", `\n... (${(text.length - shown.length).toLocaleString()} more characters not drawn)`));
  }
  map.hidden = false;
  $("chunk-legend").hidden = false;
}

async function refreshChunkPreview() {
  const text = $("build-text").value;
  if (!text.trim()) {
    $("chunk-map").hidden = true;
    $("chunk-legend").hidden = true;
    $("chunk-stats").textContent = "";
    return;
  }
  try {
    const preview = await callApi("POST", "/api/chunk-preview", { text, ...chunkParams() });
    renderChunkMap(text, preview);
    $("chunk-stats").textContent =
      `${preview.chunk_count} chunks, ${preview.avg_chars} characters on average, ${preview.overlaps.length} overlaps.`;
  } catch (err) {
    $("chunk-stats").textContent = err.message;
  }
}
const refreshChunkPreviewSoon = debounce(refreshChunkPreview, 250);

function showBuilt(data) {
  state.sessionId = data.session_id;
  state.builtText = $("build-text").value;
  const box = $("build-timing");
  box.innerHTML = "";
  box.appendChild(el("p", "stage-label", `Indexed ${data.chunk_count} chunks${data.cached ? " (cached)" : ""}`));
  const wf = el("div");
  box.appendChild(wf);
  renderTiming(wf, data.timings_ms);
  toast(`Index ready: ${data.chunk_count} chunks.`, false);
}

async function buildFromText() {
  const text = $("build-text").value;
  if (!text.trim()) throw new Error("Paste some text or load the sample first.");
  const cfg = { ...chunkParams(), embedder: $("embedder").value };
  const inspector = $("build-inspector");
  const reuse = state.sessionId && state.builtText === text;
  const data = reuse
    ? await callApi("POST", "/api/rebuild", { session_id: state.sessionId, ...cfg }, inspector)
    : await callApi("POST", "/api/build-from-text", { text, ...cfg }, inspector);
  showBuilt(data);
}

async function buildFromFile() {
  const input = $("build-file");
  if (!input.files.length) throw new Error("Choose a .txt or .md file first.");
  const form = new FormData();
  form.append("file", input.files[0]);
  form.append("chunk_size", $("chunk-size").value);
  form.append("chunk_overlap", $("chunk-overlap").value);
  form.append("embedder", $("embedder").value);
  const data = await callApi("POST", "/api/build-from-file", form, $("build-inspector"), true);
  showBuilt(data);
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadConfig();
  } catch (err) {
    toast("Could not load config: " + err.message);
    return;
  }
  const d = state.cfg.defaults;
  fillSelect($("embedder"), state.cfg.embedders, d.embedder);
  fillSelect($("strategy"), state.cfg.strategies, d.strategy);
  $("distractor").value = state.cfg.default_distractor;
  syncSliders();

  ["chunk-size", "chunk-overlap"].forEach((id) =>
    $(id).addEventListener("input", () => { syncSliders(); refreshChunkPreviewSoon(); }));
  $("build-text").addEventListener("input", refreshChunkPreviewSoon);
  $("k").addEventListener("input", () => ($("out-k").textContent = $("k").value));

  $("load-sample").addEventListener("click", () => {
    $("build-text").value = state.cfg.sample.text;
    refreshChunkPreview();
  });
  $("build-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $("build-text").value = await file.text();
    refreshChunkPreview();
  });
  guarded($("build-btn"), buildFromText);
  guarded($("build-file-btn"), buildFromFile);

  const chips = $("sample-chips");
  state.cfg.sample.questions.forEach((q) => {
    const chip = el("button", "chip", q.label);
    chip.type = "button";
    chip.addEventListener("click", () => {
      $("question").value = q.question;
      $("sample-hint").textContent = q.hint;
    });
    chips.appendChild(chip);
  });
  chips.insertAdjacentElement("beforebegin", el("p", "hint", "Sample questions (written for the sample document):"));
  initCompare();
});
