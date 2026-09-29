// ---------- SHARED STATE + HELPERS ----------
// Everything user- or model-supplied goes in via textContent, never innerHTML.
const state = { sessionId: null, cfg: null, builtText: null };
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function callApi(method, url, body, inspectorEl, isFormData) {
  const opts = { method };
  if (body !== undefined) {
    if (isFormData) {
      opts.body = body;
    } else {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
  }
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
  if (!res.ok) {
    const detail = data && data.detail;
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail;
    throw new Error(msg || `Request failed (${res.status})`);
  }
  if (inspectorEl) {
    const shown = isFormData ? "(multipart file upload)" : body ? JSON.stringify(body, null, 2) : "(none)";
    inspectorEl.querySelector("pre").textContent =
      `${method} ${url}\n\nRequest body:\n${shown}\n\nResponse (${res.status}):\n${JSON.stringify(data, null, 2)}`;
    inspectorEl.hidden = false;
  }
  return data;
}

function toast(message, isError = true) {
  let box = $("toast");
  if (!box) {
    box = el("div", "toast");
    box.id = "toast";
    document.body.appendChild(box);
  }
  box.textContent = message;
  box.className = "toast show" + (isError ? " error" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (box.className = "toast"), 5000);
}

// Run an async click handler with a busy button and error toast.
function guarded(btn, fn) {
  btn.addEventListener("click", async () => {
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Working...";
    try { await fn(); } catch (err) { toast(err.message); } finally {
      btn.disabled = false;
      btn.textContent = label;
    }
  });
}

function requireSession() {
  if (!state.sessionId) throw new Error("Build an index first (section 1).");
}
function requireQuestion() {
  const q = $("question").value.trim();
  if (!q) throw new Error("Type a question first.");
  return q;
}
function baseQuery() {
  return { session_id: state.sessionId, question: requireQuestion(), k: +$("k").value, strategy: $("strategy").value };
}

function fillSelect(select, options, selected) {
  select.innerHTML = "";
  Object.entries(options).forEach(([value, label]) => {
    const opt = el("option", "", label);
    opt.value = value;
    select.appendChild(opt);
  });
  if (selected) select.value = selected;
}

// ---------- RENDERERS ----------
const STAGE_LABELS = {
  chunk: "Chunk", embed: "Embed", store: "Store", build: "Index build", embed_query: "Embed query",
  search: "Search", augment: "Augment", generate: "Generate",
};

function renderTiming(container, timings) {
  container.innerHTML = "";
  const entries = Object.entries(timings).filter(([, v]) => v !== undefined);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  entries.forEach(([name, ms]) => {
    const row = el("div", "wf-row");
    row.appendChild(el("span", "wf-name", STAGE_LABELS[name] || name));
    const track = el("div", "wf-track");
    const fill = el("div", "wf-fill");
    fill.style.width = `${Math.max(2, (ms / max) * 100)}%`;
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", "wf-ms", `${ms} ms`));
    container.appendChild(row);
  });
  container.appendChild(el("p", "hint", `Total: ${Math.round(total)} ms`));
}

function renderBudget(container, tb) {
  container.innerHTML = "";
  const used = tb.instructions_tokens + tb.context_tokens;
  const pct = (n) => Math.max(0.5, (n / tb.window_tokens) * 100);
  const track = el("div", "budget-track");
  const a = el("div", "budget-instr"); a.style.width = `${pct(tb.instructions_tokens)}%`;
  const b = el("div", "budget-ctx"); b.style.width = `${pct(tb.context_tokens)}%`;
  a.title = `Instructions + question: ~${tb.instructions_tokens} tokens`;
  b.title = `Retrieved context: ~${tb.context_tokens} tokens`;
  track.append(a, b);
  container.appendChild(track);
  const reported = tb.reported_prompt_tokens ? ` (provider reported ${tb.reported_prompt_tokens})` : "";
  container.appendChild(el("p", "hint",
    `~${used} of ${tb.window_tokens.toLocaleString()} window tokens used${reported}: ` +
    `${tb.context_tokens} context + ${tb.instructions_tokens} instructions/question.`));
}

// hits: retrieval results. links: optional Map "hit:sentence" -> colour index (for attribution).
function renderChunkCards(container, hits, links) {
  container.innerHTML = "";
  if (!hits.length) {
    container.appendChild(el("p", "hint", "Nothing matched. (BM25 returns nothing when no query word appears in any chunk.)"));
    return;
  }
  hits.forEach((h, hIdx) => {
    const card = el("div", "chunk-card");
    const head = el("div", "chunk-head");
    head.appendChild(el("strong", "", `#${h.rank}`));
    head.appendChild(el("span", "", `chunk ${h.chunk_id} · chars ${h.start}-${h.end}`));
    card.appendChild(head);

    const body = el("p", "chunk-text");
    if (h.sentences) {
      h.sentences.forEach((s, sIdx) => {
        const span = el("span", "sent", s);
        span.dataset.key = `${hIdx}:${sIdx}`;
        if (links && links.has(span.dataset.key)) span.classList.add("link", `lc${links.get(span.dataset.key) % 5}`);
        body.appendChild(span);
      });
    } else {
      body.textContent = h.text;
    }
    card.appendChild(body);

    const track = el("div", "score-bar-track");
    const fill = el("div", "score-bar-fill");
    fill.style.width = `${Math.max(2, Math.min(100, Math.round(h.similarity * 100)))}%`;
    track.appendChild(fill);
    card.appendChild(track);
    card.appendChild(el("span", "score-label",
      `${h.metric.split(" (")[0]}: ${h.score} · cosine similarity ${h.similarity}`));
    if (h.matched_terms && h.matched_terms.length) {
      const terms = el("div", "terms");
      h.matched_terms.forEach((t) => terms.appendChild(el("span", "term", t)));
      card.appendChild(terms);
    }
    container.appendChild(card);
  });
}

async function loadConfig() {
  state.cfg = await callApi("GET", "/api/config");
  const s = $("llm-status");
  s.textContent = state.cfg.llm_configured
    ? `Generation: Groq · ${state.cfg.llm_model}`
    : "No GROQ_API_KEY set: retrieval tools work, generation is disabled.";
  s.className = "status " + (state.cfg.llm_configured ? "ok" : "warn");
}
