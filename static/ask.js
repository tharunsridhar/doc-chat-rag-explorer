// ---------- SECTION 3: ASK (answer, attribution, diagnosis, timing) ----------
function renderAnswer(container, sentences) {
  container.innerHTML = "";
  const links = new Map();
  sentences.forEach((s, i) => {
    const span = el("span", "sent", s.text + " ");
    if (s.supported) {
      const key = `${s.hit}:${s.sentence}`;
      links.set(key, i);
      span.classList.add("link", `lc${i % 5}`);
      span.dataset.key = key;
      span.title = `Most similar to retrieved chunk #${s.hit + 1} (similarity ${s.score})`;
    } else {
      span.classList.add("unsupported");
      span.title = `No retrieved sentence is similar enough (best ${s.score}). Possibly not grounded.`;
    }
    container.appendChild(span);
  });
  return links;
}

function wireAttributionHover(answerEl, chunksEl) {
  const toggle = (key, on) => {
    chunksEl.querySelectorAll(`.sent[data-key="${key}"]`).forEach((n) => n.classList.toggle("pulse", on));
    answerEl.querySelectorAll(`.sent[data-key="${key}"]`).forEach((n) => n.classList.toggle("pulse", on));
  };
  [answerEl, chunksEl].forEach((root) => {
    root.addEventListener("mouseover", (e) => {
      const n = e.target.closest(".sent.link");
      if (n) toggle(n.dataset.key, true);
    });
    root.addEventListener("mouseout", (e) => {
      const n = e.target.closest(".sent.link");
      if (n) toggle(n.dataset.key, false);
    });
  });
}

function renderDiagnosis(box, d) {
  box.innerHTML = "";
  box.className = `diagnosis ${d.verdict}`;
  box.appendChild(el("strong", "", `Diagnosis: ${d.label}`));
  box.appendChild(el("span", "", ` ${d.explanation}`));
  const how = d.method === "llm-judge" ? "LLM judge" : "similarity heuristic (no judge available)";
  box.appendChild(el("p", "hint", `Method: ${how}.${d.note ? " " + d.note : ""}`));
}

async function runAsk() {
  requireSession();
  const body = baseQuery();
  $("diagnosis").className = "diagnosis";
  $("diagnosis").textContent = "";
  const data = await callApi("POST", "/api/ask", body, $("ask-inspector"));

  const answerEl = $("ask-answer");
  const links = renderAnswer(answerEl, data.attribution);
  if (!data.attribution.length) answerEl.textContent = data.answer;
  const chunksEl = $("ask-chunks");
  renderChunkCards(chunksEl, data.retrieved, links);
  wireAttributionHover(answerEl, chunksEl);
  renderTiming($("ask-timing"), data.timings_ms);
  renderBudget($("ask-budget"), data.token_budget);
  $("ask-result").hidden = false;

  const box = $("diagnosis");
  box.className = "diagnosis pending";
  box.textContent = "Diagnosing...";
  try {
    const d = await callApi("POST", "/api/diagnose", {
      session_id: state.sessionId, question: body.question, answer: data.answer,
      retrieved_ids: data.retrieved.map((h) => h.chunk_id),
    });
    renderDiagnosis(box, d);
  } catch (err) {
    box.className = "diagnosis";
    box.textContent = "Diagnosis unavailable: " + err.message;
  }
}

document.addEventListener("DOMContentLoaded", () => guarded($("ask-btn"), runAsk));
