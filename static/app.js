// ---------- SECTION 1: SHARED "API INSPECTOR" HELPER ----------
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
  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }

  if (inspectorEl) {
    const pre = inspectorEl.querySelector("pre");
    const shownBody = isFormData ? "(multipart file upload)" : body ? JSON.stringify(body, null, 2) : "(none)";
    pre.textContent = `${method} ${url}\n\nRequest body:\n${shownBody}\n\nResponse (${res.status}):\n${JSON.stringify(data, null, 2)}`;
    inspectorEl.hidden = false;
  }
  return data;
}

let sessionId = null;

function renderChunkList(container, chunks) {
  container.innerHTML = "";
  chunks.forEach((c) => {
    const pct = Math.max(0, Math.min(100, Math.round((1 - c.score) * 100)));
    const card = document.createElement("div");
    card.className = "chunk-card";
    card.innerHTML = `
      <p class="chunk-text">${c.text}</p>
      <div class="score-bar-track"><div class="score-bar-fill" style="width:${pct}%"></div></div>
      <span class="score-label">distance score: ${c.score.toFixed(4)} (lower = closer match)</span>
    `;
    container.appendChild(card);
  });
}

// ---------- SECTION 2: BUILD PANEL (text) ----------
const buildTextForm = document.getElementById("build-text-form");
buildTextForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = buildTextForm.querySelector("button");
  button.disabled = true;
  try {
    const text = document.getElementById("build-text").value;
    const data = await callApi(
      "POST",
      "/api/build-from-text",
      { text },
      document.getElementById("build-inspector")
    );
    onBuilt(data);
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
});

// ---------- SECTION 3: BUILD PANEL (file upload) ----------
const buildFileForm = document.getElementById("build-file-form");
buildFileForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = buildFileForm.querySelector("button");
  const fileInput = document.getElementById("build-file");
  if (!fileInput.files.length) {
    alert("Choose a .txt file first.");
    return;
  }
  button.disabled = true;
  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    const data = await callApi(
      "POST",
      "/api/build-from-file",
      formData,
      document.getElementById("build-inspector"),
      true
    );
    onBuilt(data);
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
});

function onBuilt(data) {
  sessionId = data.session_id;
  document.getElementById("build-count").textContent = data.chunk_count;
  document.getElementById("build-preview").textContent = data.chunk_preview[0] || "(empty)";
  document.getElementById("build-result").hidden = false;
}

// ---------- SECTION 4: RETRIEVE PANEL (retrieval only) ----------
const retrieveForm = document.getElementById("retrieve-form");
retrieveForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!sessionId) {
    alert("Build an index first (section 1).");
    return;
  }
  const button = retrieveForm.querySelector("button");
  button.disabled = true;
  try {
    const question = document.getElementById("retrieve-question").value;
    const data = await callApi(
      "POST",
      "/api/retrieve",
      { session_id: sessionId, question },
      document.getElementById("retrieve-inspector")
    );
    const resultEl = document.getElementById("retrieve-result");
    renderChunkList(resultEl, data.retrieved);
    resultEl.hidden = false;
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
});

// ---------- SECTION 5: ASK PANEL (full RAG) ----------
const askForm = document.getElementById("ask-form");
askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!sessionId) {
    alert("Build an index first (section 1).");
    return;
  }
  const button = askForm.querySelector("button");
  button.disabled = true;
  try {
    const question = document.getElementById("ask-question").value;
    const data = await callApi(
      "POST",
      "/api/ask",
      { session_id: sessionId, question },
      document.getElementById("ask-inspector")
    );
    document.getElementById("ask-answer").textContent = data.answer;
    document.getElementById("ask-result").hidden = false;
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
  }
});
