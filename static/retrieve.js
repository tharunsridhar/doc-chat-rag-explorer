// ---------- SECTION 2: RETRIEVE + EMBEDDING SPACE ----------
const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderEmbeddingSpace(container, space) {
  container.innerHTML = "";
  const W = 440, H = 340, PAD = 34;
  const xs = space.points.map((p) => p.x).concat(space.query.x);
  const ys = space.points.map((p) => p.y).concat(space.query.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x) => PAD + ((x - minX) / (maxX - minX || 1)) * (W - 2 * PAD);
  const sy = (y) => H - PAD - ((y - minY) / (maxY - minY || 1)) * (H - 2 * PAD);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "space", role: "img",
    "aria-label": "2D projection of chunk embeddings with the query point" });
  const q = { x: sx(space.query.x), y: sy(space.query.y) };

  space.points.filter((p) => p.retrieved).forEach((p) => {
    svg.appendChild(svgEl("line", { x1: q.x, y1: q.y, x2: sx(p.x), y2: sy(p.y), class: "space-link" }));
  });
  space.points.forEach((p) => {
    const g = svgEl("g");
    g.appendChild(svgEl("title", {}, `chunk ${p.chunk_id} · distance ${p.distance}\n${p.preview}`));
    g.appendChild(svgEl("circle", { cx: sx(p.x), cy: sy(p.y), r: p.retrieved ? 9 : 6,
      class: p.retrieved ? "dot hit" : "dot" }));
    g.appendChild(svgEl("text", { x: sx(p.x), y: sy(p.y) + 3.5, class: "dot-label" },
      p.retrieved ? String(p.rank) : ""));
    if (!p.retrieved) g.appendChild(svgEl("text", { x: sx(p.x) + 9, y: sy(p.y) + 3, class: "dot-id" }, String(p.chunk_id)));
    svg.appendChild(g);
  });
  const star = svgEl("g");
  star.appendChild(svgEl("title", {}, "Your question"));
  star.appendChild(svgEl("path", { d: `M ${q.x} ${q.y - 11} L ${q.x + 4} ${q.y - 3} L ${q.x + 12} ${q.y - 3} L ${q.x + 6} ${q.y + 3} L ${q.x + 8} ${q.y + 11} L ${q.x} ${q.y + 6} L ${q.x - 8} ${q.y + 11} L ${q.x - 6} ${q.y + 3} L ${q.x - 12} ${q.y - 3} L ${q.x - 4} ${q.y - 3} Z`, class: "query-star" }));
  svg.appendChild(star);
  container.appendChild(svg);

  const [v1, v2] = space.explained_variance;
  $("space-caption").textContent =
    `Star = your question. Numbered dots = retrieved chunks by rank, grey dots = the rest (labelled by chunk id). ` +
    `Two axes keep only ${Math.round((v1 + v2) * 100)}% of the variance, so distances here are only a rough picture. ` +
    `Hover a dot for the true cosine distance.`;
}

async function runRetrieve() {
  requireSession();
  const body = baseQuery();
  const [data, space] = await Promise.all([
    callApi("POST", "/api/retrieve", body, $("retrieve-inspector")),
    callApi("POST", "/api/embedding-space", body),
  ]);
  renderChunkCards($("retrieve-result"), data.retrieved);
  renderEmbeddingSpace($("space-wrap"), space);
  renderTiming($("retrieve-timing"), data.timings_ms);
  $("retrieve-cols").hidden = false;
}

document.addEventListener("DOMContentLoaded", () => guarded($("retrieve-btn"), runRetrieve));
