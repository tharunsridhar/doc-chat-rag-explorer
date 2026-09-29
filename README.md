# RAG Explorer

A RAG debugger. Most RAG demos hide the pipeline behind a chat box; this one
lets you look inside every stage and break it on purpose.

Stack: FastAPI, Chroma (in-memory), sentence-transformers, Groq, plain HTML/CSS/JS.

## What you can do with it

- **Chunking lab.** Drag chunk size and overlap sliders and watch the source text
  recolour, with overlapping text shaded. Rebuild the index from the same text.
- **Embedding map.** Every chunk projected to 2D with PCA, your question dropped in as a
  point, retrieved chunks lit up and numbered by rank.
- **Compare two setups.** One question through two configs side by side: chunk size,
  overlap, embedding model, strategy, top-k. Shows what each retrieved, answered and cost.
- **Failure diagnosis.** After every answer: was the answer never retrieved
  (retrieval failure), retrieved but ignored (generation failure), or simply not in the
  document. Uses an LLM judge, with a similarity heuristic as fallback.
- **Break it on purpose.**
  - no context, to see a hallucination next to the grounded answer
  - distractor injection (editable passage placed at rank 1)
  - best chunk buried in the middle of the context
- **Answer attribution.** Each answer sentence is colour-linked to the retrieved sentence
  it most resembles. Sentences with no match get a dotted underline.
- **Timing and context budget.** Milliseconds per stage (chunk, embed, store, query embed,
  search, augment, generate) and how much of the model's context window the prompt uses.
- **Retrieval strategies.** Dense, dense + MMR, BM25 and hybrid (reciprocal rank fusion).
  The bundled sample document has one question where BM25 wins (an exact error code) and
  one where dense wins (a paraphrase).

Every panel also has an "under the hood" box with the raw request and response, including
the exact prompt sent to Groq.

## Run it

Needs Python 3.10+ and a [Groq API key](https://console.groq.com/keys). Retrieval-only
features work without a key; generation and diagnosis need it.

```bash
pip install -r requirements.txt
cp .env.example .env   # then set GROQ_API_KEY
python -m uvicorn app:app --reload
```

Open http://localhost:8000. Swagger UI is at `/docs`.

Quick tour:

- Click **Load sample document**, then **Build index**.
- Pick the sample question "Exact code (BM25 wins)" and hit **Retrieve chunks** with the
  Dense strategy, then again with BM25.
- Set top-k to 1 and hit **Get answer** to watch a retrieval failure get diagnosed.

## Configuration

All optional, set in `.env`:

- `GROQ_MODEL`: generation model (default `qwen/qwen3.8-27b`)
- `MODEL_CONTEXT_TOKENS`: window size used for the context budget bar (default 32768)
- `MAX_SESSIONS`: sessions kept in memory before the oldest is evicted (default 25)
- `SESSION_TTL_MINUTES`: idle time before a session expires (default 120)
- `MAX_TEXT_CHARS`: upload size limit (default 200000)

## Session isolation

Each document gets its own session and its own Chroma collections, so two visitors never
see or overwrite each other's data. Sessions expire after being idle and the oldest is
evicted past the cap, so a public deployment can't grow without bound.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite runs offline: a hashed bag-of-words embedder stands in for the real models and
the LLM is scripted, so no model download or API key is needed.

## Status

Checked so far:

- 53 offline tests pass
- Build, retrieve, embedding map, ask, diagnosis, "no context" breakage and compare,
  run by hand against live Groq
- Dense vs BM25 behaviour on the sample document, using the real MiniLM model

Not checked yet:

- Docker image (a `Dockerfile` is included but has never been built)
- Hugging Face Spaces deployment
- Distractor and shuffle buttons in the browser (backend logic is covered by tests)
- Mobile layout

## Docker (planned)

The `Dockerfile` is written but untested.

```bash
docker build -t rag-explorer .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key rag-explorer
```

## Deploy on Hugging Face Spaces (planned)

- Create a new Space with the **Docker** SDK and push this repo to it.
- Add this front matter to the very top of the Space's `README.md`:

```yaml
---
title: RAG Explorer
sdk: docker
app_port: 7860
---
```

- Add `GROQ_API_KEY` under Settings, Variables and secrets, as a **secret**.
- Consider lowering `MAX_SESSIONS` on the free tier (memory is limited).

## Layout

- `app.py`: FastAPI routes and request validation
- `rag_pipeline.py`: build, retrieve, ask, experiments, compare, diagnose
- `store.py`: sessions, per-session Chroma collections, eviction
- `retrievers.py`: dense, MMR, BM25, hybrid
- `analysis.py`: PCA, attribution, diagnosis
- `embedders.py`, `llm.py`: model registry and Groq wrapper
- `static/`: the frontend
- `tests/`: offline test suite
