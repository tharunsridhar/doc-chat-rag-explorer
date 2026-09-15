# Doc Chat RAG Explorer

A small full-stack app proving out the RAG pipeline from this folder
(`llamaIndex.py`, `RAG-app-simple chatbot/Langchain/rag.py`), running the full
chunk → embed → store → retrieve → augment → generate flow live, behind a
plain HTML/CSS/JS frontend.

| Panel | Pipeline stage | Source pattern |
|---|---|---|
| 1. Build | chunk, embed, store | `llamaIndex.py` sections 2-5 |
| 2. Retrieve | retrieval only, no LLM | `llamaIndex.py` section 6 (`retriever.retrieve`) |
| 3. Ask | retrieve + augment + generate | `rag.py` (`answer_question`) / `llamaIndex.py` sections 7-8 |

Splitting "retrieve" and "ask" into separate panels is deliberate — it's a common
point of confusion that RAG has a pure-retrieval step that runs with no LLM call
at all, before generation ever happens.

The frontend isn't just a results viewer — each panel explains the concept it's
demonstrating, and has a collapsible "Under the hood" box showing the raw
request/response JSON (including the exact augmented prompt sent to Groq for
the "Ask" panel).

**Stack:** FastAPI + LangChain + Chroma (in-memory, `chromadb.EphemeralClient`,
so it never touches the persisted `chroma_db/` used by the other scripts in this
folder) + local `sentence-transformers/all-MiniLM-L6-v2` embeddings (no API cost)
+ Groq for generation.

## Run it

From the repo root (needs the repo's `.env` with `GROQ_API_KEY` set):

```bash
pip install -r "Projects/3-Doc-Chat-RAG-Explorer/requirements.txt"
python -m uvicorn app:app --reload --app-dir "Projects/3-Doc-Chat-RAG-Explorer"
```

Then open http://localhost:8000 (or whichever port you pass with `--port`).
Interactive API docs (Swagger UI) are at `/docs`.

Paste a paragraph or two into panel 1 (or upload a `.txt` file), build the
index, then ask it questions in panels 2 and 3.
