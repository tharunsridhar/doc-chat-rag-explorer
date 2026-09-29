"""Orchestration: build -> retrieve -> augment -> generate, plus the debugging tools
(chunk preview, embedding map, breakage experiments, config comparison, diagnosis)."""
import time
from concurrent.futures import ThreadPoolExecutor

import analysis
import embedders
import llm
import store
from retrievers import STRATEGIES, retrieve_hits
from samples import DEFAULT_DISTRACTOR, SAMPLE_QUESTIONS, SAMPLE_TEXT

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about an uploaded document.
Answer using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}"""

NO_CONTEXT_TEMPLATE = """You are a helpful assistant answering questions about an uploaded document.
You cannot see the document. Answer the question below as best you can.

Question: {question}"""

EXPERIMENTS = {
    "no_context": "No context: the model answers with no retrieval at all",
    "distractor": "Distractor: an irrelevant, confident passage is injected at rank 1",
    "shuffle_middle": "Shuffle: the best chunk is buried in the middle of the context",
}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def defaults() -> dict:
    return {
        "chunk_size": store.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": store.DEFAULT_CHUNK_OVERLAP,
        "embedder": store.DEFAULT_EMBEDDER,
        "k": 4,
        "strategy": "dense",
    }


def app_config() -> dict:
    return {
        "defaults": defaults(),
        "embedders": {k: v["label"] for k, v in embedders.EMBEDDERS.items()},
        "strategies": STRATEGIES,
        "experiments": EXPERIMENTS,
        "llm_configured": llm.is_configured(),
        "llm_model": llm.GROQ_MODEL,
        "context_window_tokens": llm.MODEL_CONTEXT_TOKENS,
        "sample": {"text": SAMPLE_TEXT, "questions": SAMPLE_QUESTIONS},
        "default_distractor": DEFAULT_DISTRACTOR,
        "max_text_chars": store.MAX_TEXT_CHARS,
    }


# ---------- CHUNKING LAB (no embedding, so it is instant) ----------
def chunk_preview(text: str, chunk_size: int, chunk_overlap: int) -> dict:
    chunks = store.split_text(text, chunk_size, chunk_overlap)
    overlaps = []
    for prev, nxt in zip(chunks, chunks[1:]):
        if nxt.start < prev.end:
            overlaps.append([nxt.start, prev.end])
    return {
        "chunks": [{"id": c.id, "start": c.start, "end": c.end} for c in chunks],
        "overlaps": overlaps,
        "chunk_count": len(chunks),
        "avg_chars": round(sum(len(c.text) for c in chunks) / len(chunks)) if chunks else 0,
    }


# ---------- BUILD (chunk -> embed -> store) ----------
def build(text: str, chunk_size: int, chunk_overlap: int, embedder: str, session_id: str | None = None) -> dict:
    store.validate_chunking(chunk_size, chunk_overlap)
    embedders.get_embedder(embedder)  # validates the key before any work
    session = store.get_session(session_id) if session_id else store.create_session(text)
    index, cached = store.ensure_index(session, chunk_size, chunk_overlap, embedder, make_main=True)
    return {
        "session_id": session.id,
        "chunk_count": len(index.chunks),
        "chunk_preview": [c.text for c in index.chunks[:3]],
        "chunks": [{"id": c.id, "start": c.start, "end": c.end} for c in index.chunks],
        "config": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "embedder": embedder},
        "timings_ms": {"chunk": 0.0, "embed": 0.0, "store": 0.0} if cached else index.timings_ms,
        "cached": cached,
        "text_chars": len(session.text),
    }


# ---------- RETRIEVE ----------
def retrieve(session_id: str, question: str, k: int, strategy: str) -> dict:
    index = store.main_index(store.get_session(session_id))
    result = retrieve_hits(index, question, k, strategy)
    return {
        "retrieved": result["hits"],
        "timings_ms": result["timings_ms"],
        "config": {"k": k, "strategy": strategy, **_index_config(index)},
    }


def _index_config(index: store.Index) -> dict:
    return {"chunk_size": index.chunk_size, "chunk_overlap": index.chunk_overlap, "embedder": index.embedder_key}


# ---------- AUGMENT + GENERATE ----------
def build_prompt(question: str, texts: list[str]) -> str:
    return PROMPT_TEMPLATE.format(context="\n\n".join(texts), question=question)


def token_budget(question: str, texts: list[str], usage: dict | None = None) -> dict:
    context = "\n\n".join(texts)
    overhead = PROMPT_TEMPLATE.format(context="", question=question)
    return {
        "instructions_tokens": analysis.estimate_tokens(overhead),
        "context_tokens": analysis.estimate_tokens(context) if context else 0,
        "window_tokens": llm.MODEL_CONTEXT_TOKENS,
        "reported_prompt_tokens": (usage or {}).get("input_tokens"),
        "reported_output_tokens": (usage or {}).get("output_tokens"),
    }


def _generate(prompt: str) -> dict:
    t0 = time.perf_counter()
    out = llm.generate(prompt)
    out["generate_ms"] = _ms(t0)
    return out


def ask(session_id: str, question: str, k: int, strategy: str) -> dict:
    session = store.get_session(session_id)
    index = store.main_index(session)
    result = retrieve_hits(index, question, k, strategy)
    hits = result["hits"]
    texts = [h["text"] for h in hits]

    t0 = time.perf_counter()
    prompt = build_prompt(question, texts)
    augment_ms = _ms(t0)
    gen = _generate(prompt)

    attribution = analysis.attribute(gen["text"], hits, index.embedder_key)
    timings = {**result["timings_ms"], "augment": augment_ms, "generate": gen["generate_ms"]}
    return {
        "answer": gen["text"],
        "retrieved": hits,
        "attribution": attribution["answer_sentences"],
        "prompt_sent": prompt,
        "timings_ms": timings,
        "token_budget": token_budget(question, texts, gen),
        "config": {"k": k, "strategy": strategy, **_index_config(index)},
    }


# ---------- CONTROLLED BREAKAGE ----------
def _shuffle_middle(hits: list[dict]) -> list[dict]:
    if len(hits) < 3:
        return list(hits)
    best, rest = hits[0], hits[1:]
    weak_first = rest[::-1]
    mid = len(hits) // 2
    return weak_first[:mid] + [best] + weak_first[mid:]


def _run_side(question: str, texts: list[str] | None) -> dict:
    """texts=None means 'no retrieval at all'."""
    prompt = NO_CONTEXT_TEMPLATE.format(question=question) if texts is None else build_prompt(question, texts)
    gen = _generate(prompt)
    return {
        "answer": gen["text"],
        "prompt_sent": prompt,
        "generate_ms": gen["generate_ms"],
        "context_order": [] if texts is None else [t[:60] for t in texts],
    }


def experiment(session_id: str, question: str, k: int, strategy: str, mode: str,
               distractor: str | None = None) -> dict:
    if mode not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{mode}'. Options: {', '.join(EXPERIMENTS)}")
    index = store.main_index(store.get_session(session_id))
    hits = retrieve_hits(index, question, k, strategy)["hits"]
    baseline_texts = [h["text"] for h in hits]

    note = None
    if mode == "no_context":
        broken_texts = None
    elif mode == "distractor":
        broken_texts = [(distractor or DEFAULT_DISTRACTOR).strip()] + baseline_texts
    else:
        broken_texts = [h["text"] for h in _shuffle_middle(hits)]
        if len(hits) < 3:
            note = "Shuffle needs k >= 3 to bury a chunk; nothing was moved."

    with ThreadPoolExecutor(max_workers=2) as pool:
        base_f = pool.submit(_run_side, question, baseline_texts)
        broken_f = pool.submit(_run_side, question, broken_texts)
        baseline, broken = base_f.result(), broken_f.result()
    return {"mode": mode, "description": EXPERIMENTS[mode], "baseline": baseline, "experiment": broken,
            "note": note, "best_chunk_id": hits[0]["chunk_id"] if hits else None}


# ---------- SIDE-BY-SIDE CONFIG COMPARISON ----------
def _run_config(session: store.Session, question: str, cfg: dict, skip_llm: bool) -> dict:
    t0 = time.perf_counter()
    index, cached = store.ensure_index(session, cfg["chunk_size"], cfg["chunk_overlap"], cfg["embedder"])
    build_ms = 0.0 if cached else _ms(t0)
    result = retrieve_hits(index, question, cfg["k"], cfg["strategy"])
    hits = result["hits"]
    texts = [h["text"] for h in hits]
    side = {
        "config": cfg,
        "chunk_count": len(index.chunks),
        "retrieved": hits,
        "answer": None,
        "answer_error": None,
        "timings_ms": {"build": build_ms, **result["timings_ms"]},
        "token_budget": token_budget(question, texts),
    }
    if not skip_llm:
        try:
            gen = _generate(build_prompt(question, texts))
            side["answer"] = gen["text"]
            side["timings_ms"]["generate"] = gen["generate_ms"]
            side["token_budget"] = token_budget(question, texts, gen)
        except (llm.LLMError, llm.LLMUnavailable) as exc:
            side["answer_error"] = str(exc)
    return side


def _span_overlap(a_hits: list[dict], b_hits: list[dict]) -> float:
    """Fraction of A's retrieved text spans that intersect any of B's (works across chunk sizes)."""
    if not a_hits:
        return 0.0
    shared = sum(1 for a in a_hits if any(a["start"] < b["end"] and b["start"] < a["end"] for b in b_hits))
    return round(shared / len(a_hits), 2)


def compare(session_id: str, question: str, cfg_a: dict, cfg_b: dict, skip_llm: bool) -> dict:
    session = store.get_session(session_id)
    for cfg in (cfg_a, cfg_b):
        store.validate_chunking(cfg["chunk_size"], cfg["chunk_overlap"])
        embedders.get_embedder(cfg["embedder"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_run_config, session, question, cfg_a, skip_llm)
        fb = pool.submit(_run_config, session, question, cfg_b, skip_llm)
        a, b = fa.result(), fb.result()
    return {"a": a, "b": b, "span_overlap": _span_overlap(a["retrieved"], b["retrieved"])}


# ---------- EMBEDDING SPACE ----------
def embedding_space(session_id: str, question: str, k: int, strategy: str) -> dict:
    index = store.main_index(store.get_session(session_id))
    result = retrieve_hits(index, question, k, strategy)
    proj = analysis.pca_2d(index.vectors, result["qvec"])
    ranks = {h["chunk_id"]: h for h in result["hits"]}
    sims = index.vectors @ result["qvec"]
    points = []
    for c, (x, y) in zip(index.chunks, proj["points"]):
        hit = ranks.get(c.id)
        points.append({
            "chunk_id": c.id,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "retrieved": hit is not None,
            "rank": hit["rank"] if hit else None,
            "distance": round(1 - float(sims[c.id]), 4),
            "preview": c.text[:110],
        })
    return {
        "points": points,
        "query": {"x": round(float(proj["query"][0]), 4), "y": round(float(proj["query"][1]), 4)},
        "explained_variance": [round(v, 3) for v in proj["explained"]],
        "strategy": strategy,
    }


# ---------- "WHY DID IT FAIL?" DIAGNOSIS ----------
WIDE_DOC_CHARS = 12000


def diagnose(session_id: str, question: str, answer: str, retrieved_ids: list[int]) -> dict:
    index = store.main_index(store.get_session(session_id))
    valid = {c.id for c in index.chunks}
    retrieved_ids = [i for i in retrieved_ids if i in valid]

    wide = retrieve_hits(index, question, 12, "hybrid")
    wide_ids = [h["chunk_id"] for h in wide["hits"]]
    retrieved_text = "\n\n".join(index.chunks[i].text for i in retrieved_ids)
    full_text = "\n\n".join(c.text for c in index.chunks)
    partial = len(full_text) > WIDE_DOC_CHARS
    wide_text = "\n\n".join(index.chunks[i].text for i in wide_ids) if partial else full_text

    judged = None
    if llm.is_configured():
        try:
            judged = analysis.judge(question, answer, retrieved_text, wide_text)
        except (llm.LLMError, llm.LLMUnavailable):
            judged = None

    if judged is not None:
        verdict, explanation = analysis.verdict_from_judgement(judged)
        method = "llm-judge"
    else:
        verdict, explanation = analysis.heuristic_verdict(
            index, question, answer, retrieved_ids, wide["qvec"], wide_ids
        )
        method = "similarity-heuristic"
    note = (
        "The document is long, so the judge saw only the 12 most relevant chunks, not the whole thing."
        if partial and method == "llm-judge" else None
    )
    return {
        "verdict": verdict,
        "label": analysis.VERDICT_LABELS[verdict],
        "explanation": explanation,
        "method": method,
        "note": note,
    }
