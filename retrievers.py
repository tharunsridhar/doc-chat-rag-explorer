"""Retrieval strategies: dense, MMR, BM25 and hybrid (reciprocal rank fusion)."""
import math
import time

import numpy as np

import embedders
from store import Index, tokenize

STRATEGIES = {
    "dense": "Dense similarity",
    "mmr": "Dense + MMR (diverse)",
    "bm25": "BM25 (keywords)",
    "hybrid": "Hybrid (dense + BM25, RRF)",
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "what", "how", "why", "when", "where", "does", "do", "did", "it", "my", "i", "me", "be",
    "this", "that", "with", "as", "at", "by", "from", "can", "should", "s",
}
RRF_K = 60
MMR_LAMBDA = 0.5


def _bm25_scores(index: Index, query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 with the always-positive Lucene idf, so tiny corpora behave sanely."""
    n = len(index.doc_freqs)
    lengths = np.array([sum(d.values()) for d in index.doc_freqs], dtype=np.float64)
    avg_len = lengths.mean() if n and lengths.mean() > 0 else 1.0
    scores = np.zeros(n)
    for term in set(query_tokens):
        df = sum(1 for d in index.doc_freqs if term in d)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = np.array([d.get(term, 0) for d in index.doc_freqs], dtype=np.float64)
        scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / avg_len))
    return scores


def _dense_order(index: Index, qvec: np.ndarray, n: int) -> list[int]:
    """Ranked chunk ids straight from the Chroma collection."""
    res = index.collection.query(
        query_embeddings=[qvec.tolist()], n_results=min(n, len(index.chunks))
    )
    return [int(i) for i in res["ids"][0]]


def _mmr_order(index: Index, qvec: np.ndarray, k: int) -> list[int]:
    candidates = _dense_order(index, qvec, max(k * 4, 20))
    sims = index.vectors @ qvec
    selected: list[int] = []
    while candidates and len(selected) < k:
        def score(i):
            redundancy = max((float(index.vectors[i] @ index.vectors[j]) for j in selected), default=0.0)
            return MMR_LAMBDA * float(sims[i]) - (1 - MMR_LAMBDA) * redundancy

        best = max(candidates, key=score)
        selected.append(best)
        candidates.remove(best)
    return selected


def _bm25_order(scores: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k] if scores[i] > 0]


def retrieve_hits(index: Index, query: str, k: int, strategy: str) -> dict:
    """Run one strategy. Returns hits (best first), the query vector and stage timings."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {', '.join(STRATEGIES)}")
    k = max(1, min(k, len(index.chunks)))

    t0 = time.perf_counter()
    qvec = embedders.get_embedder(index.embedder_key).encode_query(query)
    t1 = time.perf_counter()

    q_tokens = tokenize(query)
    sims = index.vectors @ qvec  # cosine similarity to every chunk (vectors are normalised)
    bm25 = _bm25_scores(index, q_tokens) if strategy in ("bm25", "hybrid") else None

    if strategy == "dense":
        order = _dense_order(index, qvec, k)
        native = {i: 1 - float(sims[i]) for i in order}
        metric = "cosine distance (lower = closer)"
    elif strategy == "mmr":
        order = _mmr_order(index, qvec, k)
        native = {i: 1 - float(sims[i]) for i in order}
        metric = "cosine distance (lower = closer)"
    elif strategy == "bm25":
        order = _bm25_order(bm25, k)
        native = {i: float(bm25[i]) for i in order}
        metric = "BM25 score (higher = better)"
    else:
        fetch = min(len(index.chunks), max(k * 4, 20))
        fused: dict[int, float] = {}
        for rank, i in enumerate(_dense_order(index, qvec, fetch)):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
        for rank, i in enumerate(_bm25_order(bm25, fetch)):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        native = {i: fused[i] for i in order}
        metric = "RRF score (higher = better)"
    t2 = time.perf_counter()

    terms = [t for t in dict.fromkeys(q_tokens) if t not in _STOPWORDS]
    hits = []
    for rank, i in enumerate(order):
        chunk = index.chunks[i]
        hits.append({
            "rank": rank + 1,
            "chunk_id": chunk.id,
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.end,
            "score": round(native[i], 4),
            "metric": metric,
            "distance": round(1 - float(sims[i]), 4),
            "similarity": round(float(sims[i]), 4),
            "matched_terms": [t for t in terms if t in index.doc_freqs[i]],
        })
    return {
        "hits": hits,
        "qvec": qvec,
        "timings_ms": {
            "embed_query": round((t1 - t0) * 1000, 1),
            "search": round((t2 - t1) * 1000, 1),
        },
    }
