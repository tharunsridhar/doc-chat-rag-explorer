"""Analysis helpers: 2D projection, answer attribution, failure diagnosis."""
import json
import re

import numpy as np

import embedders
import llm
from store import Index

SUPPORT_THRESHOLD = 0.35  # min cosine similarity to call an answer sentence "supported"

_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_REFUSAL = re.compile(
    r"don'?t know|do not know|not (?:mentioned|provided|stated|specified|contain|available|covered)"
    r"|cannot (?:find|answer)|can'?t (?:find|answer)|no information|isn'?t (?:mentioned|in)",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Sentences keep their trailing whitespace, so ''.join(...) reads like the original."""
    parts, pos = [], 0
    for m in _BOUNDARY.finditer(text):
        parts.append(text[pos:m.end()])
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return [p for p in parts if p.strip()]


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def pca_2d(vectors: np.ndarray, qvec: np.ndarray) -> dict:
    """Project chunk vectors (and the query) onto the top-2 principal components."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    n_comp = min(2, *centered.shape)
    if n_comp == 0:
        return {"points": np.zeros((len(vectors), 2)), "query": np.zeros(2), "explained": [0.0, 0.0]}
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:n_comp]
    pts = centered @ comps.T
    q = (qvec - mean) @ comps.T
    if n_comp < 2:
        pts = np.hstack([pts, np.zeros((len(pts), 2 - n_comp))])
        q = np.concatenate([q, np.zeros(2 - n_comp)])
    var = s**2
    total = var.sum() or 1.0
    explained = [float(var[i] / total) if i < len(var) else 0.0 for i in range(2)]
    return {"points": pts, "query": q, "explained": explained}


def attribute(answer: str, hits: list[dict], embedder_key: str) -> dict:
    """Map each answer sentence to the retrieved sentence it most resembles.

    Adds a `sentences` list to every hit and returns the per-answer-sentence links.
    """
    embedder = embedders.get_embedder(embedder_key)
    src: list[tuple[int, int]] = []
    src_texts: list[str] = []
    for h_idx, hit in enumerate(hits):
        hit["sentences"] = split_sentences(hit["text"])
        for s_idx, sent in enumerate(hit["sentences"]):
            src.append((h_idx, s_idx))
            src_texts.append(sent)

    ans_sents = split_sentences(answer)
    if not ans_sents or not src_texts:
        return {"answer_sentences": [{"text": s, "hit": None, "sentence": None, "score": 0.0,
                                      "supported": False} for s in ans_sents]}
    sims = embedder.encode(ans_sents) @ embedder.encode(src_texts).T
    out = []
    for a_idx, sent in enumerate(ans_sents):
        best = int(np.argmax(sims[a_idx]))
        score = float(sims[a_idx][best])
        supported = score >= SUPPORT_THRESHOLD
        out.append({
            "text": sent,
            "hit": src[best][0] if supported else None,
            "sentence": src[best][1] if supported else None,
            "score": round(score, 3),
            "supported": supported,
        })
    return {"answer_sentences": out}


# ---------- failure diagnosis ----------
JUDGE_PROMPT = """You are auditing a retrieval-augmented question answering system.

DOCUMENT EXCERPTS (a wider view of the source document):
{wide}

RETRIEVED CONTEXT (what the assistant was actually shown):
{retrieved}

QUESTION: {question}

ASSISTANT ANSWER: {answer}

Reply with ONLY a JSON object with these keys:
"answerable": true if the DOCUMENT EXCERPTS contain the information needed to answer the question.
"in_retrieved": true if the RETRIEVED CONTEXT contains the information needed to answer the question.
"answer_correct": true if the assistant's answer is correct and faithful to the document
    (if the question is not answerable from the document, a refusal to answer counts as correct).
"reason": one short sentence."""


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if all(k in data for k in ("answerable", "in_retrieved", "answer_correct")) else None


def verdict_from_judgement(j: dict) -> tuple[str, str]:
    reason = str(j.get("reason", "")).strip()
    if not j["answerable"]:
        if j["answer_correct"]:
            return "unanswerable", "The document does not contain the answer, and the model correctly declined."
        return "generation_failure", (
            "The document does not contain the answer, but the model answered anyway (hallucination). " + reason
        )
    if not j["in_retrieved"]:
        return "retrieval_failure", (
            "The document does contain the answer, but the retriever did not surface it. " + reason
        )
    if j["answer_correct"]:
        return "ok", "The answer was retrieved and the model used it correctly."
    return "generation_failure", (
        "The needed information was in the retrieved chunks, but the model answered wrongly or ignored it. "
        + reason
    )


def heuristic_verdict(index: Index, question: str, answer: str, retrieved_ids: list[int], qvec: np.ndarray,
                      wide_ids: list[int]) -> tuple[str, str]:
    """No-LLM fallback based on embedding similarity only."""
    embedder = embedders.get_embedder(index.embedder_key)
    if is_refusal(answer):
        missed = [i for i in wide_ids if i not in retrieved_ids and float(index.vectors[i] @ qvec) >= 0.35]
        if missed:
            return "retrieval_failure", (
                f"The model declined, but chunk {missed[0]} looks relevant to the question and was not retrieved."
            )
        return "unanswerable", "The model declined and no unretrieved chunk looks relevant."
    ans_vecs = embedder.encode(split_sentences(answer) or [answer])
    best_retrieved = float((ans_vecs @ index.vectors[retrieved_ids].T).max()) if retrieved_ids else 0.0
    if best_retrieved >= SUPPORT_THRESHOLD:
        return "ok", "The answer closely resembles the retrieved chunks."
    return "generation_failure", "The answer does not resemble any retrieved chunk (possible hallucination)."


VERDICT_LABELS = {
    "ok": "Healthy",
    "retrieval_failure": "Retrieval failure",
    "generation_failure": "Generation failure",
    "unanswerable": "Not in document",
}


def judge(question: str, answer: str, retrieved_text: str, wide_text: str) -> dict | None:
    prompt = JUDGE_PROMPT.format(wide=wide_text, retrieved=retrieved_text, question=question, answer=answer)
    return _parse_json(llm.generate(prompt)["text"])
