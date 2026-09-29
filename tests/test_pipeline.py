import pytest

import analysis
import rag_pipeline as rag
import store


def build(doc, size=120, overlap=20, embedder="minilm"):
    return rag.build(doc, size, overlap, embedder)


# ---------- chunking ----------
def test_chunk_preview_offsets_match_source(doc):
    preview = rag.chunk_preview(doc, 120, 20)
    assert preview["chunk_count"] > 1
    for c in preview["chunks"]:
        assert doc[c["start"]:c["end"]].strip()  # offsets point at real text


def test_chunk_preview_reports_overlaps(doc):
    prose = " ".join(f"word{i}" for i in range(200))  # one long paragraph forces mid-text cuts
    with_overlap = rag.chunk_preview(prose, 200, 80)
    without = rag.chunk_preview(prose, 200, 0)
    assert len(with_overlap["overlaps"]) >= 1
    assert without["overlaps"] == []


@pytest.mark.parametrize("size,overlap", [(40, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_chunking_rejected(doc, size, overlap):
    with pytest.raises(ValueError):
        rag.chunk_preview(doc, size, overlap)


# ---------- sessions ----------
def test_sessions_are_isolated():
    a = build("Apples are red fruit that grow on trees in orchards every autumn season.")
    b = build("Submarines travel deep under the ocean using ballast tanks and propellers.")
    ra = rag.retrieve(a["session_id"], "apples orchards", 1, "dense")["retrieved"]
    rb = rag.retrieve(b["session_id"], "apples orchards", 1, "dense")["retrieved"]
    assert "Apples" in ra[0]["text"]
    assert "Submarines" in rb[0]["text"]


def test_unknown_session_raises_key_error():
    with pytest.raises(KeyError):
        rag.retrieve("nope", "q", 4, "dense")


def test_lru_eviction_drops_oldest(monkeypatch):
    monkeypatch.setattr(store, "MAX_SESSIONS", 2)
    first = build("first document about cats and dogs living together happily")["session_id"]
    build("second document about birds")
    build("third document about fish")
    assert not store.has_session(first)
    assert store.session_count() == 2


def test_rebuild_reuses_text_and_changes_config(doc):
    first = build(doc, 120, 20)
    second = rag.build("", 300, 30, "minilm", session_id=first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert second["chunk_count"] < first["chunk_count"]


def test_text_length_limit(monkeypatch):
    monkeypatch.setattr(store, "MAX_TEXT_CHARS", 50)
    with pytest.raises(ValueError):
        store.create_session("x" * 51)


# ---------- retrieval strategies ----------
@pytest.mark.parametrize("strategy", ["dense", "mmr", "bm25", "hybrid"])
def test_every_strategy_finds_the_warranty_chunk(doc, strategy):
    sid = build(doc)["session_id"]
    hits = rag.retrieve(sid, "how long is the warranty", 2, strategy)["retrieved"]
    assert hits and "warranty" in hits[0]["text"].lower()
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_matches_exact_code_and_reports_terms(doc):
    sid = build(doc)["session_id"]
    hit = rag.retrieve(sid, "ERR-4471", 1, "bm25")["retrieved"][0]
    assert "ERR-4471" in hit["text"]
    assert "4471" in hit["matched_terms"]


def test_bm25_returns_nothing_without_keyword_overlap(doc):
    sid = build(doc)["session_id"]
    assert rag.retrieve(sid, "zzzz qqqq", 3, "bm25")["retrieved"] == []


def test_k_larger_than_chunk_count_is_clamped(doc):
    sid = build(doc)["session_id"]
    hits = rag.retrieve(sid, "router", 20, "dense")["retrieved"]
    assert 1 <= len(hits) <= build(doc)["chunk_count"]


def test_unknown_strategy_rejected(doc):
    sid = build(doc)["session_id"]
    with pytest.raises(ValueError):
        rag.retrieve(sid, "q", 2, "magic")


# ---------- ask / attribution / budget ----------
def test_ask_returns_prompt_timings_and_budget(doc, fake_llm):
    sid = build(doc)["session_id"]
    out = rag.ask(sid, "how long is the warranty", 2, "dense")
    assert out["answer"] == "The warranty lasts 24 months."
    assert "how long is the warranty" in out["prompt_sent"]
    assert set(out["timings_ms"]) >= {"embed_query", "search", "augment", "generate"}
    assert out["token_budget"]["context_tokens"] > 0
    assert out["token_budget"]["reported_prompt_tokens"] == 100


def test_attribution_links_answer_to_source_sentence(doc, fake_llm):
    sid = build(doc)["session_id"]
    out = rag.ask(sid, "how long is the warranty", 2, "dense")
    link = out["attribution"][0]
    assert link["supported"]
    assert "warranty" in out["retrieved"][link["hit"]]["sentences"][link["sentence"]].lower()


def test_unrelated_answer_sentence_is_flagged_unsupported(doc, fake_llm):
    fake_llm["reply"] = "Quantum entanglement enables teleportation of qubits."
    sid = build(doc)["session_id"]
    out = rag.ask(sid, "how long is the warranty", 2, "dense")
    assert not out["attribution"][0]["supported"]


def test_split_sentences_keeps_decimals_together():
    parts = analysis.split_sentences("Version 3.5 is out. It is fast.\nNew line here.")
    assert parts[0].startswith("Version 3.5 is out.")
    assert len(parts) == 3


# ---------- experiments ----------
def test_no_context_prompt_has_no_document_text(doc, fake_llm):
    sid = build(doc)["session_id"]
    out = rag.experiment(sid, "how long is the warranty", 2, "dense", "no_context")
    assert "24-month" not in out["experiment"]["prompt_sent"]
    assert "24-month" in out["baseline"]["prompt_sent"]


def test_distractor_is_injected_first(doc, fake_llm):
    sid = build(doc)["session_id"]
    out = rag.experiment(sid, "how long is the warranty", 2, "dense", "distractor", "BOGUS PASSAGE")
    context = out["experiment"]["prompt_sent"].split("Context:\n")[1]
    assert context.startswith("BOGUS PASSAGE")
    assert "BOGUS" not in out["baseline"]["prompt_sent"]


def test_shuffle_buries_best_chunk_in_the_middle(doc, fake_llm):
    sid = build(doc, 90, 10)["session_id"]
    out = rag.experiment(sid, "how long is the warranty", 3, "dense", "shuffle_middle")
    order = out["experiment"]["context_order"]
    base = out["baseline"]["context_order"]
    assert order[1] == base[0]  # best chunk now sits in the middle of 3
    assert sorted(order) == sorted(base)


def test_shuffle_with_small_k_is_noop_with_note(doc, fake_llm):
    sid = build(doc)["session_id"]
    out = rag.experiment(sid, "warranty", 2, "dense", "shuffle_middle")
    assert out["note"]


# ---------- compare ----------
def test_compare_runs_two_configs_and_caches_indexes(doc, fake_llm):
    sid = build(doc, 120, 20)["session_id"]
    a = dict(chunk_size=120, chunk_overlap=20, embedder="minilm", k=2, strategy="dense")
    b = dict(chunk_size=250, chunk_overlap=0, embedder="bge-small", k=2, strategy="bm25")
    out = rag.compare(sid, "how long is the warranty", a, b, skip_llm=False)
    assert out["a"]["chunk_count"] != out["b"]["chunk_count"]
    assert out["a"]["answer"] and out["b"]["answer"]
    assert 0 <= out["span_overlap"] <= 1
    again = rag.compare(sid, "how long is the warranty", a, b, skip_llm=True)
    assert again["a"]["timings_ms"]["build"] == 0.0  # index reused from cache
    assert again["a"]["answer"] is None


def test_compare_reports_llm_error_per_side_without_failing(doc, monkeypatch):
    import llm

    def boom(_):
        raise llm.LLMError("rate limited")

    monkeypatch.setattr(llm, "generate", boom)
    sid = build(doc)["session_id"]
    cfg = dict(chunk_size=120, chunk_overlap=20, embedder="minilm", k=2, strategy="dense")
    out = rag.compare(sid, "warranty", cfg, cfg, skip_llm=False)
    assert out["a"]["answer_error"] == "Groq request failed: rate limited" or "rate limited" in out["a"]["answer_error"]
    assert out["a"]["retrieved"]


# ---------- embedding space ----------
def test_embedding_space_marks_retrieved_points(doc):
    sid = build(doc)["session_id"]
    space = rag.embedding_space(sid, "how long is the warranty", 2, "dense")
    assert len(space["points"]) == build(doc)["chunk_count"]
    assert sum(p["retrieved"] for p in space["points"]) == 2
    assert len(space["explained_variance"]) == 2


def test_pca_handles_a_single_chunk():
    sid = build("just one short chunk of text")["session_id"]
    space = rag.embedding_space(sid, "chunk", 1, "dense")
    assert len(space["points"]) == 1


# ---------- diagnosis ----------
def _judge(monkeypatch, verdict_json):
    import json

    import llm

    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "generate", lambda p: {"text": json.dumps(verdict_json)})


@pytest.mark.parametrize("judgement,expected", [
    (dict(answerable=True, in_retrieved=False, answer_correct=False), "retrieval_failure"),
    (dict(answerable=True, in_retrieved=True, answer_correct=False), "generation_failure"),
    (dict(answerable=True, in_retrieved=True, answer_correct=True), "ok"),
    (dict(answerable=False, in_retrieved=False, answer_correct=True), "unanswerable"),
    (dict(answerable=False, in_retrieved=False, answer_correct=False), "generation_failure"),
])
def test_llm_judge_verdicts(doc, monkeypatch, judgement, expected):
    _judge(monkeypatch, judgement)
    sid = build(doc)["session_id"]
    out = rag.diagnose(sid, "q", "a", [0])
    assert out["verdict"] == expected and out["method"] == "llm-judge"


def test_diagnosis_falls_back_to_heuristic_without_llm(doc, monkeypatch):
    import llm

    monkeypatch.setattr(llm, "is_configured", lambda: False)
    sid = build(doc)["session_id"]
    out = rag.diagnose(sid, "how long is the warranty", "I don't know.", [2])
    assert out["method"] == "similarity-heuristic"
    assert out["verdict"] == "retrieval_failure"  # the warranty chunk (0) was not retrieved


def test_diagnosis_survives_malformed_judge_output(doc, monkeypatch):
    import llm

    monkeypatch.setattr(llm, "is_configured", lambda: True)
    monkeypatch.setattr(llm, "generate", lambda p: {"text": "not json at all"})
    sid = build(doc)["session_id"]
    assert rag.diagnose(sid, "q", "a", [0])["method"] == "similarity-heuristic"
