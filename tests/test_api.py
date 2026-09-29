import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


@pytest.fixture
def sid(doc):
    r = client.post("/api/build-from-text", json={"text": doc, "chunk_size": 120, "chunk_overlap": 20})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_config_lists_options():
    body = client.get("/api/config").json()
    assert {"dense", "bm25", "hybrid", "mmr"} <= set(body["strategies"])
    assert body["sample"]["questions"]


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_index_page_is_served():
    assert "RAG Explorer" in client.get("/").text


def test_build_and_retrieve(sid):
    r = client.post("/api/retrieve", json={"session_id": sid, "question": "warranty", "k": 2})
    assert r.status_code == 200
    assert r.json()["retrieved"]


def test_empty_text_rejected():
    assert client.post("/api/build-from-text", json={"text": "  "}).status_code == 400


def test_bad_overlap_gives_400_not_500(doc):
    r = client.post("/api/build-from-text", json={"text": doc, "chunk_size": 100, "chunk_overlap": 100})
    assert r.status_code == 400
    assert "overlap" in r.json()["detail"]


def test_out_of_range_params_give_422(doc):
    assert client.post("/api/build-from-text", json={"text": doc, "chunk_size": 10}).status_code == 422
    assert client.post("/api/retrieve", json={"session_id": "x", "question": "q", "k": 99}).status_code == 422


def test_unknown_session_is_404():
    r = client.post("/api/retrieve", json={"session_id": "nope", "question": "q"})
    assert r.status_code == 404


def test_file_upload_with_chunk_settings(doc):
    r = client.post(
        "/api/build-from-file",
        files={"file": ("doc.txt", doc.encode())},
        data={"chunk_size": "150", "chunk_overlap": "10", "embedder": "minilm"},
    )
    assert r.status_code == 200 and r.json()["config"]["chunk_size"] == 150


def test_binary_upload_rejected():
    r = client.post("/api/build-from-file", files={"file": ("x.bin", b"\xff\xfe\x00\x81")})
    assert r.status_code == 400


def test_ask_without_api_key_is_503(sid, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import llm

    monkeypatch.setattr(llm, "_client", None)
    r = client.post("/api/ask", json={"session_id": sid, "question": "warranty"})
    assert r.status_code == 503
    assert "GROQ_API_KEY" in r.json()["detail"]


def test_ask_and_diagnose_end_to_end(sid, fake_llm):
    ask = client.post("/api/ask", json={"session_id": sid, "question": "how long is the warranty"}).json()
    assert ask["answer"]
    d = client.post("/api/diagnose", json={
        "session_id": sid, "question": "how long is the warranty", "answer": ask["answer"],
        "retrieved_ids": [h["chunk_id"] for h in ask["retrieved"]],
    })
    assert d.status_code == 200 and d.json()["verdict"] in {"ok", "retrieval_failure", "generation_failure", "unanswerable"}


def test_compare_endpoint(sid):
    side = {"chunk_size": 120, "chunk_overlap": 20, "embedder": "minilm", "k": 2, "strategy": "dense"}
    r = client.post("/api/compare", json={
        "session_id": sid, "question": "warranty", "a": side,
        "b": {**side, "strategy": "bm25"}, "skip_llm": True,
    })
    assert r.status_code == 200 and r.json()["a"]["answer"] is None


def test_experiment_rejects_unknown_mode(sid):
    r = client.post("/api/experiment", json={"session_id": sid, "question": "q", "mode": "nope"})
    assert r.status_code == 422


def test_chunk_preview_endpoint(doc):
    r = client.post("/api/chunk-preview", json={"text": doc, "chunk_size": 100, "chunk_overlap": 20})
    assert r.status_code == 200 and r.json()["chunk_count"] > 1
