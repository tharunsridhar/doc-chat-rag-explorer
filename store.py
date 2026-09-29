"""Session + index storage.

Every session owns its own Chroma collections, so two visitors can never read or
overwrite each other's data. Sessions expire (TTL) and are capped (LRU) so a public
demo cannot grow without bound.
"""
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import chromadb
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

import embedders

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "25"))
SESSION_TTL_S = int(os.getenv("SESSION_TTL_MINUTES", "120")) * 60
MAX_INDEXES_PER_SESSION = 6
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "200000"))

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDER = "minilm"

_client = chromadb.EphemeralClient()
_sessions: "OrderedDict[str, Session]" = OrderedDict()
_lock = threading.RLock()


@dataclass
class Chunk:
    id: int
    text: str
    start: int
    end: int


@dataclass
class Index:
    key: tuple
    chunks: list[Chunk]
    vectors: np.ndarray
    collection: object
    doc_freqs: list  # per-chunk term counts, for BM25
    timings_ms: dict = field(default_factory=dict)

    @property
    def chunk_size(self) -> int:
        return self.key[0]

    @property
    def chunk_overlap(self) -> int:
        return self.key[1]

    @property
    def embedder_key(self) -> str:
        return self.key[2]


@dataclass
class Session:
    id: str
    text: str
    main_key: tuple | None = None
    indexes: dict = field(default_factory=dict)
    last_used: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)


_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 characters.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    validate_chunking(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    docs = splitter.create_documents([text])
    return [
        Chunk(i, d.page_content, d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content))
        for i, d in enumerate(docs)
    ]


def _build_index(session: Session, key: tuple) -> Index:
    chunk_size, chunk_overlap, embedder_key = key
    embedder = embedders.get_embedder(embedder_key)

    t0 = time.perf_counter()
    chunks = split_text(session.text, chunk_size, chunk_overlap)
    t1 = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks])
    t2 = time.perf_counter()

    collection = _client.create_collection(
        name=f"s{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[str(c.id) for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c.text for c in chunks],
    )
    doc_freqs = []
    for c in chunks:
        counts: dict[str, int] = {}
        for tok in tokenize(c.text):
            counts[tok] = counts.get(tok, 0) + 1
        doc_freqs.append(counts)
    t3 = time.perf_counter()

    return Index(
        key=key,
        chunks=chunks,
        vectors=vectors,
        collection=collection,
        doc_freqs=doc_freqs,
        timings_ms={
            "chunk": round((t1 - t0) * 1000, 1),
            "embed": round((t2 - t1) * 1000, 1),
            "store": round((t3 - t2) * 1000, 1),
        },
    )


def _drop_collection(index: Index) -> None:
    try:
        _client.delete_collection(index.collection.name)
    except Exception:
        pass  # already gone; nothing to clean up


def _drop_session(session: Session) -> None:
    for index in session.indexes.values():
        _drop_collection(index)
    session.indexes.clear()


def _evict() -> None:
    now = time.time()
    for sid in [s for s, sess in _sessions.items() if now - sess.last_used > SESSION_TTL_S]:
        _drop_session(_sessions.pop(sid))
    while len(_sessions) > MAX_SESSIONS:
        _, oldest = _sessions.popitem(last=False)
        _drop_session(oldest)


def create_session(text: str) -> Session:
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Text is too long ({len(text):,} chars). Limit is {MAX_TEXT_CHARS:,}.")
    session = Session(id=str(uuid.uuid4()), text=text)
    with _lock:
        _sessions[session.id] = session
        _evict()
    return session


def get_session(session_id: str) -> Session:
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        session.last_used = time.time()
        _sessions.move_to_end(session_id)
        return session


def has_session(session_id: str) -> bool:
    with _lock:
        return session_id in _sessions


def session_count() -> int:
    with _lock:
        return len(_sessions)


def ensure_index(session: Session, chunk_size: int, chunk_overlap: int, embedder_key: str,
                 make_main: bool = False) -> tuple[Index, bool]:
    """Return (index, was_cached). Builds lazily; configs are cached per session."""
    key = (chunk_size, chunk_overlap, embedder_key)
    with session.lock:
        index = session.indexes.get(key)
        cached = index is not None
        if index is None:
            index = _build_index(session, key)
            session.indexes[key] = index
            while len(session.indexes) > MAX_INDEXES_PER_SESSION:
                victim = next(k for k in session.indexes if k != session.main_key and k != key)
                _drop_collection(session.indexes.pop(victim))
        if make_main:
            session.main_key = key
        return index, cached


def main_index(session: Session) -> Index:
    with session.lock:
        return session.indexes[session.main_key]


def reset_all() -> None:
    """Test helper: drop every session."""
    with _lock:
        for session in _sessions.values():
            _drop_session(session)
        _sessions.clear()
