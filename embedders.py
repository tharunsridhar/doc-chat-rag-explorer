"""Embedding model registry. Models load lazily so the app boots fast and only
pays for the models a user actually picks."""
import threading

import numpy as np

EMBEDDERS = {
    "minilm": {
        "label": "MiniLM-L6 (384d, default)",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
    },
    "bge-small": {
        "label": "BGE-small v1.5 (384d)",
        "model": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "paraphrase-l3": {
        "label": "paraphrase-MiniLM-L3 (384d, tiny)",
        "model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "query_prefix": "",
    },
}

_cache: dict[str, "Embedder"] = {}
_cache_lock = threading.Lock()


class Embedder:
    def __init__(self, key: str):
        self.key = key
        self.spec = EMBEDDERS[key]
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.spec["model"])
            return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalised float32 vectors, shape (len(texts), dim)."""
        vecs = self._get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([self.spec["query_prefix"] + text])[0]


def get_embedder(key: str) -> Embedder:
    if key not in EMBEDDERS:
        raise ValueError(f"Unknown embedder '{key}'. Options: {', '.join(EMBEDDERS)}")
    with _cache_lock:
        if key not in _cache:
            _cache[key] = Embedder(key)
        return _cache[key]
