"""Offline test setup: a deterministic hashed bag-of-words embedder and a scripted LLM,
so the suite needs no model download and no API key."""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embedders  # noqa: E402
import llm  # noqa: E402
import store  # noqa: E402

DIM = 128


class FakeEmbedder:
    """Words hash to buckets, so texts sharing words are close: enough to test ranking logic."""

    def __init__(self, key):
        self.key = key

    def encode(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row]) or 1.0
            out[row] /= norm
        return out

    def encode_query(self, text):
        return self.encode([text])[0]


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(embedders, "_cache", {k: FakeEmbedder(k) for k in embedders.EMBEDDERS})
    store.reset_all()
    yield
    store.reset_all()


@pytest.fixture
def fake_llm(monkeypatch):
    """Records prompts; answers with whatever `replies` says (default: echo a canned sentence)."""
    calls = {"prompts": [], "reply": "The warranty lasts 24 months."}

    def generate(prompt):
        calls["prompts"].append(prompt)
        return {"text": calls["reply"], "input_tokens": 100, "output_tokens": 10}

    monkeypatch.setattr(llm, "generate", generate)
    monkeypatch.setattr(llm, "is_configured", lambda: True)
    return calls


DOC = """Warranty and returns
The router is covered by a 24-month limited warranty from the date of purchase.

Firmware errors
Error ERR-4471 means the firmware signature check failed during an update.

Factory reset
Hold the reset button for ten seconds until the light pulses amber.

Mesh pairing
Place the new node within ten metres of the main router and power it on.
"""


@pytest.fixture
def doc():
    return DOC
