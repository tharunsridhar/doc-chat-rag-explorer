"""Thin Groq wrapper. The client is created lazily so every retrieval-only
feature works without an API key."""
import os
import re

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MODEL_CONTEXT_TOKENS = int(os.getenv("MODEL_CONTEXT_TOKENS", "32768"))

_client = None


class LLMUnavailable(RuntimeError):
    """No API key configured."""


class LLMError(RuntimeError):
    """The provider call failed."""


def is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _get_client():
    global _client
    if not is_configured():
        raise LLMUnavailable("GROQ_API_KEY is not set. Add it to .env to enable generation.")
    if _client is None:
        from langchain_groq import ChatGroq

        _client = ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)
    return _client


def strip_thinking(text: str) -> str:
    """Reasoning models may emit <think>...</think> blocks; the UI only wants the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(prompt: str) -> dict:
    client = _get_client()
    try:
        msg = client.invoke(prompt)
    except Exception as exc:  # provider errors vary widely; surface them as one type
        raise LLMError(f"Groq request failed: {exc}") from exc
    usage = getattr(msg, "usage_metadata", None) or {}
    return {
        "text": strip_thinking(msg.content),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
