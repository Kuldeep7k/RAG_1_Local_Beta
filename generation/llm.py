"""Calls Ollama's local chat API for generation."""
import requests

from config import OLLAMA_BASE_URL, LLM_MODEL
from ollama_client import ensure_ollama_ready


def generate(messages: list, temperature: float = 0.0) -> str:
    ensure_ollama_ready(LLM_MODEL)
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    try:
        content = response.json()["message"]["content"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("LLM provider returned an invalid response") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM provider returned an empty response")
    return content
