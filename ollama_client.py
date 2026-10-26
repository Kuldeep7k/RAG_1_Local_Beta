"""Shared Ollama availability and model checks."""
import requests

from config import OLLAMA_BASE_URL


class OllamaUnavailableError(RuntimeError):
    """Raised when the configured Ollama service cannot be used."""


def _model_names() -> set[str]:
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=3,
    )
    response.raise_for_status()
    try:
        models = response.json()["models"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaUnavailableError(
            "Ollama returned an invalid model list from /api/tags"
        ) from exc
    if not isinstance(models, list):
        raise OllamaUnavailableError("Ollama returned an invalid model list from /api/tags")
    names = set()
    for model in models:
        if isinstance(model, dict) and isinstance(model.get("name"), str):
            names.add(model["name"])
    return names


def ensure_ollama_ready(model: str) -> None:
    """Verify Ollama and ``model`` for the current operation, without caching."""
    try:
        names = _model_names()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Start Ollama before starting the worker."
        ) from exc
    if model not in names and f"{model}:latest" not in names:
        raise OllamaUnavailableError(
            f"Ollama is running at {OLLAMA_BASE_URL}, but model "
            f"{model!r} is not installed. Run: ollama pull {model}"
        )
