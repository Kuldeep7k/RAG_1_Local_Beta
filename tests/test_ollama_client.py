import unittest
from unittest.mock import patch

import requests

import ollama_client
from config import LLM_MODEL


class OllamaClientTests(unittest.TestCase):
    def test_model_readiness_is_checked_for_each_call(self):
        response = type("Response", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"models": [{"name": LLM_MODEL}]},
        })()
        with patch.object(ollama_client.requests, "get", return_value=response) as get:
            ollama_client.ensure_ollama_ready(LLM_MODEL)
            ollama_client.ensure_ollama_ready(LLM_MODEL)
        self.assertEqual(get.call_count, 2)

    def test_missing_model_is_reported(self):
        response = type("Response", (), {
            "raise_for_status": lambda self: None,
            "json": lambda self: {"models": []},
        })()
        with patch.object(ollama_client.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "ollama pull missing-model"):
                ollama_client.ensure_ollama_ready("missing-model")

    def test_unreachable_service_is_reported_without_retrying(self):
        with patch.object(
            ollama_client.requests,
            "get",
            side_effect=requests.ConnectionError(),
        ) as get:
            with self.assertRaisesRegex(RuntimeError, "Start Ollama"):
                ollama_client.ensure_ollama_ready(LLM_MODEL)
        get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
