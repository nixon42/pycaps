import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

# Mock the base class to avoid importing pycaps package
llm_module = MagicMock()


class Llm:
    pass


llm_module.Llm = Llm
sys.modules["pycaps.ai.llm"] = llm_module

# Mock requests
requests_mock = MagicMock()


class MockRequestException(Exception):
    pass


requests_mock.exceptions.RequestException = MockRequestException
sys.modules["requests"] = requests_mock
sys.modules["requests.exceptions"] = requests_mock.exceptions

# Load Ollama from file directly in isolation
test_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(test_dir, "../src/pycaps/ai/ollama.py"))
spec = importlib.util.spec_from_file_location("ollama_test", src_path)
ollama_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ollama_mod)
Ollama = ollama_mod.Ollama

import unittest


class TestOllama(unittest.TestCase):
    def setUp(self):
        self.ollama = Ollama()

    @patch("os.getenv")
    def test_is_enabled_true(self, mock_getenv):
        mock_getenv.return_value = "true"
        self.assertTrue(self.ollama.is_enabled())
        mock_getenv.assert_called_with("PYCAPS_OLLAMA_ENABLED", "false")

    @patch("os.getenv")
    def test_is_enabled_false(self, mock_getenv):
        mock_getenv.return_value = "false"
        self.assertFalse(self.ollama.is_enabled())
        mock_getenv.assert_called_with("PYCAPS_OLLAMA_ENABLED", "false")

    @patch("requests.post")
    @patch("os.getenv")
    def test_send_message_success(self, mock_getenv, mock_post):
        mock_getenv.side_effect = lambda key, default=None: {
            "PYCAPS_OLLAMA_BASE_URL": "http://test-url",
            "PYCAPS_OLLAMA_MODEL": "test-model",
        }.get(key, default)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = self.ollama.send_message("test message")

        self.assertEqual(response, "test response")
        mock_post.assert_called_with(
            "http://test-url/api/generate",
            json={"model": "test-model", "prompt": "test message", "stream": False},
        )

    @patch("requests.post")
    def test_send_message_failure(self, mock_post):
        mock_post.side_effect = ollama_mod.requests.exceptions.RequestException(
            "Connection error"
        )

        with self.assertRaises(RuntimeError):
            self.ollama.send_message("test message")


if __name__ == "__main__":
    unittest.main()
