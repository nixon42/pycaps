from pycaps.ai.llm import Llm
import os
import requests


class Ollama(Llm):
    ENABLED_ENV_VAR = "PYCAPS_OLLAMA_ENABLED"
    BASE_URL_ENV_VAR = "PYCAPS_OLLAMA_BASE_URL"
    MODEL_ENV_VAR = "PYCAPS_OLLAMA_MODEL"

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"

    def __init__(self, base_url: str = None, model: str = None):
        url = base_url or os.getenv(self.BASE_URL_ENV_VAR, self.DEFAULT_BASE_URL)
        # Sanitize base_url: remove trailing slash and api/generate if present
        if url:
            url = url.rstrip("/")
            if url.endswith("/api/generate"):
                url = url[: -len("/api/generate")]
            elif url.endswith("/api/chat"):
                url = url[: -len("/api/chat")]
        self._base_url = url
        self._model = model or os.getenv(self.MODEL_ENV_VAR, self.DEFAULT_MODEL)

    def send_message(self, prompt: str, model: str = None) -> str:
        model = model or self._model

        url = f"{self._base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5},
            "keep_alive": "1h",
        }

        try:
            response = requests.post(url, json=payload, timeout=900)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with Ollama: {e}")

    def is_enabled(self) -> bool:
        return (os.getenv(self.ENABLED_ENV_VAR, "false").lower() == "true") or (
            self._base_url and self._model
        )
