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
        self._base_url = base_url or os.getenv(
            self.BASE_URL_ENV_VAR, self.DEFAULT_BASE_URL
        )
        self._model = model or os.getenv(self.MODEL_ENV_VAR, self.DEFAULT_MODEL)

    def send_message(self, message: str, model: str = None) -> str:
        base_url = self._base_url
        model = model or self._model

        url = f"{base_url}/api/generate"

        payload = {"model": model, "prompt": message, "stream": False}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error communicating with Ollama: {e}")

    def is_enabled(self) -> bool:
        return (os.getenv(self.ENABLED_ENV_VAR, "false").lower() == "true") or (
            self._base_url and self._model
        )
