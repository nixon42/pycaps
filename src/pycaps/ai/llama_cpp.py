from pycaps.ai.llm import Llm
import os
import requests
from pycaps.logger import logger


class LlamaCpp(Llm):
    URL_ENV_VAR = "PYCAPS_LLAMA_CPP_URL"
    MODEL_ENV_VAR = "PYCAPS_LLAMA_CPP_MODEL"

    DEFAULT_URL = "http://localhost:8080/v1/chat/completions"
    DEFAULT_MODEL = "local_model"

    def __init__(
        self,
        url: str = None,
        model: str = None,
        temperature: float = 0.7,
        think: bool = False,
    ):
        self._url_explicit = url is not None
        self._model_explicit = model is not None

        base_url = url or os.getenv(self.URL_ENV_VAR, self.DEFAULT_URL)
        if base_url and not base_url.endswith("/v1/chat/completions"):
            if not base_url.endswith("/"):
                base_url += "/"
            base_url += "v1/chat/completions"

        self._url = base_url
        self._model = model or os.getenv(self.MODEL_ENV_VAR, self.DEFAULT_MODEL)
        self._temperature = 0.7
        self._think = False

    def send_message(self, prompt: str, model: str = None) -> str:
        target_model = model or self._model

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }

        try:
            logger().info(
                f"[Llama.cpp] (model: {target_model}, temp: {self._temperature}, think: {self._think})"
            )

            response = requests.post(self._url, json=payload, timeout=900)
            response.raise_for_status()

            data = response.json()
            # Handle OpenAI-compatible response format
            if "choices" in data and len(data["choices"]) > 0:
                return (data["choices"][0].get("message", {}).get("content") or "").strip()
            # Fallback for llama.cpp specific /completion endpoint if it was somehow used
            return (data.get("content") or "").strip()

        except requests.exceptions.RequestException as e:
            logger().error(f"Error communicating with Llama.cpp at {self._url}: {e}")
            raise RuntimeError(
                f"Error communicating with Llama.cpp at {self._url}: {e}"
            )

    def is_enabled(self) -> bool:
        return self._url_explicit or bool(os.getenv(self.URL_ENV_VAR))
