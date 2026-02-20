from pycaps.logger import logger
from pycaps.ai.llm import Llm
import os


class OpenRouterLlm(Llm):
    PRIMARY_API_KEY_NAME = "PYCAPS_OPENROUTER_API_KEY"

    def __init__(self, api_key: str = None):
        self._current_key = api_key or os.getenv(self.PRIMARY_API_KEY_NAME)

    def send_message(self, prompt: str, model: str = "openrouter/free") -> str:
        if not self._current_key:
            logger().error(
                f"No API key found in environment variable {self.PRIMARY_API_KEY_NAME}"
            )
            return ""

        client = self._get_client()

        try:
            response = client.chat.send(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )

            content = response.choices[0].message.content
            if content:
                return content
            else:
                raise ValueError("Empty response content")

        except Exception as e:
            logger().error(f"OpenRouter request failed: {e}")
            return ""

    def is_enabled(self) -> bool:
        return bool(self._current_key)

    def _get_client(self):
        try:
            from openrouter import OpenRouter

            return OpenRouter(api_key=self._current_key)
        except ImportError:
            raise ImportError(
                "OpenRouter library not found. "
                "Please install it with: pip install openrouter-python"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenRouter client: {e}")
