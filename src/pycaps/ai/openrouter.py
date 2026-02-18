from pycaps.ai.llm import Llm
import os


class OpenRouterLlm(Llm):
    API_KEY_NAME = "PYCAPS_OPENROUTER_API_KEY"

    def __init__(self):
        self._client = None

    def send_message(self, prompt: str, model: str = "openrouter/free") -> str:
        client = self._get_client()
        # OpenRouter library expects a specific format for messages
        # Here we follow the pattern seen in the user's llm_api/llm.py
        response = client.chat.send(
            model=model, messages=[{"role": "user", "content": prompt}], stream=False
        )

        # The return type depends on the library version,
        # but following Gpt pattern where it returns output_text or similar.
        # Based on the user's example, response.model_dump() is available.
        # However, usually we want the message content.
        # Looking at Gpt implementation: return self._get_client().responses.create(model=model, input=prompt).output_text
        # In the user example res = response.model_dump()
        # Let's try to access the content directly or via model_dump.
        try:
            return response.choices[0].message.content
        except AttributeError:
            # Fallback if the library structure is slightly different
            return str(response)

    def is_enabled(self) -> bool:
        return os.getenv(self.API_KEY_NAME) is not None

    def _get_client(self):
        try:
            from openrouter import OpenRouter

            if self._client:
                return self._client

            self._client = OpenRouter(api_key=os.getenv(self.API_KEY_NAME))
            return self._client
        except ImportError:
            raise ImportError(
                "OpenRouter library not found. "
                "Please install it with: pip install openrouter-python"  # Assuming this is the name or similar
            )
        except Exception as e:
            raise RuntimeError(
                f"Error initializing OpenRouter client: {e}\n\n"
                f"Please ensure you have authenticated correctly via {self.API_KEY_NAME}."
            )
