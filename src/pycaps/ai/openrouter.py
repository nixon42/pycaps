import time
from pycaps.logger import logger
from pycaps.ai.llm import Llm
import os


import random
from collections import defaultdict
from typing import List


class OpenRouterLlm(Llm):
    PRIMARY_API_KEY_NAME = "PYCAPS_OPENROUTER_API_KEY"

    def __init__(self):
        self._client = None
        self._current_key = os.getenv(self.PRIMARY_API_KEY_NAME)

        # Load multiple keys from environment (similar to root project)
        self._extra_keys = [
            os.getenv(f"OPENROUTER_API_KEY_{i}", "").strip()
            for i in range(1, 11)
            if os.getenv(f"OPENROUTER_API_KEY_{i}")
        ]
        self._extra_keys = [k for k in self._extra_keys if k]

        # Key scoring for rotation
        self._key_score = defaultdict(lambda: 1.0)

    def _get_all_available_keys(self) -> List[str]:
        keys = []
        if self._current_key:
            keys.append(self._current_key)
        for k in self._extra_keys:
            if k not in keys:
                keys.append(k)
        return keys

    def _mark_key_failed(self, key: str):
        self._key_score[key] *= 0.5

    def _mark_key_success(self, key: str):
        self._key_score[key] = min(self._key_score[key] + 0.1, 1.0)

    def _pick_best_key(self, keys: List[str]) -> str:
        if not keys:
            return None
        weights = [self._key_score[k] for k in keys]
        return random.choices(keys, weights=weights, k=1)[0]

    def send_message(self, prompt: str, model: str = "openrouter/free") -> str:
        available_keys = self._get_all_available_keys()

        if not available_keys:
            logger().error("No OpenRouter API keys available.")
            return ""

        # Try up to N times if we have multiple keys
        max_attempts = min(len(available_keys), 5)
        tried_keys = set()

        for attempt in range(max_attempts):
            # Pick a key, preferring the primary one on first attempt if it exists
            if attempt == 0 and self._current_key:
                key = self._current_key
            else:
                remaining_keys = [k for k in available_keys if k not in tried_keys]
                if not remaining_keys:
                    break
                key = self._pick_best_key(remaining_keys)

            tried_keys.add(key)
            client = self._get_client(key)

            try:
                response = client.chat.send(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )

                content = response.choices[0].message.content
                if content:
                    self._mark_key_success(key)
                    # If this wasn't the primary key, we might want to "stick" to it if it works well?
                    # For now just use it.
                    return content
                else:
                    raise ValueError("Empty response content")

            except Exception as e:
                self._mark_key_failed(key)
                logger().error(
                    f"OpenRouter key failed (attempt {attempt + 1}/{max_attempts}): {e}. Waiting 60s..."
                )
                time.sleep(60)
                # Continue to next attempt/key

        return ""

    def is_enabled(self) -> bool:
        return len(self._get_all_available_keys()) > 0

    def _get_client(self, api_key: str):
        try:
            from openrouter import OpenRouter

            return OpenRouter(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenRouter library not found. "
                "Please install it with: pip install openrouter-python"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenRouter client with key: {e}")
