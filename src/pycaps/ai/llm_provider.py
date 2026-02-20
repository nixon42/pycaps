from typing import Optional
from .llm import Llm
from .gpt import Gpt
from .openrouter import OpenRouterLlm
from .ollama import Ollama


class LlmProvider:
    _llm: Optional[Llm] = None

    @staticmethod
    def configure(
        provider: str = None,
        openrouter_key: str = None,
        ollama_url: str = None,
        ollama_model: str = None,
    ):
        if provider == "openrouter":
            LlmProvider._llm = OpenRouterLlm(api_key=openrouter_key)
        elif provider == "ollama":
            LlmProvider._llm = Ollama(base_url=ollama_url, model=ollama_model)
        elif provider == "gpt":
            LlmProvider._llm = Gpt()
        else:
            # Auto-detection logic if no provider specified
            openrouter = OpenRouterLlm(api_key=openrouter_key)
            ollama = Ollama(base_url=ollama_url, model=ollama_model)
            gpt = Gpt()

            if openrouter.is_enabled():
                LlmProvider._llm = openrouter
            elif ollama.is_enabled():
                LlmProvider._llm = ollama
            else:
                LlmProvider._llm = gpt

    @staticmethod
    def get() -> Llm:
        if LlmProvider._llm is None:
            LlmProvider.configure()
        return LlmProvider._llm

    @staticmethod
    def set(llm: Llm):
        LlmProvider._llm = llm
