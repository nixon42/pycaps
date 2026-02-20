from typing import Optional
from .llm import Llm
from .gpt import Gpt
from .openrouter import OpenRouterLlm
from .ollama import Ollama


class LlmProvider:
    _llm: Optional[Llm] = None

    @staticmethod
    def get() -> Llm:
        if LlmProvider._llm is None:
            openrouter = OpenRouterLlm()
            gpt = Gpt()
            ollama = Ollama()

            if openrouter.is_enabled():
                LlmProvider._llm = openrouter
            elif ollama.is_enabled():
                LlmProvider._llm = ollama
            else:
                LlmProvider._llm = gpt

        return LlmProvider._llm

    @staticmethod
    def set(llm: Llm):
        LlmProvider._llm = llm
