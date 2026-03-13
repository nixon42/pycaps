from typing import Optional
from .llm import Llm
from .gpt import Gpt
from .openrouter import OpenRouterLlm
from .ollama import Ollama
from .llama_cpp import LlamaCpp


class LlmProvider:
    _llm: Optional[Llm] = None

    @staticmethod
    def configure(
        provider: str = None,
        openrouter_key: str = None,
        ollama_url: str = None,
        ollama_model: str = None,
        llama_cpp_url: str = None,
        llama_cpp_model: str = None,
        llama_cpp_temp: float = 0.5,
        llama_cpp_think: bool = False,
    ):
        if provider == "OpenRouter":
            LlmProvider._llm = OpenRouterLlm(api_key=openrouter_key)
        elif provider == "Ollama":
            LlmProvider._llm = Ollama(base_url=ollama_url, model=ollama_model)
        elif provider == "LlamaCPP":
            LlmProvider._llm = LlamaCpp(
                url=llama_cpp_url,
                model=llama_cpp_model,
                temperature=llama_cpp_temp,
                think=llama_cpp_think,
            )
        elif provider == "gpt":
            LlmProvider._llm = Gpt()
        else:
            # Auto-detection logic if no provider specified
            openrouter = OpenRouterLlm(api_key=openrouter_key)
            ollama = Ollama(base_url=ollama_url, model=ollama_model)
            llama_cpp = LlamaCpp(
                url=llama_cpp_url,
                model=llama_cpp_model,
                temperature=llama_cpp_temp,
                think=llama_cpp_think,
            )
            gpt = Gpt()

            if openrouter.is_enabled():
                LlmProvider._llm = openrouter
            elif ollama.is_enabled():
                LlmProvider._llm = ollama
            elif llama_cpp.is_enabled():
                LlmProvider._llm = llama_cpp
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
