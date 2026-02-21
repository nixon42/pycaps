from abc import ABC, abstractmethod


class Llm(ABC):
    @abstractmethod
    def send_message(self, prompt: str, model: str) -> str:
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        pass
