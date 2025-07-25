from abc import ABC, abstractmethod

from ai.assistant import Assistant


class UI(ABC):

    @abstractmethod
    def launch(self, ip: str, port: int):
        pass


class UIFactory(ABC):

    @abstractmethod
    def get_ui(self, config: dict, assistant: Assistant) -> UI:
        pass
