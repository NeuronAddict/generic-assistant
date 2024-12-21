from ai.assistant import Assistant
from .generic_ui import GenericUI
from .specific_ui import SpecificUI
from .ui_factory import UIFactory, UI


class GenericUIFactory(UIFactory):

    def get_ui(self, config: dict, assistant: Assistant) -> UI:
        return GenericUI(config, assistant)

class SpecificUIFactory(UIFactory):

    def get_ui(self, config: dict, assistant: Assistant) -> UI:
        return SpecificUI(config, assistant)
