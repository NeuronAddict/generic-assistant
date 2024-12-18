from typing import List

from assistant import Assistant


class AISession:

    def __init__(self, assistant: Assistant, system_prompt: str, prefix: str|None = None):
        self.assistant = assistant
        self.system_prompt = system_prompt
        self.prefix = prefix
        self.history: List[str] = []

    def ask(self, question: str):
        self.assistant.ask(question, self.history, 'mistral-7b-instruct', self.system_prompt, 0.7, self.prefix)