from enum import Enum
from typing import List

from model import Model



class ChatEntry:

    def __init__(self, role: Role, ):

class AISession:

    def __init__(self, model: Model, system_prompt: str, temperature: float, prefix: str | None = None):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.prefix = prefix
        self.history: List[str] = []

    def ask(self, question: str):
        self.history.append(question)
        answer = self.model.ask(question, self.history, 'mistral-7b-instruct', self.system_prompt, self.temperature, self.prefix)
