from abc import abstractmethod
from typing import List

from mistralai import Mistral


class ChatBody:

    def __init__(self, system_prompt: str, question: str, history: List[str], prefix: str|None):
        self.system_prompt = system_prompt
        self.question = question
        self.history = history
        self.prefix = prefix

    def body(self):
        body = [{ "role": "system", "content": self.system_prompt }]
        body += self.history
        body.append({ "role": "user", "content": self.question })
        if self.prefix:
            body.append({ "role": "assistant", "content": self.prefix, "prefix": True })  # pyright: ignore [reportArgumentType]
        return body


class Assistant:

    def __call__(self, *args):
        return self.ask(*args)

    @abstractmethod
    def ask(self, question: str, history: List[str], mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
        pass


class BaseAssistant(Assistant):

    def __init__(self, client: Mistral):
        self.client = client

    def ask(self, question: str, history: List[str], mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
        chat_body = ChatBody(system_prompt, question, history, prefix)
        chat_response = self.client.chat.complete(
            model=mistral_model,
            messages=chat_body.body(),  # pyright: ignore [reportArgumentType]
            temperature=temperature
        )

        if chat_response and chat_response.choices:
            return chat_response.choices[0].message.content[len(prefix) if prefix else 0:]
        else:
            raise Exception(f'Bad chat response {chat_response}')


