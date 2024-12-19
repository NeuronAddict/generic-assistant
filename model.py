from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd
from mistralai import Mistral


class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"

class AIMessage:

    def __init__(self, role: Role, content: str, prefix: str | None = None):
        self.role = role
        self.content = content
        if prefix is not None and role != Role.ASSISTANT:
            raise ValueError("Role must be ASSISTANT for use a prefix")
        self.prefix = prefix

class History:

    def __init__(self):
        self.history: List[AIMessage] = []

    def add(self, role: Role, content: str, prefix: str | None = None):
        self.history.append(AIMessage(role, content, prefix))

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




class Model:

    def __call__(self, *args):
        return self.ask(*args)

    @abstractmethod
    def ask(self, question: str, history: List[str], mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
        pass


class BaseModel(Model):

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
            content = chat_response.choices[0].message.content
            if isinstance(content, str):
                return content[len(prefix) if prefix else 0:]
            else:
                raise Exception('stream not supported')
        else:
            raise Exception(f'Bad chat response {chat_response}')


class LogModel(Model):

    def __init__(self, assistant: Model):
        self.assistant = assistant

    def ask(self, question: str, history: List[str], mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
        answer = self.assistant.ask(question, history, mistral_model, system_prompt, temperature, prefix)
        self.__log(question, history, mistral_model, system_prompt, temperature, prefix, answer)
        return answer

    def __log(self, question, history, mistral_model, system_prompt, temperature, prefix, answer):
        print(f"""
        ####
        ####
        Use model {mistral_model} with temperature {temperature} and system prompt :
        {system_prompt}
        ###
        Question: {question}
        History: {history}
        Answer: 
        {prefix} {answer} 
        ###
        """)

class DataFrameLogModel(Model):

    def __init__(self, assistant: Model, log_filename: Path):
        self.assistant = assistant
        self.log_filename = log_filename
        self.frame = pd.DataFrame()

    def ask(self, question: str, history: List[str], mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
        answer = self.assistant.ask(question, history, mistral_model, system_prompt, temperature, prefix)
        self.__log(question, history, mistral_model, system_prompt, temperature, prefix, answer)
        return answer

    def __log(self, question, history, mistral_model, system_prompt, temperature, prefix, answer):
        to_add = pd.DataFrame({'question': [question], 'mistral_model': [mistral_model], 'system_prompt': [system_prompt], 'temperature': [temperature], 'prefix': [prefix], 'answer': [answer]})
        self.frame = pd.concat([self.frame, to_add], ignore_index=True)
        self.frame.to_csv(self.log_filename)