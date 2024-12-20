from abc import ABC, abstractmethod

from chat_entry import History
from model import Model


class AISession(ABC):

    @abstractmethod
    def ask(self, question: str) -> str:
        pass



class BaseAISession(AISession):

    def __init__(self, model: Model, system_prompt: str, temperature: float, prefix: str | None = None):
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.prefix = prefix
        self.history = History(system_prompt)


    def ask(self, question: str) -> str:
        answer = self.model.ask(question, self.history, self.temperature, self.prefix)
        self.history.add(answer)
        return answer.content[len(self.prefix) if self.prefix else 0:]

class AISessionDecorator(AISession, ABC):

    def __init__(self, ai_session: AISession):
        self.ai_session = ai_session


class LogAISession(AISessionDecorator):


    def __init__(self, ai_session: AISession):
        super().__init__(ai_session)

    def ask(self, question: str) -> str:
        answer = self.ai_session.ask(question)
        self.__log(question, answer)
        return answer

    @staticmethod
    def __log(question, answer: str):
        print(f"""
        ####
        Question: {question}
        Answer:
        {answer}
        ###
        """)