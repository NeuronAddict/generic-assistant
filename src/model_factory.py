import os
from abc import ABC, abstractmethod

from mistralai import Mistral

from chatmodel import ChatModel, BaseChatModel


class ModelFactory(ABC):

    @abstractmethod
    def get_chat_model(self, model_name: str) -> ChatModel:
        pass

class MistralFactory(ModelFactory):

    def __init__(self):
        self.client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])

    def get_chat_model(self, model_name: str) -> ChatModel:
        return BaseChatModel(self.client, model_name)