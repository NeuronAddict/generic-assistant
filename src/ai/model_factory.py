from abc import ABC, abstractmethod

from mistralai import Mistral

from .chatmodel import ChatModel, BaseChatModel


class ModelFactory(ABC):

    @abstractmethod
    def get_chat_model(self, model_name: str) -> ChatModel:
        pass

class MistralFactory(ModelFactory):

    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)

    def get_chat_model(self, model_name: str) -> ChatModel:
        return BaseChatModel(self.client, model_name)