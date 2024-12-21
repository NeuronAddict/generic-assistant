from .ai_session import AISession, BaseAISession, LogAISession
from .chatmodel import ChatModel
from .model_factory import ModelFactory


class Assistant:

    def __init__(self, model_factory: ModelFactory):
        self.model_factory = model_factory

    def new_session(self, model_name: str, system_prompt: str, temperature: float, prefix: str | None = None) -> AISession:
        model: ChatModel = self.model_factory.get_chat_model(model_name)
        ai_session = LogAISession(BaseAISession(model, system_prompt, temperature, prefix))
        return ai_session
