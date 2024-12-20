from abc import ABC, abstractmethod

from chat_entry import History
from chatmodel import ChatModel


class AISession(ABC):

    @abstractmethod
    def ask(self, question: str) -> str:
        pass


class BaseAISession(AISession):
    """
    A class for managing and configuring AI chat interactions with a customizable
    system prompt, response creativity, message prefixing, and interaction history.

    This class is designed to facilitate interactive chat sessions by encapsulating
    the necessary configurations, maintaining a history of conversations, and
    delegating the actual response generation to a chat model. It allows fine-grained
    control over the behavior of the model and ensures the continuity of the
    conversation context.

    :ivar model: The chat model instance responsible for generating responses.
    :type model: ChatModel
    :ivar system_prompt: The initial system message that defines the conversation
        context and behavior.
    :type system_prompt: str
    :ivar temperature: A floating-point value used to adjust the randomness and
        diversity of the responses.
    :type temperature: float
    :ivar prefix: A string used for message prefixing during communication with the
        model, or None if no prefix is used.
    :type prefix: str | None
    :ivar history: An object that tracks the ongoing conversation's history for
        context maintenance across multiple exchanges.
    :type history: History
    """
    def __init__(self, model: ChatModel, system_prompt: str, temperature: float, prefix: str | None = None):
        """
        Initializes an instance of a configuration object used for managing chat
        interactions. The class serves as a container to store and manage the model,
        its system prompt, chatbot response temperature, and an optional prefix for
        messages. These attributes, along with history management, enable the control
        of behavior and context during interactive chat operations.

        :param model: The chat model instance that will handle the message exchanges.
        :type model: ChatModel
        :param system_prompt: The system-wide initialization prompt that sets the base
            context for conversation.
        :type system_prompt: str
        :param temperature: A float value to determine response randomness and
            creativity in model predictions.
        :type temperature: float
        :param prefix: Optional string used to prepend to the messages before passing
            to the model.
        :type prefix: str | None
        """
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.prefix = prefix
        self.history = History(system_prompt)


    def ask(self, question: str) -> str:
        """
        Processes a user's question and generates a model-based answer while maintaining a history
        of interactions. It supports customization of the output through prefixes and temperature
        settings.

        :param question: The input question provided by the user to generate an answer.
        :type question: str
        :return: The processed answer string from the model after applying the prefix.
        :rtype: str
        """
        answer = self.model.ask(question, self.history, self.temperature, self.prefix)
        self.history.add(answer)
        return answer.content[len(self.prefix) if self.prefix else 0:]



class AISessionDecorator(AISession, ABC):
    """
    A decorator class for the AISession interface.

    This class provides a way to extend or modify the behavior of an AISession
    instance by wrapping it with additional functionality. It implements the
    AISession interface and acts as a base class for creating such decorators.

    :ivar ai_session: The underlying AISession instance that this decorator wraps.
    :type ai_session: AISession
    """
    def __init__(self, ai_session: AISession):
        self.ai_session = ai_session


class LogAISession(AISessionDecorator):
    """
    Decorator class for logging AI Session interactions.

    This class is used to wrap an existing AISession instance and extend its behavior
    by logging all questions asked and their respective answers. It provides additional
    logging functionality while maintaining the core AI session behavior.

    :ivar ai_session: The decorated AISession instance.
    :type ai_session: AISession
    """
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