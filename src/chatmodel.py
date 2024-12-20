from abc import abstractmethod

from mistralai import Mistral

from chat_entry import ChatEntry, AssistantChatEntry, History, Role


class ChatModel:

    def __call__(self, *args):
        return self.ask(*args)

    @abstractmethod
    def ask(self, question: str, history: History, temperature: float, prefix: None|str = None) -> AssistantChatEntry:
        pass


class BaseChatModel(ChatModel):

    def __init__(self, client: Mistral, model_name: str):
        self.client = client
        self.model_name = model_name

    def ask(self, question: str, history: History, temperature: float, prefix: None|str = None) -> AssistantChatEntry:

        history.add(ChatEntry(Role.USER, question))

        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages=history.json(prefix),  # pyright: ignore [reportArgumentType]
            temperature=temperature
        )

        if chat_response and chat_response.choices:
            message = chat_response.choices[0].message
            content = message.content

            if isinstance(content, str):
                tool_calls = None
                if message.tool_calls is not None:
                    tool_calls = list(map(
                                        lambda tc: AssistantChatEntry.ToolCall(tc.function.name, tc.function.arguments),  # pyright: ignore [reportArgumentType, reportAttributeAccessIssue]
                                            message.tool_calls)
                                        )

                return AssistantChatEntry(content, tool_calls=tool_calls)
            else:
                raise Exception('stream not supported')
        else:
            raise Exception(f'Bad chat response {chat_response}')
