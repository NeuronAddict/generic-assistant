from abc import abstractmethod

from mistralai import Mistral

from chat_entry import ChatEntry, AssistantChatEntry, History, Role


class Model:

    def __call__(self, *args):
        return self.ask(*args)

    @abstractmethod
    def ask(self, question: str, history: History, temperature: float, prefix: None|str = None) -> AssistantChatEntry:
        pass


class BaseModel(Model):

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


# class LogModel(Model):
#
#     def __init__(self, assistant: Model):
#         self.assistant = assistant
#
#     def ask(self, question: str, history: History, mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
#         answer = self.assistant.ask(question, history, mistral_model, system_prompt, temperature, prefix)
#         self.__log(question, history, mistral_model, system_prompt, temperature, prefix, answer)
#         return answer
#
#     def __log(self, question, history, mistral_model, system_prompt, temperature, prefix, answer):
#         print(f"""
#         ####
#         ####
#         Use model {mistral_model} with temperature {temperature} and system prompt :
#         {system_prompt}
#         ###
#         Question: {question}
#         History: {history}
#         Answer:
#         {prefix} {answer}
#         ###
#         """)
#
# class DataFrameLogModel(Model):
#
#     def __init__(self, assistant: Model, log_filename: Path):
#         self.assistant = assistant
#         self.log_filename = log_filename
#         self.frame = pd.DataFrame()
#
#     def ask(self, question: str, history: List[str], mistral_model: str, system_prompt: str, temperature: float, prefix: None|str = None) -> str:
#         answer = self.assistant.ask(question, history, mistral_model, system_prompt, temperature, prefix)
#         self.__log(question, history, mistral_model, system_prompt, temperature, prefix, answer)
#         return answer
#
#     def __log(self, question, history, mistral_model, system_prompt, temperature, prefix, answer):
#         to_add = pd.DataFrame({'question': [question], 'mistral_model': [mistral_model], 'system_prompt': [system_prompt], 'temperature': [temperature], 'prefix': [prefix], 'answer': [answer]})
#         self.frame = pd.concat([self.frame, to_add], ignore_index=True)
#         self.frame.to_csv(self.log_filename)