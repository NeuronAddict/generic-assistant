from enum import Enum
from typing import List, Dict, Any


class Role(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"

    def __repr__(self):
        return self.value

class ChatEntry:

    def __init__(self, role: Role, content: str, name: str | None = None ):
        self.role = role
        self.content = content
        self.name = name

class AssistantChatEntry(ChatEntry):

    class ToolCall:

        def __init__(self, function_name: str, args: Dict[str, Any]):
            self.function_name = function_name
            self.args = args

    def __init__(self, content: str, prefix: bool = False, name: str | None = None, refusal: bool = False, tool_calls: List[ToolCall] | None = None):
        super().__init__(Role.ASSISTANT, content, name)
        self.prefix = prefix
        self.refusal = refusal
        self.tool_calls = tool_calls if tool_calls is not None else []


class History:

    def __init__(self, system_prompt: str):
        self.history: List[ChatEntry] = [ChatEntry(Role.SYSTEM, system_prompt)]

    def add(self, chat_entry: ChatEntry):
        self.history.append(chat_entry)

    def json(self):
        return list(map(lambda entry: entry.__dict__, self.history))