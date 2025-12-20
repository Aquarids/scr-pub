import dataclasses

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

TYPE_SETTING = "setting"
TYPE_CONTEXT = "context"
TYPE_REASONING = "reasoning"
TYPE_CONTENT = "content"

from typing import List

@dataclasses.dataclass
class MessageContent:
    type: str
    content: str
    metadata: dict = None

@dataclasses.dataclass
class Message:
    role: str
    contents: List[MessageContent]