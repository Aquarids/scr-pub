from abc import ABC, abstractmethod

from ..template_wrapper import TemplateWrapper
from ..message import (
    Message,
    MessageContent,
    ROLE_SYSTEM,
    ROLE_USER,
    TYPE_SETTING,
    TYPE_CONTENT,
)
from utils.logger import Logger
from typing import List

class BaseLLM(ABC):
    def __init__(self, model_id: str, logger: Logger):
        self.model_id = model_id
        self.logger = logger

        self.template_wrapper = TemplateWrapper(model_id)

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        pass

    @abstractmethod
    def batch_generate(self, messages_list: List[List[Message]], batch_size=16):
        pass

    @abstractmethod
    def talk(self, instruction: str, default_system=False) -> str:
        pass

    @abstractmethod
    def batch_talk(self, instructions, batch_size=16, default_system=False):
        pass

    def _common_message_template(
        self, instruction: str, default_system=False
    ) -> List[Message]:
        messages = []
        if default_system:
            messages.append(
                Message(
                    ROLE_SYSTEM,
                    [MessageContent(TYPE_SETTING, "You are an AI assistant.")],
                )
            )

        messages.append(Message(ROLE_USER, [MessageContent(TYPE_CONTENT, instruction)]))
        return messages