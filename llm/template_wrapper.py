from .message import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    TYPE_SETTING,
    TYPE_REASONING,
    TYPE_CONTEXT,
)
from .template_manager import TemplateManager

from transformers import AutoTokenizer

from typing import List

class TemplateWrapper:

    def __init__(self, model_id):
        self.template_manager = TemplateManager(model_id)

    def get_config(self):
        return self.template_manager.get_template_config()

    def apply_chat_template(self, messages: List[Message], tokenizer: AutoTokenizer):
        use_default = self.template_manager.use_huggingface()
        if use_default:
            chat = self._messages_to_chat(messages)
            prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt = self.template_manager.build_template(messages)
        return prompt

    def _messages_to_chat(self, messages: List[Message]):
        chat = []
        for message in messages:
            if message.role == ROLE_SYSTEM:
                contents = message.contents
                system_content = ""
                for content in contents:
                    if content.type == TYPE_SETTING:
                        system_content += f"{content.content}\n"
                    elif content.type == TYPE_CONTEXT:
                        system_content += f"Context: {content.content}\n"
                    elif content.type == TYPE_REASONING:
                        system_content += f"Reasoning: {content.content}\n"
                chat.append(
                    {
                        "role": ROLE_SYSTEM,
                        "content": system_content,
                    }
                )
            else:
                content = message.contents[0]  # only one content
                chat.append(
                    {
                        "role": message.role,
                        "content": content.content,
                    }
                )
        return chat
