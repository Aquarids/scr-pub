from .message import (
    Message,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    TYPE_SETTING,
    TYPE_CONTEXT,
    TYPE_REASONING,
)
from .llm_const import (
    MODEL_ID_DEEPSEEK_7B,
    MODEL_NAME_DEEPSEEK_7B,
    MODEL_ID_DEEPSEEK_R1_32B,
    MODEL_NAME_DEEPSEEK_R1_32B,
    MODEL_ID_QwQ_32B,
    MODEL_NAME_QwQ_32B,
    MODEL_ID_LLAMA3_8B,
    MODEL_NAME_LLAMA3_8B,
)
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer

@dataclass
class ModelTemplateConfig:
    id: str
    name: str
    template_system: str
    template_user: str
    template_assistant: str
    bos_token: str
    eos_token: str
    pad_token: str


class TemplateManager:
    def __init__(self, model_id: str):
        self._template_registry = {
            # DeepSeek-7B
            MODEL_ID_DEEPSEEK_7B: ModelTemplateConfig(
                id=MODEL_ID_DEEPSEEK_7B,
                name=MODEL_NAME_DEEPSEEK_7B,
                template_system="System",
                template_user="User",
                template_assistant="Assistant",
                bos_token="<|begin▁of▁sentence|>",
                eos_token="<|end▁of▁sentence|>",
                pad_token="[PAD]",
            ),
            MODEL_ID_DEEPSEEK_R1_32B: ModelTemplateConfig(
                id=MODEL_ID_DEEPSEEK_R1_32B,
                name=MODEL_NAME_DEEPSEEK_R1_32B,
                template_system="System",
                template_user="User",
                template_assistant="Assistant",
                bos_token="<|begin▁of▁sentence|>",
                eos_token="<|end▁of▁sentence|>",
                pad_token="[PAD]",
            ),
            MODEL_ID_QwQ_32B: ModelTemplateConfig(
                id=MODEL_ID_QwQ_32B,
                name=MODEL_NAME_QwQ_32B,
                template_system="System",
                template_user="User",
                template_assistant="Assistant",
                bos_token=None,
                eos_token=None,
                pad_token=None,
            ),
            MODEL_ID_LLAMA3_8B: ModelTemplateConfig(
                id=MODEL_ID_LLAMA3_8B,
                name=MODEL_NAME_LLAMA3_8B,
                template_system="System", 
                template_user="User",
                template_assistant="Assistant",
                bos_token="<s>",
                eos_token="</s>",
                pad_token="</s>",
            ),
        }
        self.config: ModelTemplateConfig = self._init_config(model_id)

    def use_huggingface(self):
        if self.config.id in [MODEL_ID_DEEPSEEK_R1_32B, MODEL_ID_QwQ_32B, MODEL_ID_LLAMA3_8B]:
            return True
        return False

    def build_template(self, messages: List[Message]):
        config = self.config
        if config.id in [MODEL_ID_DEEPSEEK_7B, MODEL_ID_DEEPSEEK_R1_32B]:
            prompt = config.bos_token
            for message in messages:
                if message.role == ROLE_SYSTEM:
                    prompt += self._build_system_message(message)
                elif message.role == ROLE_USER:
                    prompt += self._build_user_message(message)
                elif message.role == ROLE_ASSISTANT:
                    prompt += self._build_assistant_message(message)
            prompt += f"{config.eos_token}\n{config.template_assistant}:"
        elif config.id == MODEL_ID_LLAMA3_8B:
            prompt = config.bos_token
            for message in messages:
                if message.role == ROLE_SYSTEM:
                    prompt += self._build_system_message(message)
                elif message.role == ROLE_USER:
                    prompt += self._build_user_message(message)
                elif message.role == ROLE_ASSISTANT:
                    prompt += self._build_assistant_message(message)
            prompt += f"{config.template_assistant}:"
        else:
            raise ValueError("Unknown model id")
        return prompt

    def _build_system_message(self, message: Message):
        if self.config.id in [MODEL_ID_DEEPSEEK_7B, MODEL_ID_DEEPSEEK_R1_32B]:
            prompt = f"{self.config.template_system}:\n"
            contents = message.contents
            for content in contents:
                if content.type == TYPE_SETTING:
                    prompt += f"{content.content}\n"
                elif content.type == TYPE_CONTEXT:
                    prompt += f"Context: {content.content}\n"
                elif content.type == TYPE_REASONING:
                    prompt += f"Reasoning: {content.content}\n"
        elif self.config.id == MODEL_ID_LLAMA3_8B:
            prompt = f"{self.config.template_system}:\n"
            contents = message.contents
            for content in contents:
                if content.type == TYPE_SETTING:
                    prompt += f"{content.content}\n"
                elif content.type == TYPE_CONTEXT:
                    prompt += f"Context: {content.content}\n"
                elif content.type == TYPE_REASONING:
                    prompt += f"Reasoning: {content.content}\n"
        else:
            raise ValueError("Unknown model id")
        return prompt

    def _build_user_message(self, message: Message):
        if self.config.id in [MODEL_ID_DEEPSEEK_7B, MODEL_ID_DEEPSEEK_R1_32B]:
            prompt = f"{self.config.template_user}:\n"
            content = message.contents[0]
            prompt += f"{content.content}\n"
        elif self.config.id == MODEL_ID_LLAMA3_8B:
            prompt = f"{self.config.template_user}:\n"
            content = message.contents[0]
            prompt += f"{content.content}\n"
        else:
            raise ValueError("Unknown model id")
        return prompt

    def _build_assistant_message(self, message: Message):
        if self.config.id in [MODEL_ID_DEEPSEEK_7B, MODEL_ID_DEEPSEEK_R1_32B]:
            prompt = f"{self.config.template_assistant}:\n"
            content = message.contents[0]
            prompt += f"{content.content}\n"
        elif self.config.id == MODEL_ID_LLAMA3_8B:
            prompt = f"{self.config.template_assistant}:\n"
            content = message.contents[0]
            prompt += f"{content.content}\n"
        else:
            raise ValueError("Unknown model id")
        return prompt

    def _get_default_token(self, tokenizer: AutoTokenizer) -> str:
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token
        return bos_token, eos_token, pad_token

    def _default_template_config(self, model_id) -> str:
        return ModelTemplateConfig(
            id=model_id,
            name="Default",
            template_system="System",
            template_user="User",
            template_assistant="Assistant",
            bos_token=None,
            eos_token=None,
            pad_token=None,
        )

    def _init_config(self, model_id) -> ModelTemplateConfig:
        for key in self._template_registry:
            if model_id == key:
                return self._template_registry[key]

        return self._default_template_config(model_id)

    def get_template_config(self) -> ModelTemplateConfig:
        return self.config