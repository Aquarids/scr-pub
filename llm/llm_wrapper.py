from .model.base_llm import BaseLLM
from .model.api_llm import ApiLLM
from .model.local_llm import LocalLLM
from .llm_const import API_MODEL, LOCAL_MODEL
from .message import Message

from utils.logger import Logger
from typing import List

class LLMWrapper:

    def __init__(self, config, logger: Logger):
        self.logger = logger
        self.config = config
        self.llm: BaseLLM = None

    def init(self):
        self.model_id = self.config["model_id"]
        if self.model_id in API_MODEL:
            api_key = self.config["api_key"]
            self.llm = ApiLLM(self.model_id, api_key, self.logger)
        elif self.model_id in LOCAL_MODEL:
            model_save_path = self.config["model_save_path"]
            model_cache_path = self.config.get("model_cache_path", None)
            token = self.config.get("token", None)
            self.llm = LocalLLM(
                self.model_id, model_save_path, self.logger, model_cache_path, token
            )
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}")

    def generate(self, messages: List[Message]):
        response = self.llm.generate(messages)
        return response
    
    def batch_generate(self, messages_list: List[List[Message]], batch_size=16):
        responses = self.llm.batch_generate(messages_list, batch_size)
        return responses

    def talk(self, instruction, default_system=False):
        response = self.llm.talk(instruction, default_system)
        return response
    
    def batch_talk(self, instructions, batch_size=16, default_system=False):
        responses = self.llm.batch_talk(instructions, batch_size, default_system)
        return responses
