from .base_llm import BaseLLM
from ..message import Message
from ..llm_const import (
    END_POINT_MAP,
)

from utils.logger import Logger
from typing import List
from openai import OpenAI
import time
import math

class ApiLLM(BaseLLM):

    def __init__(self, model_id: str, api_key: str, logger: Logger = None):
        super().__init__(model_id, logger)
        self.api_key = api_key

        self.rpm_limit = math.inf
        self.tpm_limit = math.inf
        # self.rpm_limit = 1000
        # self.tpm_limit = 10000
        self.request_count = 0
        self.token_count = 0
        self.last_reset = time.time()

        self._init_api()

    def _init_api(self):
        base_url = None
        for endpoint, model_ids in END_POINT_MAP.items():
            if self.model_id in model_ids:
                base_url = endpoint

        if not base_url:
            raise ValueError(f"Empty api end point for {self.model_id}")

        self.api = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(self, messages: List[Message]) -> str:
        now = time.time()

        if now - self.last_reset >= 60:
            self.request_count = 0
            self.token_count = 0
            self.last_reset = now

        if self.request_count >= self.rpm_limit:
            wait_time = max(0, 60 - (now - self.last_reset))
            self.logger.info(f"Sleep in {wait_time} seconds")
            time.sleep(wait_time + 2)
            self.request_count = 0
            self.token_count = 0
            self.last_reset = time.time()


        input_tokens = self._estimate_tokens(messages)
        safe_margin = 1.2
        estimated_tokens = int(input_tokens * safe_margin)
        
        if self.token_count + estimated_tokens > self.tpm_limit:
            wait_time = max(0, 60 - (now - self.last_reset))
            self.logger.info(f"Sleep in {wait_time} seconds")
            time.sleep(wait_time + 2)
            self.request_count = 0
            self.token_count = 0
            self.last_reset = time.time()

        chat = self.template_wrapper._messages_to_chat(messages)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                chat = self.template_wrapper._messages_to_chat(messages)
                response = self.api.chat.completions.create(
                    model=self.model_id, messages=chat, temperature=1.0, stream=False
                )

                self.request_count += 1
                actual_tokens = response.usage.total_tokens
                self.token_count += actual_tokens

                self.logger.info(f"Request success with {actual_tokens} tokens")
                return response.choices[0].message.content

            except Exception as e:
                if "429" in str(e) or "rate limiting" in str(e):
                    sleep_time = 2**attempt + 1
                    self.logger.warning(
                        f"Rate limit hit! Retry #{attempt+1} in {sleep_time}s"
                    )
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Fatal error: {e}")
                    return "Generation failed."

    def batch_generate(self, messages_list: List[List[Message]], batch_size=16):
        results = []
        for message in messages_list:
            results.append(self.generate(message))
        return results

    def talk(self, instruction: str, default_system=False) -> str:
        return self.generate(self._common_message_template(instruction, default_system))

    def batch_talk(self, instructions, batch_size=16, default_system=False):
        results = []
        for instruction in instructions:
            results.append(self.talk(instruction, default_system))
        return results

    def _estimate_tokens(self, messages: List[Message]) -> int:
        return 5000