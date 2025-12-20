from .base_llm import BaseLLM
from ..message import Message
from utils.logger import Logger

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import os
import torch
from typing import List


class LocalLLM(BaseLLM):

    def __init__(
        self, model_id: str, model_save_path: str, logger: Logger, model_cache_path=None, token=None
    ):
        super().__init__(model_id, logger)
        self.token = token
        self.save_dir = model_save_path
        self.cache_dir = model_cache_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.llm = None
        self.template = None
        self.init()

    def init(self):
        self.logger.info("Loading local llm model")
        model, tokenizer = self._load_model(self.model_id, self.token)

        text_generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024 * 8,
            truncation=True,
            return_full_text=False,
            temperature=1.0,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
        )
        self.llm = text_generation_pipe

    def generate(self, messages: List[Message]):
        prompt = self.template_wrapper.apply_chat_template(messages, self.tokenizer)
        self.logger.info(f"Prompt: {prompt}")
        with torch.inference_mode():
            output = self.llm(prompt)
            response = output[0]["generated_text"]

        return response
    
    def batch_generate(self, messages_list: List[List[Message]], batch_size=16):
        prompts = []
        for messages in messages_list:
            prompt = self.template_wrapper.apply_chat_template(messages, self.tokenizer)
            prompts.append(prompt)

        with torch.inference_mode():
            outputs = self.llm(prompts, batch_size=batch_size)

        return [output[0]["generated_text"].strip() for output in outputs]

    def talk(self, instruction, default_system=False):
        messages = self._common_message_template(instruction, default_system)
        prompt = self.template_wrapper.apply_chat_template(messages, self.tokenizer)
        with torch.inference_mode():
            output = self.llm(prompt)
            response = output[0]["generated_text"]

        return response
    
    def batch_talk(self, instructions, batch_size=16, default_system=False):
        all_messages = [
            self._common_message_template(instr, default_system)
            for instr in instructions
        ]
        
        prompts = [
            self.template_wrapper.apply_chat_template(messages, self.tokenizer)
            for messages in all_messages
        ]
        
        with torch.inference_mode():
            outputs = self.llm(prompts, batch_size=batch_size)
            
        return [output[0]["generated_text"].strip() for output in outputs]

    def _load_model(self, model_id, token=None):
        template_config = self.template_wrapper.get_config()
        self.model_name = template_config.name
        model_path = self._model_path()
        need_save = False

        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
                use_cache=True,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, device_map="auto", trust_remote_code=True, use_fast=False
            )
        else:
            model, tokenizer = self._download_model(model_id, token)
            need_save = True

        special_tokens = {}
        if template_config.bos_token is not None:
            special_tokens["bos_token"] = template_config.bos_token
        if template_config.eos_token is not None:
            special_tokens["eos_token"] = template_config.eos_token
        if template_config.pad_token is not None:
            special_tokens["pad_token"] = template_config.pad_token
        tokenizer.add_special_tokens(special_tokens)

        if need_save:
            self._save_model(model, tokenizer, model_path)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def _quant_config(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        return quant_config

    def _download_model(self, model_id, token=None):
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "use_cache": True,
            "cache_dir": self.cache_dir,
        }
        
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": False,
            "cache_dir": self.cache_dir,
        }
        
        if token:
            model_kwargs["token"] = token
            tokenizer_kwargs["token"] = token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )
        
        return model, tokenizer

    def _save_model(self, model, tokenizer, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    def _model_path(self):
        return f"{self.save_dir}/{self.model_name}"

    def tokenize(self, text, padding=True, truncation=True):
        return self.tokenizer(
            text, padding=padding, truncation=truncation, return_tensors="pt"
        ).to(self.device)

    def forward(self, inputs_ids, labels_ids):
        with torch.no_grad():
            outputs = self.model(inputs_ids, labels=labels_ids)
        return outputs

    def compute_gradients(self, inputs_ids, labels_ids):
        with torch.enable_grad():
            outputs = self.model(inputs_ids, labels=labels_ids)
            loss = outputs.loss
            loss.backward()
            grad = self.model.get_input_embeddings().weight.grad
        return grad

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_input_embeddings_weight(self):
        return self.model.get_input_embeddings().weight.detach()

    def get_sentence_embedding(self, sentence):
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]
        embeddings = torch.mean(last_hidden, dim=1)
        return embeddings.squeeze()

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def batch_decode(self, input_ids):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def embedding(self, input_ids):
        with torch.no_grad():
            outputs = self.model.get_input_embeddings()(input_ids)
        return outputs

    def embedding_token(self, token):
        input_ids = token.input_ids
        return self.embedding(input_ids)

    def perplexity(self, passage):
        inputs_token = self.tokenizer(passage, return_tensors="pt", truncation=True).to(self.device)
        loss = self.model(**inputs_token, labels=inputs_token["input_ids"]).loss
        ppl = torch.exp(loss)
        return ppl
    
    def calculate_probs(self, targets, query):
        inputs_token = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)
        targets_token = []
        for target_str in targets:
            target_token = self.tokenizer.encode(target_str, add_special_tokens=False)
            targets_token.append(target_token)
        outputs = self.model.generate(
            inputs_token.input_ids,
            output_scores=True,
            return_dict_in_generate=True,
        )
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        probabilities = transition_scores.exp().cpu().numpy()
        generated_tokens = outputs.sequences[0].tolist()

        probs = []
        for target_token in targets_token:
            if all(token in generated_tokens for token in target_token):
                start_index = generated_tokens.index(target_token[0])
                joint_prob = 1.0
                for i, token in enumerate(target_token):
                    joint_prob *= probabilities[start_index + i][0]
                    probs.append(joint_prob)
        return probs
    
    def next_token_probs(self, inputs):
        inputs_token = self.tokenizer(inputs, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs_token)
            logits = outputs.logits[0, -1]
        probs = torch.exp(logits) / torch.sum(torch.exp(logits))
        valid_ids = list(self.tokenizer.get_vocab().values())
        tokens = self.tokenizer.convert_ids_to_tokens(valid_ids)
        return tokens, probs
