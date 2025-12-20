from utils.logger import Logger
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from llm.model.local_llm import LocalLLM
from llm.model.api_llm import ApiLLM
from llm.llm_const import (
    MODEL_ID_DEEPSEEK_R1_32B,
    MODEL_ID_FACEBOOK_CONTRIEVER,
    MODEL_NAME_FACEBOOK_CONTRIEVER,
    MODEL_ID_CLINICAL_BERT,
    MODEL_NAME_CLINICAL_BERT,
    MODEL_ID_FIN_BERT,
    MODEL_NAME_FIN_BERT
)

import threading

class Auxiliary:

    def __init__(self, config, logger: Logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        self.semantic_model = None
        self.specific_model = None
        self.local_generate_model = None
        self.api_generate_model = None

        self.model_save_path = config["model_save_path"]
        self.model_cache_path = config["model_cache_path"]
        self.generate_model_id = config["generate_model_id"]
        self.specific_model_id = config["specific_model_id"]
        self.api_key = config["api_key"]

        self.semantic_model_lock = threading.Lock()
        self.specific_model_lock = threading.Lock()
        self.local_generate_lock = threading.Lock()
        self.api_generate_lock = threading.Lock()

    def get_semantic_model(self):
        if not self.semantic_model:
            with self.semantic_model_lock:
                if not self.semantic_model:
                    self.semantic_model = self._init_semantic_model(
                        MODEL_ID_FACEBOOK_CONTRIEVER,
                        MODEL_NAME_FACEBOOK_CONTRIEVER,
                        self.model_save_path,
                        self.model_cache_path,
                    )
        return self.semantic_model

    def get_specific_model(self):
        if not self.specific_model:
            with self.specific_model_lock:
                if not self.specific_model:
                    model_id = None
                    model_name = None
                    if self.specific_model_id == MODEL_ID_CLINICAL_BERT:
                        model_id = MODEL_ID_CLINICAL_BERT
                        model_name = MODEL_NAME_CLINICAL_BERT
                    elif self.specific_model_id == MODEL_ID_FIN_BERT:
                        model_id = MODEL_ID_FIN_BERT
                        model_name = MODEL_NAME_FIN_BERT
                    elif self.specific_model_id == MODEL_ID_FACEBOOK_CONTRIEVER:
                        model_id = MODEL_ID_FACEBOOK_CONTRIEVER
                        model_name = MODEL_NAME_FACEBOOK_CONTRIEVER
                    
                    self.specific_model = self._init_semantic_model(
                        model_id,
                        model_name,
                        self.model_save_path,
                        self.model_cache_path,
                    )
        return self.specific_model
    
    def set_local_genereate_model(self, local_llm):
        if not self.local_generate_model:
            with self.local_generate_lock:
                if not self.local_generate_model:
                    self.local_generate_model = local_llm

    def get_local_genereate_model(self) -> LocalLLM:
        return self.local_generate_model

    def get_api_generate_model(self) -> ApiLLM:
        if not self.api_generate_model:
            with self.api_generate_lock:
                if not self.api_generate_model:
                    self.api_generate_model = self._init_api_generate_model(
                        self.generate_model_id, self.api_key, self.logger
                    )
        return self.api_generate_model

    def _init_semantic_model(
        self, model_id, model_name, model_save_path, model_cache_path
    ):
        model_dir = os.path.join(model_save_path, model_name)
        self.logger.info(f"Loading semantic model from {model_dir}")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            model = AutoModel.from_pretrained(
                model_id,
                cache_dir=model_cache_path,
                torch_dtype="auto",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, cache_dir=model_cache_path
            )

            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        else:
            model = AutoModel.from_pretrained(
                model_dir, torch_dtype="auto", device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

        return (model, tokenizer)

    def _init_local_generate_model(
        self, model_id, model_save_path, model_cache_path, logger
    ):
        generate_model = LocalLLM(model_id, model_save_path, logger, model_cache_path)
        return generate_model

    def _init_api_generate_model(self, model_id, api_key, logger):
        generate_model = ApiLLM(model_id, api_key, logger)
        return generate_model
    
    def specific_encode(self, passages):
        model, tokenizer = self.get_specific_model()
        inputs = tokenizer(
            passages, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]
        embeddings = torch.mean(last_hidden, dim=1)
        return embeddings
    
    def specific_similarity(self, p1, p2):
        emb1 = self.specific_encode(p1)
        emb2 = self.specific_encode(p2)

        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        return similarity

    def semantic_encode(self, passages):
        model, tokenizer = self.get_semantic_model()
        inputs = tokenizer(
            passages, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]
        embeddings = torch.mean(last_hidden, dim=1)
        return embeddings

    def semantic_similarity(self, p1, p2):
        emb1 = self.semantic_encode(p1)
        emb2 = self.semantic_encode(p2)

        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        return similarity

    def perplexity(self, passage):
        return self.get_local_genereate_model().perplexity(passage)
