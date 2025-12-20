from utils.logger import Logger
from agent.auxiliary import Auxiliary

import torch
import os
import time

class BasicQueryClient:

    def __init__(
        self,
        logger: Logger,
        name,
        tag,
        db_path,
        auxiliary: Auxiliary,
        extra=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logger
        self.name = name
        self.tag = tag
        self.extra = extra
        self.db_path = db_path

        self.auxiliary = auxiliary
        self.retriever = self._init_retriever()

    def _init_retriever(self):
        # implement your retriever
        return None
    
    def db_folder(self) -> str:
        return "normal"

    def retrieve(self, query: str, top_k: int = 10):
        self.logger.info(f"Client {self.tag} retrieving documents for query: {query}")

        retrieved_docs = self.retriever.retrieve_docs(query, top_k=top_k)
        
        if not retrieved_docs:
            self.logger.warning(
                f"Client {self.tag} no documents found for query: {query}"
            )
            return None

        doc_infos = []
        for doc in retrieved_docs:
            doc_infos.append({
                "doc_id": doc.doc_id,
                "content": doc.content,
                "category": doc.metadata.get("category", "Unknown"),
            })


        self.logger.info(f"Client {self.tag} retrieve docs:\n{doc_infos}")

        return doc_infos

    def is_malicious(self):
        return False
