from .graph_builder import GraphBuilder
from .dynamic_graph_net import DynamicGraphClusterNet, GraphClusterNet
from .basic_query_client import BasicQueryClient
from agent.auxiliary import Auxiliary
from utils.logger import Logger
from torch_geometric.data import Data

import os
import torch
import json
from typing import Dict, List
from collections import defaultdict
import numpy as np
import datetime
import threading

class Assessor:

    def __init__(self, logger: Logger, auxiliary: Auxiliary, clients, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auxiliary = auxiliary
        self.config = config
        self.cache_dir = config.get("cache_dir")
        self.output_dir = config.get("output_dir")
        self.assets_path = config.get("assets_path")
        self.attack_mode = config.get("attack_mode")
        self.mode = config.get("assess_mode", "no_check")
        self.dataset_type = config.get("dataset_type")

        # only for source credit
        # todo: should not hold
        self.clients = clients
        self.config = config
        self.logger = logger

        self.graph_builder = GraphBuilder(
            self.logger, self.auxiliary, self.cache_dir, self.assets_path, self.dataset_type
        )
        self.base_model, self.postaudit_model = self._init_model(config.get("cache_model_path"))
        
        self.reset()

    def reset(self):

        base_name = "client_credits"
        self.credits_filename = f"{self.attack_mode}_{base_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        self.client_credits = self._init_client_credits(self.clients, default_credit=0.5)
        self.client_history_credits = self._init_client_credits(self.clients, default_credit=1.0)

    def _init_model(self, model_path):

        if self.mode == "no_sec":
            return None, None

        base_model = DynamicGraphClusterNet(static_dim=12).to(self.device)
        postaudit_model = DynamicGraphClusterNet(static_dim=13).to(self.device)

        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model from {model_path}")

            base_model_filepath = os.path.join(model_path, "base_model.pt")
            base_model.load_state_dict(torch.load(base_model_filepath, map_location=self.device))

            postaudit_model_filepath = os.path.join(model_path, "postaudit_model.pt")
            postaudit_model.load_state_dict(torch.load(postaudit_model_filepath, map_location=self.device))
        else:
            self.logger.info("Initializing new model")

        base_model.to(self.device)
        postaudit_model.to(self.device)
        return base_model, postaudit_model

    def _save_client_credits(self, client_credits, cache_dir, name=None):
        if not name:
            name = "client_credits.json"
        dir = os.path.join(cache_dir, self.mode)
        filepath = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(client_credits))

    def _init_client_credits(self, clients: List[BasicQueryClient], default_credit = 0.5):

        client_credits = {}
        for client in clients:
            client_credits[client.tag] = default_credit

        return client_credits

    def save_client_credits(self):
        self._save_client_credits(self.client_credits, self.cache_dir, name=self.credits_filename)

    def _calculate_score(self, clients_data, model, fact_check):
        node_features, edge_index, doc_features = self.graph_builder.build_graph(
            clients_data, self.client_credits, self.client_history_credits, fact_check=fact_check
        )
        raw_data = {"node_features": node_features, "edge_index": edge_index}
        data = self.create_graph_data(raw_data, fact_check=fact_check)
        model.eval()
        with torch.no_grad():
            output = model(data)
            prob = torch.sigmoid(output).squeeze()
            self.logger.warning(prob)
        return prob, doc_features

    def assess(self, clients_data):
        self.logger.info(f"Assessor start to assess in mode [{self.mode}]")

        credit_decay = self.config.get("credit_decay", 0.5)
        doc_scores, doc_features = self._calculate_score(clients_data, self.base_model, False)

        uncertain_mask = (doc_scores >= 0.45) & (doc_scores <= 0.55)
        if uncertain_mask.any():
            doc_scores, doc_features = self._calculate_score(clients_data, self.postaudit_model, True)

        doc_results = {}
        for idx, (doc_id, feature) in enumerate(doc_features.items()):
            client_id = feature["client_id"]
            doc_trust_score = doc_scores[idx].cpu().numpy()

            doc_results[doc_id] = {
                "trust_score": float(doc_trust_score),
                "client_id": client_id,
                "doc_id": doc_id,
                "content": feature["content"],
                "category": feature["category"],
            }

        self.client_history_credits = self.client_credits
        self.client_credits = self._update_client_credits(
            self.client_credits, doc_results, credit_decay
        )
        self._save_client_credits(self.client_credits, self.cache_dir, name=self.credits_filename)

        return doc_results

    def create_graph_data(self, raw_dict, fact_check=False):
        node_features = self._process_node_features(raw_dict["node_features"], fact_check)
        edge_index = self._process_edge_index(raw_dict["edge_index"])

        return Data(
            x=node_features,
            edge_index=edge_index,
        ).to(self.device)

    def _process_node_features(self, raw_features, fact_check=False):
        feature_tensors = []
        for node in raw_features:

            node_feature = [
                node["content_ratio"],
                node["kw_tfidf_ratio"],
                node["kw_count_ratio"],
                node["category_rarity"],
                node["semantic_cohesion"],
                node["tfidf_semantic"],
                node["neg_sentiment"],
                node["neu_sentiment"],
                node["pos_sentiment"],
                node["compound_sentiment"],
                node["edge_density"],
                node["degree_rank"],
            ]

            if fact_check:
                node_feature.append(node["commonsense_score"])
            node_feature.append(node["client_history_credit"])
            node_feature.append(node["client_credit"])

            tensor = torch.tensor(node_feature, device=self.device)
            feature_tensors.append(tensor)

        return torch.stack(feature_tensors).float()

    def _process_edge_index(self, raw_edges):
        edge_tensor = torch.as_tensor(raw_edges, dtype=torch.long)
        if edge_tensor.dim() == 1:
            edge_tensor = edge_tensor.view(2, -1)
        return edge_tensor.to(self.device)

    def _update_client_credits(self, client_credits, doc_results: Dict, decay):
        new_credits = dict(client_credits)

        client_trust = defaultdict(
            lambda: {
                "doc_trust_scores": [],
                "client_credit": client_credits.get(client_id, 0.5),
            }
        )

        for doc_result in doc_results.values():
            client_id = doc_result["client_id"]
            client_trust[client_id]["doc_trust_scores"].append(
                doc_result["trust_score"]
            )

        for client_id, trust_data in client_trust.items():
            if len(trust_data["doc_trust_scores"]) > 0:
                old_credit = trust_data["client_credit"]
                new_trust = np.mean(trust_data["doc_trust_scores"])
                updated = decay * old_credit + (1 - decay) * new_trust
                new_credits[client_id] = np.clip(updated, 0, 1)

        return new_credits
