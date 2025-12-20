from agent.auxiliary import Auxiliary
from llm.message import (
    Message,
    MessageContent,
    ROLE_SYSTEM,
    ROLE_USER,
    TYPE_SETTING,
    TYPE_CONTENT,
)
import core.agent_prompt as AgentPrompt

from utils.logger import Logger
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import re
import numpy as np
from typing import Dict
import os
import json
import threading


class GraphBuilder:

    def __init__(
        self, logger: Logger, auxiliary: Auxiliary, cache_dir, assets_path, dataset_type
    ):
        self.edge_threshold = 0.7
        self.keyword_threshold = 1
        self.logger = logger
        self.auxiliary = auxiliary
        self.cache_dir = cache_dir
        self.dataset_type = dataset_type
        self.category_map = self._load_category_map(cache_dir)
        self.stopwords = self._stopwords(assets_path)
        self.commonsense_map = self._load_commonsense_map(cache_dir)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.lock = threading.Lock()

    def build_graph(
        self, clients_data, client_credits, client_history_credits, fact_check=False
    ):
        doc_features = self._initialize_graph_data(clients_data, fact_check)
        return self._build_graph_data(doc_features, client_credits, client_history_credits, fact_check)

    def get_category_map(self):
        return self.category_map

    def _load_category_map(self, cache_dir):
        category_map = None
        filepath = os.path.join(cache_dir, "category_map.pt")
        if os.path.exists(filepath):
            category_map = torch.load(filepath, map_location=self.auxiliary.device)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            category_map = self._init_category_map()
            torch.save(category_map, filepath)
        return category_map

    # only for saving memory, record each doc validate result
    def _load_commonsense_map(self, cache_dir):
        commonsense_map = {}

        filepath = os.path.join(cache_dir, "commonsense_map.jsonl")
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(filepath):
            return commonsense_map
        
        self.logger.info(f"Start to load commonsense map from {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                info = json.loads(line)
                commonsense_map[info["doc_id"]] = info

        return commonsense_map

    def _append_commonsense(self, cache_dir, info):
        filepath = os.path.join(cache_dir, "commonsense_map.jsonl")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
            f.flush()

    def _build_graph_data(
        self,
        doc_features: Dict,
        client_credits,
        client_history_credits,
        fact_check=False,
    ):
        keywords_voc = set()
        for doc_feature in doc_features.values():
            keywords_voc.update(doc_feature["keywords"])

        if keywords_voc and len(keywords_voc) >= 2:
            corpus_doc_count = len(doc_features)

            vectorzier = CountVectorizer(vocabulary=keywords_voc, binary=False)
            doc_keywords = [
                " ".join(doc_feature["keywords"])
                for doc_feature in doc_features.values()
            ]

            tf_matrix = vectorzier.transform(doc_keywords).toarray()
            doc_freq = (tf_matrix > 0).sum(axis=0)
            idf = np.log(corpus_doc_count / (doc_freq + 1e-6))
            tf_idf = tf_matrix * idf
            tf_idf[tf_idf < 0] = 0

            if np.any(tf_idf < 0):
                self.logger.warning(f"Negative TF-IDF detected: min={np.min(tf_idf)}")

            for idx, doc_id in enumerate(doc_features):
                doc_features[doc_id].update({"tf_idf": tf_idf[idx]})

        semantic_matrix = self._build_semantic_matrix(doc_features)
        category_matrix = self._build_category_matrix(doc_features)

        node_features, edge_index = self._create_graph_data(
            doc_features,
            client_credits,
            client_history_credits,
            semantic_matrix,
            category_matrix,
            fact_check,
        )
        return node_features, edge_index, doc_features

    def _analyze_sentiment(self, text):
        return self.sentiment_analyzer.polarity_scores(text)

    def _validate_commonsense(self, doc_id, content, dataset_type, use_cache=True):
        validate_info = None
        if use_cache:
            validate_info = self.commonsense_map.get(doc_id)

        if validate_info:
            return validate_info.get("commonsense_score", 0.0)

        with self.lock:
            validate_info = self.commonsense_map.get(doc_id)
            if not validate_info:
                validate_info = self._ask_commonsense(content, dataset_type)
                if validate_info and isinstance(validate_info, dict):
                    validate_info.update({"doc_id": doc_id})
                    self.commonsense_map[doc_id] = validate_info
                    self._append_commonsense(self.cache_dir, validate_info)

        return validate_info.get("commonsense_score", 0.0) if validate_info else 0.0

    def _ask_commonsense(self, content, dataset_type):
        if dataset_type == "medical":
            prompt = AgentPrompt.medical_commonsense_validate()
        elif dataset_type == "finance":
            prompt = AgentPrompt.financial_commonsense_validate()
        else:
            raise ValueError(f"Unknown dataset type {dataset_type}")

        messages = [
            Message(
                ROLE_SYSTEM,
                [MessageContent(TYPE_SETTING, prompt)],
            ),
            Message(ROLE_USER, [MessageContent(TYPE_CONTENT, content)]),
        ]
        response = self.auxiliary.get_api_generate_model().generate(messages)
        self.logger.debug(f"Ask medical commonsense result:\n{response}")
        return AgentPrompt.extract_json(response)

    def _create_graph_data(
        self,
        doc_features: Dict,
        client_credits,
        client_history_credits,
        semantic_matrix,
        category_matrix,
        fact_check=False,
    ):
        feature_pool = {
            "content_lens": [],
            "n_keywords": [],
            "all_tf_idf": [],
            "categories": [],
        }

        for doc_feature in doc_features.values():
            feature_pool["content_lens"].append(len(doc_feature["content"]))
            kw_count = len(doc_feature["keywords"])
            feature_pool["n_keywords"].append(kw_count)
            feature_pool["all_tf_idf"].append(
                np.sum(doc_feature.get("tf_idf", np.zeros(1)))
            )
            feature_pool["categories"].append(doc_feature["category"])

        global_stats = {
            "median_content": np.median(feature_pool["content_lens"]),
            "median_keywords": np.median(feature_pool["n_keywords"]),
            "median_tfidf": np.median(feature_pool["all_tf_idf"]),
            "category_dist": Counter(feature_pool["categories"]),
        }
        total_docs = len(doc_features)

        edge_index = self._create_edges(semantic_matrix, category_matrix)
        edge_degrees = np.zeros(total_docs, dtype=int)
        if edge_index.nelement() > 0:
            rows = edge_index[0].numpy()
            cols = edge_index[1].numpy()
            edge_degrees = np.bincount(
                np.concatenate([rows, cols]), minlength=total_docs
            )

        node_features = []
        for idx, (doc_id, doc_feature) in enumerate(doc_features.items()):
            content_rel = len(doc_feature["content"]) / (
                global_stats["median_content"] + 1e-8
            )

            kw_count = len(doc_feature["keywords"])
            kw_rel = kw_count / (global_stats["median_keywords"] + 1e-8)
            tfidf = np.sum(doc_feature["tf_idf"])

            category_ratio = (
                global_stats["category_dist"][doc_feature["category"]] / total_docs
            )

            doc_idx = list(doc_features.keys()).index(doc_id)
            global_sim = np.mean(semantic_matrix[doc_idx])
            local_sim = self._local_semantic_sim(doc_idx, semantic_matrix)

            edge_density = edge_degrees[idx] / total_docs * 10
            degree_rank = np.log1p(edge_degrees[idx])
            content_ratio = np.log1p(content_rel)

            median_tfidf = global_stats["median_tfidf"]
            kw_tfidf_ratio = np.log1p(tfidf / (median_tfidf + 1e-8))
            if np.isnan(kw_tfidf_ratio):
                raise ValueError(f"NaN detected in {tfidf} / {median_tfidf}")

            kw_count_ratio = np.log1p(kw_rel)
            category_rarity = category_ratio

            beta = 0.4
            semantic_cohesion = (1 - beta) * np.log1p(global_sim) + beta * np.log1p(
                local_sim
            )
            tfidf_semantic = kw_tfidf_ratio * semantic_cohesion

            sentiment_scores = doc_feature["sentiment"]

            client_id = doc_feature["client_id"]
            client_credit = client_credits.get(client_id, 0.5)
            client_history_credit = client_history_credits.get(
                client_id, 1.0
            )  # 0.5 diff make first round unknown

            node_feature = {
                "doc_id": doc_id,
                "client_history_credit": client_history_credit,
                "client_credit": client_credit,
                "content_ratio": content_ratio,
                "kw_tfidf_ratio": kw_tfidf_ratio,
                "kw_count_ratio": kw_count_ratio,
                "category_rarity": category_rarity,
                "semantic_cohesion": semantic_cohesion,
                "tfidf_semantic": tfidf_semantic,
                "neg_sentiment": sentiment_scores["neg"],
                "neu_sentiment": sentiment_scores["neu"],
                "pos_sentiment": sentiment_scores["pos"],
                "compound_sentiment": sentiment_scores["compound"],
                "edge_density": edge_density,
                "degree_rank": degree_rank,
            }

            if fact_check:
                node_feature["commonsense_score"] = doc_feature["commonsense_score"]

            node_features.append(node_feature)

        return node_features, edge_index

    def _create_edges(self, semantic_sim, category_sim, alpha=0.6, threshold=0.7):
        combined_sim = alpha * semantic_sim + (1 - alpha) * category_sim
        rows, cols = np.where(combined_sim > threshold)
        combined = np.array([rows, cols])
        return torch.tensor(combined, dtype=torch.long)

    def _local_semantic_sim(self, doc_idx, semantic_matrix, k=5):
        similarities = semantic_matrix[doc_idx]
        topk_idx = np.argpartition(similarities, -k)[-k:]
        return np.mean(similarities[topk_idx])

    def _get_category_one_hot(self, category):
        category_map = self.category_map
        category = category_map.get(category, category_map["Unknown"])
        return category["one_hot"]

    def _get_category_embedding(self, category):
        category_map = self.category_map
        category = category_map.get(category, category_map["Unknown"])
        return category["embedding"]

    def _build_semantic_matrix(self, doc_features: Dict):
        embeddings = torch.stack(
            [doc_feature["specific_semantic"] for doc_feature in doc_features.values()]
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.T).cpu().numpy()
        semantic_matrix = MinMaxScaler().fit_transform(sim_matrix)

        n = sim_matrix.shape[0]
        mask = np.ones_like(sim_matrix) - np.eye(n)
        avg_sim = (semantic_matrix * mask).sum(axis=1) / (n - 1)
        for i, doc_feature in enumerate(doc_features.values()):
            doc_feature["avg_sim"] = avg_sim[i]

        return semantic_matrix

    def _build_category_matrix(self, doc_features: Dict):
        embeddings = torch.stack(
            [
                self._get_category_embedding(doc_feature["category"])
                for doc_feature in doc_features.values()
            ]
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.T).cpu().numpy()
        semantic_matrix = MinMaxScaler().fit_transform(sim_matrix)
        return semantic_matrix

    def _initialize_graph_data(self, clients_data, fact_check, use_cache=True):

        doc_features = {}

        for client in clients_data:
            client_id = client["source"]
            docs = client["data"]
            n_doc = len(docs)
            if n_doc == 0:
                continue

            for doc in docs:
                doc_id = doc["doc_id"]
                content = doc["content"]
                category = doc["category"]
                specific_semantic = self.auxiliary.specific_encode(content).squeeze()
                keywords = self._extract_keywords(content)
                sentiment_scores = self._analyze_sentiment(content)

                doc_features[doc_id] = {
                    "client_id": client_id,
                    "doc_id": doc_id,
                    "category": category,
                    "keywords": keywords,
                    "content": content,
                    "sentiment": sentiment_scores,
                    "specific_semantic": specific_semantic,
                }

                if fact_check:
                    commonsense_score = self._validate_commonsense(
                        doc_id, content, self.dataset_type, use_cache=use_cache
                    )
                    self.logger.debug(
                        f"Validate {doc_id} commonsense get result {commonsense_score}"
                    )
                    doc_features[doc_id].update(
                        {"commonsense_score": commonsense_score}
                    )

        return doc_features

    def _clinical_tokenizer(self, text):
        text = re.sub(r"(?<=\w)-(?=\w)", "--", text.lower())
        tokens = re.findall(r"\b[a-z]{3,}(?:--[a-z]{3,})*\b", text)
        return [t.replace("--", "-") for t in tokens]

    def _extract_keywords(self, document):
        tokens = self._clinical_tokenizer(document)
        enhanced_stopwords = self.stopwords | {"document", "note", "detail", "include"}

        phrases = []
        for n in [1, 2, 3]:
            phrases += [
                " ".join(tokens[i : i + n])
                for i in range(len(tokens) - n + 1)
                if not any(t in enhanced_stopwords for t in tokens[i : i + n])
            ]

        return [
            kw
            for kw in set(phrases)
            if 2 <= len(kw) <= 35 and sum(len(word) for word in kw.split()) >= 6
        ]

    def _init_category_map(self):
        from core.const import MEDICAL_CATEGORY, FINANCE_CATEGORY

        category_map = {}

        if self.dataset_type == "medical":
            descriptions = MEDICAL_CATEGORY
        elif self.dataset_type == "finance":
            descriptions = FINANCE_CATEGORY
        else:
            raise ValueError(f"Unknown dataset type {self.dataset_type}")

        categories = list(descriptions.keys())
        all_descriptions = [descriptions[cat] for cat in categories]

        batch_embeddings = self.auxiliary.specific_encode(all_descriptions)

        indices = torch.arange(len(categories))
        one_hot_vectors = torch.nn.functional.one_hot(
            indices, num_classes=len(categories)
        ).float()

        category_map = {
            cat: {"embedding": batch_embeddings[i], "one_hot": one_hot_vectors[i]}
            for i, cat in enumerate(categories)
        }
        return category_map

    def _stopwords(self, assets_path):
        english_stopwords = self._load_stopwords(assets_path, "english-stopwords.txt")
        clinical_stopwords = self._load_stopwords(assets_path, "clinical-stopwords.txt")
        return english_stopwords.union(clinical_stopwords)

    def _load_stopwords(self, assets_path, filename):
        filepath = os.path.join(assets_path, filename)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return {line.strip().lower() for line in f if line.strip()}
