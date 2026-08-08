import numpy as np
import torch

from .models.swing import build_swing_model
from .models.item2vec import build_item2vec_model
from .models.dssm import build_dssm_model
from .models.mind import build_mind_model
from .models.sdm import build_sdm_model
from .models.freshness import build_freshness_model
from .data.recall_data_loaders import (
    SwingDataLoader,
    Item2VecDataLoader,
    build_dataloader,
)
from .training.trainer import train_model
from .evaluation.evaluator import (
    evaluate_recall,
    generate_recall_candidates_dssm,
    generate_recall_candidates_mind,
    generate_recall_candidates_sdm,
)


class RecallManager:
    def __init__(self, config: dict):
        self.config = config
        self.model_type = config.get("model_type", "dssm")
        self.models = {}
        self.item_feature_dims = config.get("item_feature_dims", {})
        self.user_feature_dims = config.get("user_feature_dims", {})

    def train(self, train_data: dict, user_sequences: dict, video_info=None):
        if self.model_type == 'swing':
            data_loader = SwingDataLoader(train_data)
            model = build_swing_model(self.config, item_vocab_size=self.item_feature_dims.get("video_id", 1))
            train_model('swing', model, data_loader, self.config)
            self.models['swing'] = model

        elif self.model_type == 'item2vec':
            data_loader = Item2VecDataLoader(user_sequences)
            model = build_item2vec_model(self.config, item_vocab_size=self.item_feature_dims.get("video_id", 1))
            train_model('item2vec', model, data_loader, self.config)
            self.models['item2vec'] = model

        elif self.model_type == 'dssm':
            dataloader = build_dataloader('dssm', train_data, self.item_feature_dims, batch_size=self.config.get("batch_size", 256))
            model = build_dssm_model(self.config, self.user_feature_dims, self.item_feature_dims)
            train_model('dssm', model, dataloader, self.config)
            self.models['dssm'] = model

        elif self.model_type == 'mind':
            dataloader = build_dataloader('mind', train_data, self.item_feature_dims, batch_size=self.config.get("batch_size", 256))
            model = build_mind_model(self.config, self.user_feature_dims, self.item_feature_dims)
            train_model('mind', model, dataloader, self.config)
            self.models['mind'] = model

        elif self.model_type == 'sdm':
            dataloader = build_dataloader('sdm', train_data, self.item_feature_dims, batch_size=self.config.get("batch_size", 256))
            model = build_sdm_model(self.config, self.user_feature_dims, self.item_feature_dims)
            train_model('sdm', model, dataloader, self.config)
            self.models['sdm'] = model

        elif self.model_type == 'freshness':
            if video_info is not None:
                model = build_freshness_model(self.config, video_info)
                model.fit()
                self.models['freshness'] = model

    def generate_candidates(self, user_sequences: dict, all_item_ids: list, top_k: int = 100):
        recall_results = {}

        if 'swing' in self.models:
            model = self.models['swing']
            for i, user_id in enumerate(user_sequences['user_id']):
                user_hist_items = user_sequences['long_seq'][i]
                user_hist_items = user_hist_items[user_hist_items != 0].tolist()
                candidates = model.generate_recall_candidates(user_id, user_hist_items, all_item_ids, top_k)
                recall_results[user_id] = candidates

        elif 'item2vec' in self.models:
            model = self.models['item2vec']
            for i, user_id in enumerate(user_sequences['user_id']):
                user_hist_items = user_sequences['long_seq'][i]
                user_hist_items = user_hist_items[user_hist_items != 0].tolist()
                candidates = model.generate_recall_candidates(user_id, user_hist_items, all_item_ids, top_k)
                recall_results[user_id] = candidates

        elif 'dssm' in self.models:
            recall_results = generate_recall_candidates_dssm(
                self.models['dssm'], user_sequences, all_item_ids, top_k
            )

        elif 'mind' in self.models:
            recall_results = generate_recall_candidates_mind(
                self.models['mind'], user_sequences, all_item_ids, top_k
            )

        elif 'sdm' in self.models:
            recall_results = generate_recall_candidates_sdm(
                self.models['sdm'], user_sequences, all_item_ids, top_k
            )

        elif 'freshness' in self.models:
            model = self.models['freshness']
            for i, user_id in enumerate(user_sequences['user_id']):
                user_hist_items = user_sequences['long_seq'][i]
                user_hist_items = user_hist_items[user_hist_items != 0].tolist()
                candidates = model.generate_recall_candidates(user_id, user_hist_items, all_item_ids, top_k)
                recall_results[user_id] = candidates

        return recall_results

    def evaluate(self, recall_results: dict, test_data: dict):
        return evaluate_recall(recall_results, test_data)

    def save_model(self, path: str):
        if self.model_type in ['dssm', 'mind', 'sdm'] and self.model_type in self.models:
            torch.save(self.models[self.model_type].state_dict(), path)
            print(f"模型已保存到 {path}")

    def load_model(self, path: str):
        if self.model_type in ['dssm', 'mind', 'sdm']:
            if self.model_type == 'dssm':
                model = build_dssm_model(self.config, self.user_feature_dims, self.item_feature_dims)
            elif self.model_type == 'mind':
                model = build_mind_model(self.config, self.user_feature_dims, self.item_feature_dims)
            elif self.model_type == 'sdm':
                model = build_sdm_model(self.config, self.user_feature_dims, self.item_feature_dims)
            model.load_state_dict(torch.load(path, map_location='cpu'))
            self.models[self.model_type] = model
            print(f"模型已从 {path} 加载")
