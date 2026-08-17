import random
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .feature_column import FeatureColumn, FEATURE_GROUPS, RECALL_USER_FEATURES, RECALL_ITEM_FEATURES
from .labels import build_recall_positive_mask, build_user_positive_items


class SwingDataLoader:
    def __init__(self, train_data: Dict, user_sequences: Dict = None):
        self.train_data = train_data
        self.user_sequences = user_sequences
        self.user_item_dict = {}

    def load_data(self):
        if self.user_sequences is not None and "full_sequences" in self.user_sequences:
            for user_id, seq in zip(
                self.user_sequences["user_id"],
                self.user_sequences["full_sequences"],
            ):
                clean_seq = [int(item_id) for item_id in seq if int(item_id) != 0]
                if clean_seq:
                    self.user_item_dict[int(user_id)] = clean_seq
            return

        user_ids = self.train_data["user_id"]
        video_ids = self.train_data["video_id"]
        positive_mask = build_recall_positive_mask(self.train_data)
        for user_id, video_id, is_positive in zip(user_ids, video_ids, positive_mask):
            if not is_positive:
                continue
            if user_id not in self.user_item_dict:
                self.user_item_dict[user_id] = []
            self.user_item_dict[user_id].append(video_id)

    def get_user_item_pairs(self) -> List[Tuple[int, int, int]]:
        samples = []
        for user_id, items in self.user_item_dict.items():
            for item_id in items:
                samples.append((user_id, item_id, 1))
        return samples


class Item2VecDataLoader:
    def __init__(self, user_sequences: Dict, train_data: Dict = None):
        self.user_sequences = user_sequences
        self.train_data = train_data
        self.sequences = []

    def load_data(self):
        if self.train_data is not None:
            user_positive_items = {}
            positive_mask = build_recall_positive_mask(self.train_data)
            for user_id, video_id, is_positive in zip(
                self.train_data["user_id"],
                self.train_data["video_id"],
                positive_mask,
            ):
                if not is_positive:
                    continue
                user_positive_items.setdefault(int(user_id), [])
                user_positive_items[int(user_id)].append(int(video_id))
            for seq in user_positive_items.values():
                if len(seq) >= 2:
                    self.sequences.append([str(x) for x in seq])
        elif "full_sequences" in self.user_sequences:
            for seq in self.user_sequences["full_sequences"]:
                if len(seq) >= 2:
                    self.sequences.append([str(x) for x in seq])
        else:
            for i in range(len(self.user_sequences["user_id"])):
                seq = self.user_sequences["long_seq"][i]
                seq = seq[seq != 0].tolist()
                if len(seq) >= 2:
                    self.sequences.append([str(x) for x in seq])

    def generate_samples(self) -> List[List[str]]:
        return self.sequences


class DSSMDataset(Dataset):
    def __init__(self, train_data: Dict, item_feature_dims: Dict, num_negatives: int = 4):
        self.train_data = train_data
        self.item_feature_dims = item_feature_dims
        self.num_negatives = num_negatives
        self.all_items = np.array(list(set(train_data["video_id"])))
        self.positive_mask = build_recall_positive_mask(train_data)
        self.positive_indices = np.where(self.positive_mask)[0]
        self.user_positive_items = build_user_positive_items(train_data)
        self.item_index = {}
        for idx, video_id in enumerate(train_data["video_id"]):
            if video_id not in self.item_index:
                self.item_index[video_id] = idx
        self._pre_sample_negatives()

    def _pre_sample_negatives(self):
        self.neg_item_ids_list = []
        for idx in self.positive_indices:
            user_id = int(self.train_data["user_id"][idx])
            pos_item_id = self.train_data["video_id"][idx]
            user_positive_items = self.user_positive_items.get(user_id, set())
            candidate_items = [
                int(item_id)
                for item_id in self.all_items
                if int(item_id) not in user_positive_items
            ]
            if not candidate_items:
                candidate_items = [
                    int(item_id)
                    for item_id in self.all_items
                    if int(item_id) != int(pos_item_id)
                ]

            if len(candidate_items) >= self.num_negatives:
                neg_item_ids = random.sample(candidate_items, self.num_negatives)
            else:
                neg_item_ids = [
                    candidate_items[random.randint(0, len(candidate_items) - 1)]
                    for _ in range(self.num_negatives)
                ] if candidate_items else []
            self.neg_item_ids_list.append(neg_item_ids)

    def __len__(self):
        return len(self.positive_indices)

    def __getitem__(self, idx):
        data_idx = self.positive_indices[idx]

        user_features = {}
        user_features["short_seq"] = self.train_data["short_seq"][data_idx]
        user_features["short_mask"] = self.train_data["short_mask"][data_idx]
        for feat_name in RECALL_USER_FEATURES:
            if feat_name in self.train_data:
                user_features[feat_name] = self.train_data[feat_name][data_idx]

        pos_item_features = {}
        for feat_name in RECALL_ITEM_FEATURES:
            if feat_name in self.train_data:
                pos_item_features[feat_name] = self.train_data[feat_name][data_idx]

        neg_item_features_list = []
        for neg_item_id in self.neg_item_ids_list[idx]:
            if neg_item_id in self.item_index:
                neg_idx = self.item_index[neg_item_id]
                neg_item_features = {}
                for feat_name in RECALL_ITEM_FEATURES:
                    if feat_name in self.train_data:
                        neg_item_features[feat_name] = self.train_data[feat_name][neg_idx]
                neg_item_features_list.append(neg_item_features)
            else:
                neg_item_features_list.append({"video_id": neg_item_id})

        return {
            "user_features": user_features,
            "pos_item_features": pos_item_features,
            "neg_item_features_list": neg_item_features_list,
        }


class MINDDataset(Dataset):
    def __init__(self, train_data: Dict, item_feature_dims: Dict):
        self.train_data = train_data
        self.item_feature_dims = item_feature_dims
        self.labels = build_recall_positive_mask(train_data).astype(np.float32)

    def __len__(self):
        return len(self.train_data["user_id"])

    def __getitem__(self, idx):
        user_features = {}
        user_features["short_seq"] = self.train_data["short_seq"][idx]
        user_features["short_mask"] = self.train_data["short_mask"][idx]
        for feat_name in RECALL_USER_FEATURES:
            if feat_name in self.train_data:
                user_features[feat_name] = self.train_data[feat_name][idx]

        item_features = {}
        for feat_name in RECALL_ITEM_FEATURES:
            if feat_name in self.train_data:
                item_features[feat_name] = self.train_data[feat_name][idx]

        return {
            "user_features": user_features,
            "item_features": item_features,
            "label": self.labels[idx],
            "short_seq_len": np.sum(self.train_data["short_mask"][idx]),
        }


class SDMDataset(Dataset):
    def __init__(self, train_data: Dict, item_feature_dims: Dict):
        self.train_data = train_data
        self.item_feature_dims = item_feature_dims
        self.labels = build_recall_positive_mask(train_data).astype(np.float32)

    def __len__(self):
        return len(self.train_data["user_id"])

    def __getitem__(self, idx):
        user_features = {}
        user_features["short_seq"] = self.train_data["short_seq"][idx]
        user_features["short_mask"] = self.train_data["short_mask"][idx]
        user_features["long_seq"] = self.train_data["long_seq"][idx]
        user_features["long_mask"] = self.train_data["long_mask"][idx]
        for feat_name in RECALL_USER_FEATURES:
            if feat_name in self.train_data:
                user_features[feat_name] = self.train_data[feat_name][idx]

        item_features = {}
        for feat_name in RECALL_ITEM_FEATURES:
            if feat_name in self.train_data:
                item_features[feat_name] = self.train_data[feat_name][idx]

        return {
            "user_features": user_features,
            "item_features": item_features,
            "label": self.labels[idx],
            "short_seq_len": np.sum(self.train_data["short_mask"][idx]),
            "long_seq_len": np.sum(self.train_data["long_mask"][idx]),
        }


def get_dataloader(dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


def build_dataloader(model_type: str, train_data: Dict, item_feature_dims: Dict, batch_size: int = 256):
    if model_type == 'dssm':
        dataset = DSSMDataset(train_data, item_feature_dims)
    elif model_type == 'mind':
        dataset = MINDDataset(train_data, item_feature_dims)
    elif model_type == 'sdm':
        dataset = SDMDataset(train_data, item_feature_dims)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return get_dataloader(dataset, batch_size=batch_size, shuffle=True)
