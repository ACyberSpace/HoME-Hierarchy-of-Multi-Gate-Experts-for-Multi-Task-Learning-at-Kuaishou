"""
全链路流程串联：多路召回 → 候选融合 → HoME精排 → 指标评估

流程：
1. 加载训练好的召回模型和精排模型
2. 对测试集用户执行多路召回，生成候选集
3. 将召回候选构建为精排输入样本
4. HoME精排模型推理打分
5. 计算评估指标（AUC、GAUC、HitRate、NDCG）
"""

import pickle
import numpy as np
import pandas as pd
import torch
import joblib
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
import time

from config import DataConfig


# ==================== 精排推理数据集 ====================

class RankInferenceDataset(Dataset):
    """
    精排推理数据集
    将召回候选构建为精排模型输入
    """

    def __init__(
            self,
            user_features: Dict[str, np.ndarray],
            candidate_items: List[int],
            item_feature_index: Dict[int, Dict],
            user_sequences: Dict,
            data_config: DataConfig,
    ):
        """
        Args:
            user_features: 用户特征字典 {feature_name: value}
            candidate_items: 召回候选物品ID列表
            item_feature_index: 视频特征索引 {video_id: feature_dict}
            user_sequences: 用户序列信息
            data_config: 排序特征配置
        """
        self.user_features = user_features
        self.candidate_items = candidate_items
        self.item_feature_index = item_feature_index
        self.user_sequences = user_sequences
        self.data_config = data_config

        # 构建样本
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        """构建推理样本"""
        user_id = self.user_features.get("user_id", 0)

        for item_id in self.candidate_items:
            sample = {"user_id": user_id, "video_id": item_id}

            # 用户特征
            for key in self.data_config.user_cols:
                sample[key] = self.user_features.get(key, 0)

            # 视频特征
            item_features = self.item_feature_index.get(item_id, {})
            for key in self.data_config.item_cols:
                if key == "video_id":
                    sample[key] = item_id
                    continue
                value = item_features.get(key, 0)
                if key == "tag" and isinstance(value, list):
                    value = value[0] if value else 0
                sample[key] = value

            # 序列特征
            user_sequences = self.user_sequences or {}
            sample["short_seq"] = user_sequences.get("short_seq", np.zeros(50, dtype=np.int64))
            sample["short_seq_mask"] = user_sequences.get("short_mask", np.zeros(50, dtype=np.int64))

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def rank_collate_fn(batch):
    """精排推理的collate函数"""
    result = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], np.ndarray):
            result[key] = torch.tensor(np.array(values)).long()
        elif isinstance(values[0], list):
            # 序列特征
            max_len = max(len(v) for v in values)
            padded = np.zeros((len(values), max_len), dtype=np.int64)
            for i, v in enumerate(values):
                padded[i, :len(v)] = v
            result[key] = torch.tensor(padded).long()
        elif isinstance(values[0], (int, np.integer)):
            result[key] = torch.tensor(values).long()
        elif isinstance(values[0], float):
            result[key] = torch.tensor(values).float()
        else:
            result[key] = torch.tensor(values).long()

    return result


# ==================== 评估指标 ====================

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算AUC"""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_gauc(labels: np.ndarray, scores: np.ndarray, user_ids: np.ndarray) -> float:
    """计算GAUC"""
    user_auc_list = []
    user_count_list = []

    for uid in np.unique(user_ids):
        mask = user_ids == uid
        uid_labels = labels[mask]
        uid_scores = scores[mask]

        if len(np.unique(uid_labels)) < 2:
            continue

        auc = compute_auc(uid_labels, uid_scores)
        user_auc_list.append(auc)
        user_count_list.append(len(uid_labels))

    if not user_auc_list:
        return 0.0

    # 加权平均
    total = sum(user_count_list)
    gauc = sum(auc * count for auc, count in zip(user_auc_list, user_count_list)) / total

    return gauc


def compute_hit_rate(labels: np.ndarray, scores: np.ndarray, k: int = 50) -> float:
    """计算HitRate@K"""
    # 按分数排序，取Top-K
    top_k_indices = np.argsort(scores)[-k:]
    hit = np.sum(labels[top_k_indices])
    return hit / k if k > 0 else 0.0


def compute_ndcg(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """计算NDCG@K"""
    # 按分数排序
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order][:k]

    # DCG
    dcg = 0.0
    for i, label in enumerate(sorted_labels):
        dcg += (2 ** label - 1) / np.log2(i + 2)

    # IDCG
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = 0.0
    for i, label in enumerate(ideal_labels):
        idcg += (2 ** label - 1) / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ==================== 全链路Pipeline ====================

class FullPipeline:
    """
    全链路推荐Pipeline

    流程：
    1. 加载数据和模型
    2. 多路召回生成候选集
    3. 构建精排样本
    4. HoME精排推理
    5. 评估指标计算
    """

    def __init__(
            self,
            data_path: str,
            recall_checkpoints: Dict[str, str],
            rank_checkpoint: str,
            data_config: Optional[DataConfig] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            data_path: 数据路径
            recall_checkpoints: 召回模型checkpoint路径 {model_name: path}
            rank_checkpoint: 精排模型checkpoint路径
            device: 设备
        """
        self.data_path = Path(data_path)
        self.device = device

        # 数据
        self.train_eval = None
        self.feature_dict = None
        self.user_sequences = None
        self.video_info = None
        self.item_feature_index = None

        # 模型
        self.recall_managers = []
        self.rank_model = None
        self.embeddings = None

        # 配置
        self.recall_checkpoints = recall_checkpoints
        self.rank_checkpoint = rank_checkpoint
        self.data_config = data_config or DataConfig()

    def load_data(self):
        """加载基础数据"""
        print("加载基础数据...")

        self.train_eval = joblib.load(self.data_path / "kuairand_train_eval.pkl")

        with open(self.data_path / "kuairand_feature_dict.pkl", "rb") as f:
            self.feature_dict = pickle.load(f)

        self.user_sequences = joblib.load(self.data_path / "user_sequences.pkl")

        with open(self.data_path / "video_info.pkl", "rb") as f:
            self.video_info = pickle.load(f)

        print(f"  训练集: {len(self.train_eval['train']['user_id'])} 条")
        print(f"  测试集: {len(self.train_eval['test']['user_id'])} 条")
        print(f"  用户数: {len(self.user_sequences['user_id'])}")
        print(f"  视频数: {len(self.video_info)}")

    def _build_item_feature_index(self) -> Dict[int, Dict]:
        """Build item feature lookup from video_info plus train/test samples."""
        item_index = {}

        if isinstance(self.video_info, pd.DataFrame):
            item_index.update(
                self.video_info.set_index("video_id").to_dict(orient="index")
            )
        elif isinstance(self.video_info, dict):
            if "video_id" in self.video_info:
                for idx, vid in enumerate(self.video_info["video_id"]):
                    item_index[int(vid)] = {
                        key: values[idx]
                        for key, values in self.video_info.items()
                        if key != "video_id"
                    }
            else:
                item_index.update(self.video_info)

        for split in ("train", "test"):
            data = self.train_eval[split]
            for idx, vid in enumerate(data["video_id"]):
                if int(vid) not in item_index:
                    item_index[int(vid)] = {}
                for key in self.data_config.item_cols:
                    if key != "video_id" and key in data:
                        item_index[int(vid)][key] = data[key][idx]

        return item_index

    def load_recall_models(self, model_configs: Dict):
        """加载召回模型"""
        from recall.manager import RecallManager

        print("加载召回模型...")
        self.recall_managers = []

        for model_type, checkpoint_path in self.recall_checkpoints.items():
            config = dict(model_configs.get(model_type, {}))
            config["model_type"] = model_type
            config.setdefault("item_feature_dims", {
                "video_id": self.feature_dict.get("video_id", 1),
                "author_id": self.feature_dict.get("author_id", 1),
                "music_id": self.feature_dict.get("music_id", 1),
            })
            config.setdefault("user_feature_dims", {
                "user_id": self.feature_dict.get("user_id", 1),
            })

            manager = RecallManager(config)
            if checkpoint_path:
                manager.load_model(checkpoint_path)
            elif model_type == "freshness":
                manager.train(self.train_eval["train"], self.user_sequences, self.video_info)
            else:
                raise ValueError(f"Missing checkpoint for recall model: {model_type}")
            self.recall_managers.append(manager)

    def load_rank_model(self, rank_model, embeddings):
        """加载精排模型"""
        print("加载精排模型...")
        self.rank_model = rank_model.to(self.device)
        self.embeddings = embeddings.to(self.device)
        if self.rank_checkpoint:
            checkpoint = torch.load(self.rank_checkpoint, map_location=self.device)
            self.rank_model.load_state_dict(checkpoint["model_state_dict"])
            if "embedding_state_dict" in checkpoint:
                self.embeddings.load_state_dict(checkpoint["embedding_state_dict"])
        self.rank_model.eval()
        self.embeddings.eval()
        print("  精排模型加载完成")

    def _fuse_recall_results(
            self,
            recall_result_list: List[Dict[int, List[int]]],
            user_ids: List[int],
            top_k: int = 200,
    ) -> Dict[int, List[int]]:
        """Fuse multi-channel recall by reciprocal-rank scores."""
        fused = {}
        for user_id in user_ids:
            scores = defaultdict(float)
            for channel_weight, results in enumerate(recall_result_list, start=1):
                for rank, item_id in enumerate(results.get(user_id, []), start=1):
                    scores[int(item_id)] += 1.0 / (channel_weight * rank)
            fused[user_id] = [
                item_id for item_id, _ in
                sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            ]
        return fused

    def generate_recall_candidates(self, user_ids: List[int], top_k: int = 200) -> Dict[int, List[int]]:
        """为测试用户生成召回候选"""
        print(f"为 {len(user_ids)} 个用户生成召回候选...")

        all_item_ids = list(set(self.train_eval["train"]["video_id"]))
        recall_result_list = []
        for manager in self.recall_managers:
            results = manager.generate_candidates(
                self.user_sequences,
                all_item_ids,
                top_k=top_k,
            )
            recall_result_list.append(results)

        return self._fuse_recall_results(recall_result_list, user_ids, top_k=top_k)

    def build_rank_samples(
            self,
            recall_results: Dict[int, List[int]],
            test_data: Dict,
    ) -> Tuple[List[Dict], List[int], List[int]]:
        """
        将召回候选构建为精排样本

        Returns:
            samples: 精排样本列表
            all_labels: 对应标签
            all_user_ids: 对应用户ID
        """
        print("构建精排样本...")

        # 构建测试集标签索引：(user_id, video_id) -> label
        test_labels = {}
        for i in range(len(test_data["user_id"])):
            key = (test_data["user_id"][i], test_data["video_id"][i])
            test_labels[key] = {
                label: test_data[label][i]
                for label in self.data_config.label_cols
                if label in test_data
            }

        # 构建测试集用户特征索引
        test_user_features = defaultdict(dict)
        for i in range(len(test_data["user_id"])):
            uid = test_data["user_id"][i]
            if uid not in test_user_features:
                for key in self.data_config.user_cols:
                    if key in test_data:
                        test_user_features[uid][key] = test_data[key][i]

        # 构建用户序列索引
        user_seq_dict = {}
        for i, uid in enumerate(self.user_sequences["user_id"]):
            user_seq_dict[uid] = {
                "short_seq": self.user_sequences["short_seq"][i],
                "short_mask": self.user_sequences["short_mask"][i],
            }

        all_samples = []
        all_labels = []
        all_user_ids = []

        for user_id, candidates in tqdm(recall_results.items(), desc="构建样本"):
            if user_id not in test_user_features:
                continue

            user_feat = test_user_features[user_id]
            candidate_items = [int(c) for c in candidates]

            # 构建数据集
            dataset = RankInferenceDataset(
                user_features=user_feat,
                candidate_items=candidate_items,
                item_feature_index=self.item_feature_index,
                user_sequences=user_seq_dict.get(user_id, {}),
                data_config=self.data_config,
            )

            # 获取标签
            for sample in dataset.samples:
                vid = sample["video_id"]
                if isinstance(vid, torch.Tensor):
                    vid = vid.item()

                key = (user_id, vid)
                label_info = test_labels.get(
                    key,
                    {label: 0 for label in self.data_config.label_cols}
                )

                all_samples.append(sample)
                all_labels.append(label_info)
                all_user_ids.append(user_id)

        print(f"  构建精排样本: {len(all_samples)} 条")
        return all_samples, all_labels, all_user_ids

    def _split_rank_batch(self, batch_input: Dict[str, torch.Tensor]):
        user_features = {
            key: batch_input[key].long()
            for key in self.data_config.user_cols
            if key in batch_input
        }
        item_features = {
            key: batch_input[key].long()
            for key in self.data_config.item_cols
            if key in batch_input
        }
        short_seq = batch_input.get("short_seq")
        short_seq_mask = batch_input.get("short_seq_mask")
        if short_seq is None:
            batch_size = next(iter(user_features.values())).shape[0]
            short_seq = torch.zeros(batch_size, 50, dtype=torch.long, device=self.device)
            short_seq_mask = torch.zeros(batch_size, 50, dtype=torch.float, device=self.device)
        return user_features, item_features, short_seq.long(), short_seq_mask.float()

    def rank_inference(self, samples: List[Dict], batch_size: int = 256) -> Dict[str, np.ndarray]:
        """精排模型推理"""
        print("精排模型推理...")

        all_predictions = defaultdict(list)
        if not samples:
            return {}

        for i in tqdm(range(0, len(samples), batch_size), desc="精排推理"):
            batch = samples[i:i + batch_size]
            batch_input = rank_collate_fn(batch)

            # 将输入移到设备
            batch_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch_input.items()}

            with torch.no_grad():
                user_features, item_features, short_seq, short_seq_mask = self._split_rank_batch(batch_input)
                outputs = self.rank_model(
                    user_features,
                    item_features,
                    self.embeddings,
                    short_seq,
                    short_seq_mask,
                )

                for task_name, score in outputs.items():
                    all_predictions[task_name].extend(score.cpu().numpy().tolist())

        # 转换为numpy
        predictions = {task: np.array(scores) for task, scores in all_predictions.items()}

        return predictions

    def evaluate(
            self,
            predictions: Dict[str, np.ndarray],
            labels: List[Dict],
            user_ids: List[int],
    ) -> Dict:
        """计算评估指标"""
        print("计算评估指标...")

        results = {}
        user_ids = np.array(user_ids)

        for task_name in self.data_config.label_cols:
            if task_name not in predictions:
                continue

            task_labels = np.array([l[task_name] for l in labels])
            task_scores = predictions[task_name]

            # AUC
            auc = compute_auc(task_labels, task_scores)

            # GAUC
            gauc = compute_gauc(task_labels, task_scores, user_ids)

            # HitRate@50
            hit_rate = compute_hit_rate(task_labels, task_scores, k=50)

            # NDCG@10
            ndcg = compute_ndcg(task_labels, task_scores, k=10)

            results[task_name] = {
                "AUC": auc,
                "GAUC": gauc,
                "HitRate@50": hit_rate,
                "NDCG@10": ndcg,
            }

        # 打印结果
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)

        for task_name, metrics in results.items():
            print(f"\n[{task_name}]")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        if results:
            avg_auc = np.mean([m["AUC"] for m in results.values()])
            avg_gauc = np.mean([m["GAUC"] for m in results.values()])
            print(f"\n[平均]")
            print(f"  AUC: {avg_auc:.4f}")
            print(f"  GAUC: {avg_gauc:.4f}")
        else:
            print("\n没有可评估的精排预测结果")

        return results

    def run(
            self,
            model_configs: Dict,
            rank_model: nn.Module,
            embeddings: nn.Module,
            batch_size: int = 256,
            max_users: Optional[int] = None,
    ):
        """
        执行完整Pipeline

        Args:
            model_configs: 召回模型配置
            rank_model: 精排模型
            batch_size: 推理批次大小
            max_users: 最大评估用户数（调试用）
        """
        start_time = time.time()

        # 1. 加载数据
        self.load_data()
        self.item_feature_index = self._build_item_feature_index()

        # 2. 加载召回模型
        self.load_recall_models(model_configs)

        # 3. 加载精排模型
        self.load_rank_model(rank_model, embeddings)

        # 4. 获取测试用户
        test_user_ids = list(set(self.train_eval["test"]["user_id"]))
        if max_users:
            test_user_ids = test_user_ids[:max_users]
        print(f"\n评估用户数: {len(test_user_ids)}")

        # 5. 多路召回
        recall_results = self.generate_recall_candidates(test_user_ids)

        # 6. 构建精排样本
        samples, labels, user_ids = self.build_rank_samples(
            recall_results, self.train_eval["test"]
        )

        # 7. 精排推理
        predictions = self.rank_inference(samples, batch_size=batch_size)

        # 8. 评估
        results = self.evaluate(predictions, labels, user_ids)

        # 9. 输出耗时
        elapsed = time.time() - start_time
        print(f"\n总耗时: {elapsed:.1f}s")

        return results


# ==================== 便捷入口函数 ====================

def run_full_pipeline(
        data_path: str,
        recall_checkpoints: Dict[str, str],
        rank_model: nn.Module,
        embeddings: nn.Module,
        rank_checkpoint: str = "",
        recall_model_configs: Optional[Dict] = None,
        data_config: Optional[DataConfig] = None,
        device: str = None,
        batch_size: int = 256,
        max_users: Optional[int] = None,
) -> Dict:
    """
    一键运行全链路Pipeline

    Args:
        data_path: 数据路径
        recall_checkpoints: 召回模型checkpoint路径
        rank_model: 精排模型（已加载权重）
        recall_model_configs: 召回模型配置（可选）
        device: 设备
        batch_size: 推理批次大小
        max_users: 最大评估用户数

    Returns:
        评估结果字典
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if recall_model_configs is None:
        recall_model_configs = {}

    pipeline = FullPipeline(
        data_path=data_path,
        recall_checkpoints=recall_checkpoints,
        rank_checkpoint=rank_checkpoint,
        data_config=data_config,
        device=device,
    )

    return pipeline.run(
        model_configs=recall_model_configs,
        rank_model=rank_model,
        embeddings=embeddings,
        batch_size=batch_size,
        max_users=max_users,
    )
