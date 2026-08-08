import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np


class HoMEEvaluator:
    """
    HoME评估器

    负责:
    1. 模型预测
    2. 指标计算（AUC、LogLoss等）
    """

    def __init__(
        self,
        model: nn.Module,
        embeddings: nn.Module,
        task_names: list,
        device: str = 'cuda'
    ):
        self.model = model
        self.embeddings = embeddings
        self.task_names = task_names
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )

        self.model.to(self.device)
        self.embeddings.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        预测

        Args:
            dataloader: 数据加载器

        Returns:
            {
                'predictions': {task_name: [N]},
                'labels': {task_name: [N]}
            }
        """
        all_predictions = {task: [] for task in self.task_names}
        all_labels = {task: [] for task in self.task_names}

        for batch in dataloader:
            id_features = {
                k: v.to(self.device)
                for k, v in batch['id_features'].items()
            }
            sparse_features = {
                k: v.to(self.device)
                for k, v in batch['sparse_features'].items()
            }
            labels = {
                k: v.to(self.device)
                for k, v in batch['labels'].items()
            }

            predictions = self.model(
                id_features, sparse_features, self.embeddings
            )

            for task in self.task_names:
                all_predictions[task].append(predictions[task].cpu().numpy())
                all_labels[task].append(labels[task].cpu().numpy())

        return {
            'predictions': {
                k: np.concatenate(v) for k, v in all_predictions.items()
            },
            'labels': {
                k: np.concatenate(v) for k, v in all_labels.items()
            }
        }

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            predictions: 预测值 [N]
            labels: 真实标签 [N]

        Returns:
            {
                'auc': AUC值,
                'logloss': LogLoss值,
                'accuracy': 准确率
            }
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

        metrics = {}

        try:
            metrics['auc'] = roc_auc_score(labels, predictions)
        except:
            metrics['auc'] = 0.5

        try:
            metrics['logloss'] = log_loss(labels, predictions)
        except:
            metrics['logloss'] = float('inf')

        # Accuracy at threshold 0.5
        preds_binary = (predictions > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(labels, preds_binary)

        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        完整评估

        Args:
            dataloader: 数据加载器

        Returns:
            {task_name}_{metric}: 各任务各指标
            avg_auc: 平均AUC
        """
        results = self.predict(dataloader)

        all_metrics = {}
        for task in self.task_names:
            task_metrics = self.compute_metrics(
                results['predictions'][task],
                results['labels'][task]
            )
            for metric_name, value in task_metrics.items():
                all_metrics[f'{task}_{metric_name}'] = value

        # 平均指标
        all_metrics['avg_auc'] = sum(
            all_metrics[f'{task}_auc'] for task in self.task_names
        ) / len(self.task_names)

        return all_metrics
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from typing import Dict
# import numpy as np
#
#
# class HoMEEvaluator:
#     """
#     HoME评估器
#
#     负责:
#     1. 模型预测
#     2. 指标计算（AUC、GAUC、LogLoss等）
#     """
#
#     def __init__(
#             self,
#             model: nn.Module,
#             embeddings: nn.Module,
#             task_names: list,
#             device: str = 'cuda'
#     ):
#         self.model = model
#         self.embeddings = embeddings
#         self.task_names = task_names
#         self.device = torch.device(
#             device if torch.cuda.is_available() else 'cpu'
#         )
#
#         self.model.to(self.device)
#         self.embeddings.to(self.device)
#         self.model.eval()
#
#     @torch.no_grad()
#     def predict(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
#         """
#         预测
#
#         Args:
#             dataloader: 数据加载器
#
#         Returns:
#             {
#                 'predictions': {task_name: [N]},
#                 'labels': {task_name: [N]},
#                 'user_ids': [N]  (用于GAUC计算)
#             }
#         """
#         all_predictions = {task: [] for task in self.task_names}
#         all_labels = {task: [] for task in self.task_names}
#         all_user_ids = []
#
#         for batch in dataloader:
#             id_features = {
#                 k: v.to(self.device)
#                 for k, v in batch['id_features'].items()
#             }
#             sparse_features = {
#                 k: v.to(self.device)
#                 for k, v in batch['sparse_features'].items()
#             }
#             labels = {
#                 k: v.to(self.device)
#                 for k, v in batch['labels'].items()
#             }
#
#             predictions = self.model(
#                 id_features, sparse_features, self.embeddings
#             )
#
#             for task in self.task_names:
#                 all_predictions[task].append(predictions[task].cpu().numpy())
#                 all_labels[task].append(labels[task].cpu().numpy())
#
#             # 收集user_id用于GAUC计算
#             if 'user_id' in batch['id_features']:
#                 all_user_ids.append(batch['id_features']['user_id'].numpy())
#
#         results = {
#             'predictions': {
#                 k: np.concatenate(v) for k, v in all_predictions.items()
#             },
#             'labels': {
#                 k: np.concatenate(v) for k, v in all_labels.items()
#             }
#         }
#
#         if all_user_ids:
#             results['user_ids'] = np.concatenate(all_user_ids)
#
#         return results
#
#     def compute_metrics(
#             self,
#             predictions: np.ndarray,
#             labels: np.ndarray
#     ) -> Dict[str, float]:
#         """
#         计算评估指标
#
#         Args:
#             predictions: 预测值 [N]
#             labels: 真实标签 [N]
#
#         Returns:
#             {
#                 'auc': AUC值,
#                 'logloss': LogLoss值,
#                 'accuracy': 准确率
#             }
#         """
#         from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
#
#         metrics = {}
#
#         try:
#             metrics['auc'] = roc_auc_score(labels, predictions)
#         except:
#             metrics['auc'] = 0.5
#
#         try:
#             metrics['logloss'] = log_loss(labels, predictions)
#         except:
#             metrics['logloss'] = float('inf')
#
#         # Accuracy at threshold 0.5
#         preds_binary = (predictions > 0.5).astype(int)
#         metrics['accuracy'] = accuracy_score(labels, preds_binary)
#
#         return metrics
#
#     def compute_gauc(
#             self,
#             predictions: np.ndarray,
#             labels: np.ndarray,
#             user_ids: np.ndarray
#     ) -> float:
#         """
#         计算GAUC (Group AUC)
#
#         按用户分组计算AUC，再以各用户的样本数为权重做加权平均
#
#         Args:
#             predictions: 预测值 [N]
#             labels: 真实标签 [N]
#             user_ids: 用户ID [N]
#
#         Returns:
#             gauc: 加权平均GAUC值
#         """
#         from sklearn.metrics import roc_auc_score
#
#         unique_users = np.unique(user_ids)
#         total_weight = 0.0
#         weighted_auc_sum = 0.0
#
#         for uid in unique_users:
#             mask = (user_ids == uid)
#             uid_preds = predictions[mask]
#             uid_labels = labels[mask]
#
#             n_pos = uid_labels.sum()
#             n_neg = len(uid_labels) - n_pos
#
#             # 跳过正样本或负样本为0的用户（无法计算AUC）
#             if n_pos == 0 or n_neg == 0:
#                 continue
#
#             try:
#                 uid_auc = roc_auc_score(uid_labels, uid_preds)
#                 weight = len(uid_labels)
#                 weighted_auc_sum += uid_auc * weight
#                 total_weight += weight
#             except:
#                 continue
#
#         if total_weight == 0:
#             return 0.5
#
#         return weighted_auc_sum / total_weight
#
#     def evaluate(self, dataloader: DataLoader) -> Dict:
#         """
#         完整评估
#
#         Args:
#             dataloader: 数据加载器
#
#         Returns:
#             {task_name}_{metric}: 各任务各指标
#             avg_auc: 平均AUC
#             avg_gauc: 平均GAUC
#         """
#         results = self.predict(dataloader)
#
#         all_metrics = {}
#         gauc_sum = 0.0
#         gauc_count = 0
#
#         for task in self.task_names:
#             task_metrics = self.compute_metrics(
#                 results['predictions'][task],
#                 results['labels'][task]
#             )
#             for metric_name, value in task_metrics.items():
#                 all_metrics[f'{task}_{metric_name}'] = value
#
#             # 计算GAUC
#             if 'user_ids' in results:
#                 gauc = self.compute_gauc(
#                     results['predictions'][task],
#                     results['labels'][task],
#                     results['user_ids']
#                 )
#                 all_metrics[f'{task}_gauc'] = gauc
#                 gauc_sum += gauc
#                 gauc_count += 1
#
#         # 平均指标
#         all_metrics['avg_auc'] = sum(
#             all_metrics[f'{task}_auc'] for task in self.task_names
#         ) / len(self.task_names)
#
#         if gauc_count > 0:
#             all_metrics['avg_gauc'] = gauc_sum / gauc_count
#
#         return all_metrics