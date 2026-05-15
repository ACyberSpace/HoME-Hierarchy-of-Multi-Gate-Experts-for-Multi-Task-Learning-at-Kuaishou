import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
from tqdm import tqdm



class HoMETrainer:
    """
    HoME训练器

    负责:
    1. 模型训练
    2. 损失计算与反向传播
    3. 学习率调度
    4. 早停与模型保存
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config,  # TrainConfig
        embeddings: nn.Module
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.embeddings = embeddings

        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )

        # 移动到设备
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.embeddings.to(self.device)

        # 优化器：包含所有可学习参数
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.criterion.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )

        # 权重追踪
        self.weight_history = []

    def train_epoch(self, epoch: int) -> Dict:
        """
        训练一个epoch

        Returns:
            {
                'total_loss': 平均总损失,
                'task_losses': {task_name: 平均损失},
                'task_weights': {task_name: 平均权重},
                'uncertainties': {task_name: 不确定性}
            }
        """
        self.model.train()

        total_loss = 0.0
        task_loss_sums = {task: 0.0 for task in self.criterion.task_names}
        task_weight_sums = {task: 0.0 for task in self.criterion.task_names}
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="HoME模型训练", total=len(self.train_loader))):
            # 移动数据到设备
            # id_features = {
            #     k: v.to(self.device)
            #     for k, v in batch['id_features'].items()
            # }
            # sparse_features = {
            #     k: v.to(self.device)
            #     for k, v in batch['sparse_features'].items()
            # }
            user_features = {
                k: v.to(self.device)
                for k, v in batch['user_features'].items()
            }
            item_features = {
                k: v.to(self.device)
                for k, v in batch['item_features'].items()
            }
            # ========== 新增：序列特征 ==========
            short_seq = batch['short_seq'].to(self.device)
            short_seq_mask = batch['short_seq_mask'].to(self.device)
            # long_seq = batch['long_seq'].to(self.device)
            # long_seq_mask = batch['long_seq_mask'].to(self.device)
            labels = {
                k: v.to(self.device)
                for k, v in batch['labels'].items()
            }

            # 前向传播
            # predictions = self.model(
            #     id_features, sparse_features, self.embeddings,
            #     short_seq, short_seq_mask, long_seq, long_seq_mask
            # )
            predictions = self.model(
                user_features, item_features, self.embeddings,
                short_seq, short_seq_mask
            )

            # 计算损失
            loss_dict = self.criterion(predictions, labels)
            loss = loss_dict['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.criterion.parameters()),
                max_norm=self.config.grad_clip
            )

            self.optimizer.step()

            # 累计统计
            total_loss += loss.item()
            for task, task_loss in loss_dict['task_losses'].items():
                task_loss_sums[task] += task_loss
            for task, weight in loss_dict['task_weights'].items():
                task_weight_sums[task] += weight
            num_batches += 1

        # 计算平均值
        metrics = {
            'total_loss': total_loss / num_batches,
            'task_losses': {
                k: v / num_batches for k, v in task_loss_sums.items()
            },
            'task_weights': {
                k: v / num_batches for k, v in task_weight_sums.items()
            },
            'uncertainties': self.criterion.get_uncertainties()
        }

        return metrics

    # @torch.no_grad()
    # def evaluate(self) -> Dict:
    #     """
    #     评估模型
    #
    #     Returns:
    #         {task_name}_auc: 各任务AUC
    #         avg_auc: 平均AUC
    #     """
    #     self.model.eval()
    #
    #     all_predictions = {
    #         task: [] for task in self.criterion.task_names
    #     }
    #     all_labels = {
    #         task: [] for task in self.criterion.task_names
    #     }
    #
    #     for batch in self.test_loader:
    #         id_features = {
    #             k: v.to(self.device)
    #             for k, v in batch['id_features'].items()
    #         }
    #         sparse_features = {
    #             k: v.to(self.device)
    #             for k, v in batch['sparse_features'].items()
    #         }
    #         labels = {
    #             k: v.to(self.device)
    #             for k, v in batch['labels'].items()
    #         }
    #
    #         predictions = self.model(
    #             id_features, sparse_features, self.embeddings
    #         )
    #
    #         for task in self.criterion.task_names:
    #             all_predictions[task].append(predictions[task].cpu())
    #             all_labels[task].append(labels[task].cpu())
    #
    #     # 计算AUC
    #     from sklearn.metrics import roc_auc_score
    #
    #     metrics = {}
    #     for task in self.criterion.task_names:
    #         pred = torch.cat(all_predictions[task]).numpy()
    #         label = torch.cat(all_labels[task]).numpy()
    #
    #         try:
    #             metrics[f'{task}_auc'] = roc_auc_score(label, pred)
    #         except:
    #             metrics[f'{task}_auc'] = 0.5
    #
    #     metrics['avg_auc'] = sum(
    #         metrics[f'{task}_auc'] for task in self.criterion.task_names
    #     ) / len(self.criterion.task_names)
    #
    #     return metrics
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        评估模型

        Returns:
            {task_name}_auc: 各任务AUC
            {task_name}_gauc: 各任务GAUC
            avg_auc: 平均AUC
            avg_gauc: 平均GAUC
        """
        self.model.eval()

        all_predictions = {
            task: [] for task in self.criterion.task_names
        }
        all_labels = {
            task: [] for task in self.criterion.task_names
        }
        all_user_ids = []

        for batch in self.test_loader:
            user_features = {
                k: v.to(self.device)
                for k, v in batch['user_features'].items()
            }
            item_features = {
                k: v.to(self.device)
                for k, v in batch['item_features'].items()
            }
            # ========== 新增：序列特征 ==========
            short_seq = batch['short_seq'].to(self.device)
            short_seq_mask = batch['short_seq_mask'].to(self.device)
            # long_seq = batch['long_seq'].to(self.device)
            # long_seq_mask = batch['long_seq_mask'].to(self.device)
            labels = {
                k: v.to(self.device)
                for k, v in batch['labels'].items()
            }

            predictions = self.model(
                user_features, item_features, self.embeddings,
                short_seq, short_seq_mask
            )

            for task in self.criterion.task_names:
                all_predictions[task].append(predictions[task].cpu())
                all_labels[task].append(labels[task].cpu())

            # 收集user_id用于GAUC计算
            if 'user_id' in batch['user_features']:
                all_user_ids.append(batch['user_features']['user_id'].cpu().numpy())

        # 计算AUC和GAUC
        from sklearn.metrics import roc_auc_score

        metrics = {}
        gauc_sum = 0.0
        gauc_count = 0

        for task in self.criterion.task_names:
            pred = torch.cat(all_predictions[task]).numpy()
            label = torch.cat(all_labels[task]).numpy()

            # AUC
            try:
                metrics[f'{task}_auc'] = roc_auc_score(label, pred)
            except:
                metrics[f'{task}_auc'] = 0.5

            # GAUC
            if all_user_ids:
                user_ids = np.concatenate(all_user_ids)
                gauc = self._compute_gauc(pred, label, user_ids)
                metrics[f'{task}_gauc'] = gauc
                gauc_sum += gauc
                gauc_count += 1

        metrics['avg_auc'] = sum(
            metrics[f'{task}_auc'] for task in self.criterion.task_names
        ) / len(self.criterion.task_names)

        if gauc_count > 0:
            metrics['avg_gauc'] = gauc_sum / gauc_count

        return metrics

    def _compute_gauc(self, predictions: np.ndarray, labels: np.ndarray, user_ids: np.ndarray) -> float:
        """
        计算GAUC (Group AUC)

        按用户分组计算AUC，再以各用户的样本数为权重做加权平均
        """
        from sklearn.metrics import roc_auc_score

        unique_users = np.unique(user_ids)
        total_weight = 0.0
        weighted_auc_sum = 0.0

        for uid in unique_users:
            mask = (user_ids == uid)
            uid_preds = predictions[mask]
            uid_labels = labels[mask]

            n_pos = uid_labels.sum()
            n_neg = len(uid_labels) - n_pos

            # 跳过无法计算AUC的用户
            if n_pos == 0 or n_neg == 0:
                continue

            try:
                uid_auc = roc_auc_score(uid_labels, uid_preds)
                weight = len(uid_labels)
                weighted_auc_sum += uid_auc * weight
                total_weight += weight
            except:
                continue

        if total_weight == 0:
            return 0.5

        return weighted_auc_sum / total_weight

    def train(self, num_epochs: Optional[int] = None) -> None:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数，默认使用config中的值
        """
        if num_epochs is None:
            num_epochs = self.config.max_epochs

        best_auc = 0.0
        patience_counter = 0

        os.makedirs(self.config.save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*60)

            # 训练
            train_metrics = self.train_epoch(epoch)
            print(f"\n训练损失: {train_metrics['total_loss']:.4f}")

            # 打印任务权重
            print("\n任务权重:")
            for task, weight in train_metrics['task_weights'].items():
                uncertainty = train_metrics['uncertainties'][task]
                print(f"  {task}: weight={weight:.4f}, σ={uncertainty:.4f}")

            # 评估
            eval_metrics = self.evaluate()
            print(f"\n测试集AUC:")
            for task in self.criterion.task_names:
                print(f"  {task}: {eval_metrics[f'{task}_auc']:.4f}")
            print(f"  平均AUC: {eval_metrics['avg_auc']:.4f}")

            # 学习率调度
            self.scheduler.step(eval_metrics['avg_auc'])

            # 保存最佳模型
            if eval_metrics['avg_auc'] > best_auc:
                best_auc = eval_metrics['avg_auc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'criterion_state_dict': self.criterion.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': best_auc
                }, f"{self.config.save_dir}/best_model.pt")
                print(f"\n✓ 保存最佳模型 (AUC: {best_auc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"\n未改进 ({patience_counter}/{self.config.patience})")

            # 早停
            if patience_counter >= self.config.patience:
                print(f"\n早停: 连续{self.config.patience}个epoch未改进")
                break

        print(f"\n训练完成! 最佳AUC: {best_auc:.4f}")
