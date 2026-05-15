import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import pickle


class KuairandDataset(Dataset):
    """
    Kuairand-1K 多任务数据集
    
    数据格式:
        data = {
            'train': {feature_name: ndarray, ...},
            'test': {feature_name: ndarray, ...}
        }
    """
    
    def __init__(
        self,
        data: Dict,
        config,  # DataConfig
        mode: str = 'train'
    ):
        self.config = config
        self.mode = mode
        self.data = data[mode]
        
        # 获取样本数量
        self.num_samples = len(self.data[config.id_cols[0]])
        
        # 打印统计信息
        print(f"[{mode}] 样本数: {self.num_samples:,}")
        print(f"[{mode}] 标签分布:")
        for label in config.label_cols:
            if label in self.data:
                pos_rate = self.data[label].mean()
                print(f"    {label}: {pos_rate:.4%}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'user_features': {},
            'item_features': {},
            'labels': {},
            # ========== 新增：序列特征 ==========
            'short_seq': torch.tensor(self.data['last_k_clicked_items'][idx], dtype=torch.long),
            'short_seq_mask': torch.tensor(self.data['short_seq_mask'][idx], dtype=torch.float),
            # 'long_seq': torch.tensor(self.data['long_seq'][idx], dtype=torch.long),
            # 'long_seq_mask': torch.tensor(self.data['long_seq_mask'][idx], dtype=torch.float),
        }
        
        # # ID特征
        # for feat in self.config.id_cols:
        #     if feat in self.data:
        #         sample['id_features'][feat] = torch.tensor(
        #             self.data[feat][idx], dtype=torch.long
        #         )
        #
        # # 稀疏特征
        # for feat in self.config.sparse_cols:
        #     if feat in self.data:
        #         sample['sparse_features'][feat] = torch.tensor(
        #             self.data[feat][idx], dtype=torch.long
        #         )
        # user特征
        for feat in self.config.user_cols:
            if feat in self.data:
                sample['user_features'][feat] = torch.tensor(
                    self.data[feat][idx], dtype=torch.long
                )

        # item特征
        for feat in self.config.item_cols:
            if feat in self.data:
                sample['item_features'][feat] = torch.tensor(
                    self.data[feat][idx], dtype=torch.long
                )
        
        # 多任务标签
        for label in self.config.label_cols:
            if label in self.data:
                sample['labels'][label] = torch.tensor(
                    self.data[label][idx], dtype=torch.float
                )
        
        return sample


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    批量整理函数
    
    将多个样本整理成一个batch
    """
    # batch_dict = {
    #     'id_features': {},
    #     'sparse_features': {},
    #     'labels': {},
    #     # ========== 新增：序列特征 ==========
    #     'short_seq': torch.stack([s['short_seq'] for s in batch]),
    #     'short_seq_mask': torch.stack([s['short_seq_mask'] for s in batch]),
    #     'long_seq': torch.stack([s['long_seq'] for s in batch]),
    #     'long_seq_mask': torch.stack([s['long_seq_mask'] for s in batch]),
    # }
    batch_dict = {
        'user_features': {},
        'item_features': {},
        'labels': {},
        # ========== 新增：序列特征 ==========
        'short_seq': torch.stack([s['short_seq'] for s in batch]),
        'short_seq_mask': torch.stack([s['short_seq_mask'] for s in batch]),
        # 'long_seq': torch.stack([s['long_seq'] for s in batch]),
        # 'long_seq_mask': torch.stack([s['long_seq_mask'] for s in batch]),
    }
    
    sample = batch[0]
    
    # # ID特征
    # for feat in sample['id_features'].keys():
    #     batch_dict['id_features'][feat] = torch.stack(
    #         [s['id_features'][feat] for s in batch]
    #     )
    #
    # # 稀疏特征
    # for feat in sample['sparse_features'].keys():
    #     batch_dict['sparse_features'][feat] = torch.stack(
    #         [s['sparse_features'][feat] for s in batch]
    #     )
    # ID特征
    for feat in sample['user_features'].keys():
        batch_dict['user_features'][feat] = torch.stack(
            [s['user_features'][feat] for s in batch]
        )

    # 稀疏特征
    for feat in sample['item_features'].keys():
        batch_dict['item_features'][feat] = torch.stack(
            [s['item_features'][feat] for s in batch]
        )
    
    # 标签
    for label in sample['labels'].keys():
        batch_dict['labels'][label] = torch.stack(
            [s['labels'][label] for s in batch]
        )
    
    return batch_dict
