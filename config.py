#!/usr/bin/env python3
"""
HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning
快手短视频推荐多任务学习复现

基于论文:
    HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou
    KDD 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str = './KuaiRand-1K/kuairand_train_eval.pkl'
    feature_path: str = './KuaiRand-1K/kuairand_feature_dict.pkl'
    batch_size: int = 256
    num_workers: int = 0
    
    # 标签列（多任务）
    label_cols: List[str] = field(default_factory=lambda: [
        'is_click',       # 全局共享任务
        'long_view',      # 观看类任务
        'is_like',        # 交互类任务
        'is_comment',     # 交互类任务（稀疏）
        'is_forward',     # 交互类任务（稀疏）
        'is_follow',      # 交互类任务
    ])
    
    # ID特征（高基数）
    id_cols: List[str] = field(default_factory=lambda: [
        'user_id', 'video_id', 'author_id', "music_id"
    ])
    
    # 稀疏特征（已分桶）
    sparse_cols: List[str] = field(default_factory=lambda: [
        "user_active_degree", "is_live_streamer", "is_video_author", "tag",
        "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range",
        "music_type", "tab", "video_type", "upload_type", "visible_status"
    ])

    # user特征（高基数）
    user_cols: List[str] = field(default_factory=lambda: [
        'user_id', "user_active_degree", "is_live_streamer", "is_video_author", "tag",
        "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range",
    ])

    # teim特征（已分桶）
    item_cols: List[str] = field(default_factory=lambda: [
        'video_id', 'author_id', "music_id",
        "music_type", "tab", "video_type", "upload_type", "visible_status"
    ])
    
    # 任务分组（对应HoME的Hierarchy Mask）
    task_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'watch': ['long_view'],
        'interact': ['is_click', 'is_like', 'is_comment', 'is_forward', 'is_follow']
    })
    
    # 稀疏任务列表（用于损失加权）
    sparse_tasks: List[str] = field(default_factory=lambda: [
        'is_comment', 'is_forward', 'is_follow'
    ])


@dataclass
class ModelConfig:
    """模型配置"""
    # 特征嵌入维度（自动计算）
    embed_dim: int = 128
    
    # ========== Layer 1: Meta Expert Layer ==========
    num_meta_shared_experts: int = 4      # 全局共享专家
    num_meta_category_experts: int = 2    # 类别内共享专家
    
    # ========== Layer 2: Task Expert Layer ==========
    num_task_shared_experts: int = 2     # 任务层的全局共享专家
    num_task_in_category_experts: int = 2 # 任务层的类别内共享专家
    num_task_specific_experts: int = 1    # 任务特定专家
    
    # Expert维度
    expert_dim: int = 256
    
    # Feature-Gate配置
    lora_dim: int = 64
    num_lora: int = 2
    
    # 塔网络配置
    tower_dims: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class TrainConfig:
    """训练配置"""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 10
    grad_clip: float = 5.0
    patience: int = 3
    device: str = "cuda"
    save_dir: str = "checkpoints"


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
