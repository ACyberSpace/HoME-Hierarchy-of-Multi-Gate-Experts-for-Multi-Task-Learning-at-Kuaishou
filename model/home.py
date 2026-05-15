import torch
import torch.nn as nn
from typing import Dict, List

from .SeqModel import SDMInterestNetwork, DIN
from .embedding import FeatureEmbedding
from .experts import MetaExpertLayer, TaskExpertLayer
from .gates import MultiFeatureGate, SelfGate


class HoME(nn.Module):
    """
    HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning
    
    完整三层结构:
    Layer 0: Feature-Gate (输入层)
    Layer 1: Meta Expert Layer (全局共享 + 类别内共享)
    Layer 2: Task Expert Layer (全局共享 + 类别内共享 + 任务特定)
    """
    
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        expert_dim: int,
        tower_dims: List[int],
        # Layer 1 参数
        num_meta_shared_experts: int,
        num_meta_category_experts: int,
        # Layer 2 参数
        num_task_shared_experts: int,
        num_task_in_category_experts: int,
        num_task_specific_experts: int,
        # 其他参数
        lora_dim: int,
        num_lora: int,
        task_groups: Dict[str, List[str]],
        all_tasks: List[str]
    ):
        super().__init__()
        
        self.task_groups = task_groups
        self.all_tasks = all_tasks
        # self.user_init_proj = nn.Linear(user_dim, 64)
        # self.item_init_proj = nn.Linear(item_dim, 64)
        # self.all_init_proj = nn.Linear(2*64+user_dim+item_dim, expert_dim)
        # self.sdm_interest = SDMInterestNetwork(input_dim=user_dim)
        self.din_interest = DIN()
        
        # ========== Layer 0: Feature-Gate ==========
        self.feature_gates = MultiFeatureGate(
            input_dim=user_dim+item_dim+64,
            lora_dim=lora_dim,
            num_lora=num_lora,
            num_gates=3  # shared, watch, interact
        )
        
        # ========== Layer 1: Meta Expert Layer ==========
        self.meta_experts = MetaExpertLayer(
            input_dim=user_dim+item_dim+64,
            expert_dim=expert_dim,
            num_shared=num_meta_shared_experts,
            num_category=num_meta_category_experts,
            task_groups=task_groups
        )
        
        # # ========== Self-Gate (Layer 1 -> Layer 2) ==========
        # self.self_gates = nn.ModuleDict({
        #     'shared': SelfGate(expert_dim, num_meta_shared_experts),
        #     'watch': SelfGate(expert_dim, num_meta_category_experts),
        #     'interact': SelfGate(expert_dim, num_meta_category_experts)
        # })
        
        # ========== Layer 2: Task Expert Layer ==========
        self.task_experts = TaskExpertLayer(
            expert_dim=expert_dim,
            num_shared_experts=num_task_shared_experts,
            num_in_category_experts=num_task_in_category_experts,
            num_task_specific_experts=num_task_specific_experts,
            lora_dim=lora_dim,
            num_lora=num_lora,
            task_groups=task_groups,
            all_tasks=all_tasks
        )
        
        # ========== Task Towers ==========
        self.towers = nn.ModuleDict({
            task: self._build_tower(expert_dim, tower_dims)
            for task in all_tasks
        })
    
    def _build_tower(self, input_dim: int, tower_dims: List[int]) -> nn.Module:
        """构建任务塔"""
        layers = []
        prev_dim = input_dim
        
        for dim in tower_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        user_features: Dict[str, torch.Tensor],
        item_features: Dict[str, torch.Tensor],
        embeddings: FeatureEmbedding,
        short_seq, short_seq_mask
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            id_features: ID特征 {特征名: [B]}
            sparse_features: 稀疏特征 {特征名: [B]}
            embeddings: 特征嵌入层
        
        Returns:
            predictions: {task_name: [B]}
        """
        # Layer 0: 特征嵌入 + Feature-Gate
        # all_features = {**user_features, **item_features}
        # x = embeddings(all_features)
        user_feat = {**user_features}
        user_emb = embeddings(user_feat)  # [B, embed_dim]

        # 获取序列item的嵌入
        short_seq_emb = embeddings.embeddings['video_id'](short_seq)  # [B, seq_len, emb_dim]
        # long_seq_emb = embeddings.embeddings['video_id'](long_seq)

        # # SDM提取长短期兴趣
        # interest = self.sdm_interest(
        #     short_seq_emb, short_seq_mask,
        #     long_seq_emb, long_seq_mask,
        #     user_emb
        # )
        item_feat = {**item_features}

        item_emb = embeddings(item_feat)
        interest = self.din_interest(item_emb[:,:64], short_seq_emb, short_seq_mask)
        # 拼接原始特征 + 短期兴趣 + 长期兴趣
        x = torch.cat([user_emb, interest, item_emb], dim=-1)
        
        # 为三类专家生成个性化输入
        x_shared, x_watch, x_interact = self.feature_gates(x)

        x_dict = {
            "shared": x_shared,
            "watch": x_watch,
            "interact": x_interact
        }
        # Layer 1: Meta Experts
        meta_outputs = self.meta_experts(x_dict)

        # meta_outputs_shared = self.meta_experts(x_shared)
        # meta_outputs_watch = self.meta_experts(x_watch)
        # meta_outputs_interact = self.meta_experts(x_interact)
        
        # # 选择对应类别的meta输出
        # meta_outputs = {
        #     'shared': meta_outputs_shared['shared'],
        #     'watch': meta_outputs_watch['watch'],
        #     'interact': meta_outputs_interact['interact']
        # }
        # 选择对应类别的meta输出
        meta_outputs = {
            'shared': meta_outputs['shared'],
            'watch': meta_outputs['watch'],
            'interact': meta_outputs['interact']
        }
        
        # Layer 2: Task Experts
        task_outputs = self.task_experts(meta_outputs, self.task_groups)
        
        # Task Towers: 生成预测
        predictions = {}
        for task in self.all_tasks:
            if task in task_outputs:
                predictions[task] = self.towers[task](task_outputs[task]).squeeze(-1)
        
        return predictions
