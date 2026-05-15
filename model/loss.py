import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class HoMEUncertaintyLoss(nn.Module):
    """
    基于不确定性的自适应多任务损失
    
    论文: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    
    损失形式:
        L_total = Σ (1/(2σ²_i)) * L_i + log(σ_i)
    
    其中 σ_i 是可学习的任务不确定性参数
    """
    
    def __init__(
        self,
        task_names: List[str],
        task_groups: Dict[str, List[str]],
        sparse_tasks: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.task_names = task_names
        self.task_groups = task_groups
        self.sparse_tasks = sparse_tasks or []
        
        # 可学习的log方差参数
        # 稀疏任务给更大的初始不确定性（更小的权重）
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(
                1.0 if name in self.sparse_tasks else 0.0
            ))
            for name in task_names
        })
        
        # 组级别的不确定性
        self.group_log_vars = nn.ParameterDict({
            group: nn.Parameter(torch.tensor(0.0))
            for group in task_groups.keys()
        })
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算加权总损失
        
        Args:
            predictions: {task_name: [B]}
            labels: {task_name: [B]}
        
        Returns:
            {
                'total_loss': 总损失,
                'task_losses': {task_name: 原始损失},
                'task_weights': {task_name: 当前权重},
                'uncertainties': {task_name: 当前σ值}
            }
        """
        total_loss = 0.0
        task_losses = {}
        task_weights = {}
        
        for task_name in self.task_names:
            if task_name not in predictions or task_name not in labels:
                continue
            
            pred = predictions[task_name]
            label = labels[task_name]
            
            # BCE损失
            task_loss = F.binary_cross_entropy(
                pred, label, reduction='mean'
            )
            task_losses[task_name] = task_loss
            
            # 任务级别权重: 1/(2σ²) = exp(-log_var) / 2
            log_var = self.log_vars[task_name]
            task_weight = torch.exp(-log_var) / 2
            
            # 确定任务所属组
            task_group = None
            for group_name, tasks in self.task_groups.items():
                if task_name in tasks:
                    task_group = group_name
                    break
            
            # 组级别权重
            if task_group and task_group in self.group_log_vars:
                group_weight = torch.exp(-self.group_log_vars[task_group]) / 2
            else:
                group_weight = 1.0
            
            # 总权重
            total_weight = task_weight * group_weight
            
            # 加权损失: (1/2σ²) * L_i + log(σ)
            weighted_loss = total_weight * task_loss + log_var / 2
            total_loss = total_loss + weighted_loss
            
            task_weights[task_name] = total_weight.item()
        
        return {
            'total_loss': total_loss,
            'task_losses': {k: v.item() for k, v in task_losses.items()},
            'task_weights': task_weights,
            'uncertainties': self.get_uncertainties()
        }
    
    def get_uncertainties(self) -> Dict[str, float]:
        """获取各任务的不确定性 σ = exp(log_var/2)"""
        return {
            name: torch.exp(self.log_vars[name] / 2).item()
            for name in self.task_names
        }
