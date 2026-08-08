import torch
import torch.nn as nn
from typing import List


class FeatureGate(nn.Module):
    """
    Feature-Gate模块 (解决Expert Underfitting)
    
    使用LoRA技术为不同专家生成个性化输入
    """
    
    def __init__(self, input_dim: int, lora_dim: int, num_lora: int = 2):
        super().__init__()
        
        self.num_lora = num_lora
        
        # LoRA层
        self.lora_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, lora_dim, bias=False),
                nn.Linear(lora_dim, input_dim, bias=False)
            )
            for _ in range(num_lora)
        ])
        
        # 门控聚合
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_lora),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回特征重要性权重
        
        Args:
            x: [B, input_dim]
        
        Returns:
            [B, input_dim] 门控后的特征
        """
        # 生成多个视角的特征重要性
        importance_weights = [lora(x) for lora in self.lora_layers]
        
        # 加权聚合
        gate_weights = self.gate(x)
        combined_weight = sum(
            w * g.unsqueeze(-1) 
            for w, g in zip(importance_weights, gate_weights.T)
        )
        
        # Sigmoid归一化到(0,1)
        combined_weight = torch.sigmoid(combined_weight)
        
        return x * combined_weight


class SelfGate(nn.Module):
    """
    Self-Gate模块 (解决Expert Underfitting + 深层梯度传递)
    
    残差连接确保梯度有效传递
    """
    
    def __init__(self, expert_dim: int, num_experts: int):
        super().__init__()
        
        self.num_experts = num_experts
        
        if num_experts == 1:
            self.gate = nn.Sequential(
                nn.Linear(expert_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(expert_dim, num_experts),
                nn.Softmax(dim=-1)
            )
    
    def forward(
        self,
        inputs: torch.Tensor,
        expert_outputs: torch.Tensor,
        residual: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            expert_outputs: [B, num_experts, expert_dim]
            residual: 来自上一层的残差 [B, expert_dim]
        
        Returns:
            [B, expert_dim]
        """
        # 计算门控权重
        # gate_input = expert_outputs.mean(dim=1)  # [B, expert_dim]
        gate_input = inputs         # [B, expert_dim]
        gate_weights = self.gate(gate_input)  # [B, num_experts]
        
        # 加权聚合
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        
        # 残差连接
        if residual is not None:
            output = output + residual
        
        return output


class MultiFeatureGate(nn.Module):
    """
    多专家Feature-Gate
    
    为不同Meta专家生成个性化输入
    """
    
    def __init__(self, input_dim: int, lora_dim: int, num_lora: int, num_gates: int):
        super().__init__()
        
        self.gates = nn.ModuleList([
            FeatureGate(input_dim, lora_dim, num_lora)
            for _ in range(num_gates)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回多个门控后的特征
        
        Args:
            x: [B, input_dim]
        
        Returns:
            list of [B, input_dim]
        """
        return [gate(x) for gate in self.gates]
