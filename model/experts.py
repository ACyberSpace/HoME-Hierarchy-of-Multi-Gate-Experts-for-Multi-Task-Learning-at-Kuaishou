from collections import defaultdict

import torch
import torch.nn as nn
from typing import Dict, List

from torch.ao.quantization import default_weight_only_qconfig

from .gates import SelfGate, MultiFeatureGate

# base_Expert
class HoMEExpert(nn.Module):
    """
    HoME专家模块

    包含:
    1. Batch Normalization (解决Expert Collapse)
    2. Swish激活函数 (替代ReLU)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.swish = nn.SiLU()  # Swish = x * sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.bn(h)
        h = self.swish(h)
        return h

# feature_interaction_expert
# class HoMEExpert(nn.Module):
#     """
#     Gated Cross Network Expert
#
#     用Gated Cross Layer替换HoME Expert中的线性层
#     公式: c_{l+1} = c_0 ⊙ (W_c × c_l + b) ⊙ σ(W_g × c_l) + c_l
#     """
#
#     def __init__(self, input_dim: int, output_dim: int, num_cross_layers: int = 2):
#         super().__init__()
#
#         self.input_dim = input_dim
#         self.num_cross_layers = num_cross_layers
#
#         # 特征交叉参数 (W_l^(c), b_l)
#         self.cross_weights = nn.ModuleList([
#             nn.Linear(input_dim, input_dim)
#             for _ in range(num_cross_layers)
#         ])
#
#         # 门控网络参数 (W_l^(g))
#         self.gate_weights = nn.ModuleList([
#             nn.Linear(input_dim, input_dim)
#             for _ in range(num_cross_layers)
#         ])
#
#         # 输出投影（保持维度一致）
#         self.output_proj = nn.Linear(input_dim, output_dim)
#
#         # BatchNorm + Swish (保留HoME的改进)
#         self.bn = nn.BatchNorm1d(output_dim, eps=1e-3)
#         # self.ln = nn.LayerNorm(input_dim)
#         self.swish = nn.SiLU()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: [B, input_dim] 输入特征
#
#         Returns:
#             output: [B, input_dim] 门控交叉后的输出
#         """
#         c_0 = x  # 保存初始输入
#         c_l = x  # 当前层输入
#
#         # 多层Gated Cross
#         for i in range(self.num_cross_layers):
#             # 特征交叉: W_c × c_l + b
#             cross_term = self.cross_weights[i](c_l)  # [B, input_dim]
#
#             # 门控: σ(W_g × c_l)
#             gate = torch.sigmoid(self.gate_weights[i](c_l))  # [B, input_dim]
#
#             # Gated Cross: c_0 ⊙ cross_term ⊙ gate + c_l
#             c_l = c_0 * cross_term * gate + c_l  # [B, input_dim]
#
#         # 输出投影 + BN + Swish
#         output = self.output_proj(c_l)
#         output = self.bn(output)
#         # output = self.ln(output)
#         output = self.swish(output)
#
#         return output


class MetaExpertLayer(nn.Module):
    """
    第一层: Meta专家层

    包含:
    - 全局共享专家 (所有任务共享)
    - 类别内共享专家 (watch类 / interact类)
    """

    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        num_shared: int,
        num_category: int,
        task_groups: dict
    ):
        super().__init__()

        self.task_groups = task_groups
        self.expert_dim = expert_dim

        # 全局共享专家
        self.shared_experts = nn.ModuleList([
            HoMEExpert(input_dim, expert_dim)
            for _ in range(num_shared)
        ])

        # 类别内共享专家
        self.category_experts = nn.ModuleDict({
            category: nn.ModuleList([
                HoMEExpert(input_dim, expert_dim)
                for _ in range(num_category)
            ])
            for category in task_groups.keys()
        })

        # Meta门控网络
        self.meta_gates = nn.ModuleDict({
            category: nn.Sequential(
                nn.Linear(input_dim, num_shared + num_category),
                nn.Softmax(dim=-1)
            )
            for category in task_groups.keys()
        })
        self.shared_gates = nn.Sequential(
                nn.Linear(input_dim, num_shared + 2 * num_category),
                nn.Softmax(dim=-1)
            )
        # ========== Self-Gate (Layer 1 -> Layer 2) ==========
        self.self_gates = nn.ModuleDict({
            'shared': SelfGate(input_dim, num_shared),
            'watch': SelfGate(input_dim, num_category),
            'interact': SelfGate(input_dim, num_category)
        })

    def forward(self, x_dict: dict) -> dict:
        """
        Args:
            x: [B, input_dim]

        Returns:
            {
                'shared': [B, expert_dim],
                'watch': [B, expert_dim],
                'interact': [B, expert_dim]
            }
        """
        # 全局共享专家输出
        shared_outputs = [expert(x_dict["shared"]) for expert in self.shared_experts]
        shared_outputs = torch.stack(shared_outputs, dim=1)  # [B, num_shared, expert_dim]
        self_shared_outputs = self.self_gates["shared"](x_dict["shared"], shared_outputs)
        all_outputs = shared_outputs
        meta_outputs = {'shared': None}

        # 各类别内专家
        for category in self.task_groups.keys():
            category_outputs = [expert(x_dict[category]) for expert in self.category_experts[category]]
            category_outputs = torch.stack(category_outputs, dim=1)
            self_intra_all_outputs = self.self_gates[category](x_dict[category], category_outputs)
            # self_outputs[category] = self_intra_all_outputs
            all_outputs = torch.cat((all_outputs, category_outputs), dim=1)
            intra_all_outputs = torch.cat([shared_outputs, category_outputs], dim=1)

            gate = self.meta_gates[category](x_dict[category])
            weighted = torch.sum(intra_all_outputs * gate.unsqueeze(-1), dim=1)

            meta_outputs[category] = weighted + self_intra_all_outputs

        # 全局共享
        shared_gate = self.shared_gates(x_dict["shared"])
        meta_outputs['shared'] = torch.sum(all_outputs * shared_gate.unsqueeze(-1), dim=1) + self_shared_outputs

        return meta_outputs


class TaskExpertLayer(nn.Module):
    """
    第二层: 任务专家层

    包含三类专家:
    1. 全局共享专家 (shared): 所有任务共享
    2. 类别内共享专家 (in-category): 同类型任务共享
    3. 任务特定专家 (specific): 每个任务独有
    """

    def __init__(
        self,
        expert_dim: int,
        num_shared_experts: int,
        num_in_category_experts: int,
        num_task_specific_experts: int,
        lora_dim: int,
        num_lora: int,
        task_groups: Dict[str, List[str]],
        all_tasks: List[str]
    ):
        super().__init__()

        self.task_groups = task_groups
        self.all_tasks = all_tasks
        self.expert_dim = expert_dim

        # ========== 1. 全局共享专家 ==========
        self.shared_experts = nn.ModuleList([
            HoMEExpert(expert_dim, expert_dim)
            for _ in range(num_shared_experts)
        ])

        # ========== 2. 类别内共享专家 ==========
        self.in_category_experts = nn.ModuleDict({
            category: nn.ModuleList([
                HoMEExpert(expert_dim, expert_dim)
                for _ in range(num_in_category_experts)
            ])
            for category in task_groups.keys()
        })

        # ========== 3. 任务特定专家 ==========
        self.task_specific_experts = nn.ModuleDict({
            task: nn.ModuleList([
                HoMEExpert(expert_dim, expert_dim)
                for _ in range(num_task_specific_experts)
            ])
            for tasks in task_groups.values() for task in tasks
        })

        # ========== 门控网络 ==========
        num_experts_per_task = (
            num_shared_experts +
            num_in_category_experts +
            num_task_specific_experts
        )

        self.task_gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(expert_dim * 2, num_experts_per_task),
                nn.Softmax(dim=-1)
            )
            for tasks in task_groups.values() for task in tasks
        })

        self.local_feature_gates = nn.ModuleDict({
            key:MultiFeatureGate(
                input_dim=expert_dim,
                lora_dim=lora_dim,
                num_lora=num_lora,
                num_gates=1
            )
            for key in self.task_groups.keys()
        })
        self.task_feature_gates = nn.ModuleDict({
            key: MultiFeatureGate(
                input_dim=expert_dim,
                lora_dim=lora_dim,
                num_lora=num_lora,
                num_gates=1
            )
            for key in all_tasks
        })

        self.shared_feature_gates = MultiFeatureGate(
                input_dim=expert_dim,
                lora_dim=lora_dim,
                num_lora=num_lora,
                num_gates=1
            )

        # ========== Self-Gate (Layer 1 -> Layer 2) ==========
        self.self_gates = nn.ModuleDict({
            task: SelfGate(expert_dim, 1)
            for task in all_tasks
        })

    def forward(
        self,
        meta_outputs: Dict[str, torch.Tensor],
        task_groups: Dict[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            meta_outputs: {
                'shared': [B, expert_dim],
                'watch': [B, expert_dim],
                'interact': [B, expert_dim]
            }

        Returns:
            {task_name: [B, expert_dim]}
        """
        task_outputs = {}
        # 1. 全局共享专家输出
        global_shared_ouputs = []
        gloabal_shared_input = self.shared_feature_gates(meta_outputs['shared'])[0]
        for expert in self.shared_experts:
            global_shared_ouputs.append(expert(gloabal_shared_input))
        local_shared_ouputs = defaultdict(list)
        for category, tasks in task_groups.items():
            local_shared_input = self.local_feature_gates[category](meta_outputs[category])[0]
            for expert in self.in_category_experts[category]:
                local_shared_ouputs[category].append(expert(local_shared_input))

        for category, tasks in task_groups.items():
            for task in tasks:
                # 拼接全局共享和类别共享作为输入
                # [B, expert_dim * 2]
                # task_input = torch.cat([
                #     meta_outputs['shared'],
                #     meta_outputs[category]
                # ], dim=-1)
                task_input = self.task_feature_gates[task](meta_outputs[category])[0]
                # 收集该任务的所有专家输出
                expert_outputs = []

                # # 1. 全局共享专家输出
                # for expert in self.shared_experts:
                #     expert_outputs.append(expert(task_input))

                # # 2. 类别内共享专家输出
                # for expert in self.in_category_experts[category]:
                #     expert_outputs.append(expert(task_input))
                expert_outputs.extend(global_shared_ouputs)
                expert_outputs.extend(local_shared_ouputs[category])

                # 3. 任务特定专家输出
                temp_output = []
                for expert in self.task_specific_experts[task]:
                    x = expert(task_input)
                    temp_output.append(x)
                    expert_outputs.append(x)

                self_output = self.self_gates[task](task_input, torch.stack(temp_output, dim=1))

                # 堆叠所有专家输出: [B, num_experts, expert_dim]
                expert_outputs = torch.stack(expert_outputs, dim=1)

                # 计算门控权重
                gate_weights = self.task_gates[task](torch.cat([task_input, gloabal_shared_input], dim=-1))  # [B, num_experts]

                # 加权聚合
                task_output = torch.sum(
                    expert_outputs * gate_weights.unsqueeze(-1),
                    dim=1
                )  # [B, expert_dim]

                task_outputs[task] = task_output + self_output

        return task_outputs
