import torch
import torch.nn as nn
from typing import Dict, List

from .embedding import FeatureEmbedding


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = None,
        dropout: float = 0.1,
        activate_last: bool = True,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims
        if output_dim is not None:
            dims.append(output_dim)

        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if activate_last or not is_last:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMoE(nn.Module):
    """
    PyTorch MMoE baseline adapted from fun-rec's TensorFlow implementation.

    It shares the same FeatureEmbedding and data contract as HoME:
    forward(user_features, item_features, embeddings, short_seq, short_seq_mask)
    returns {task_name: probability}.
    """

    def __init__(
        self,
        input_dim: int,
        all_tasks: List[str],
        expert_nums: int = 4,
        expert_dnn_units: List[int] = None,
        gate_dnn_units: List[int] = None,
        task_tower_dnn_units: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        expert_dnn_units = expert_dnn_units or [128, 64]
        gate_dnn_units = gate_dnn_units or [128, 64]
        task_tower_dnn_units = task_tower_dnn_units or [128, 64]
        expert_dim = expert_dnn_units[-1]

        self.all_tasks = all_tasks
        self.experts = nn.ModuleList([
            MLPBlock(input_dim, expert_dnn_units[:-1], expert_dim, dropout)
            for _ in range(expert_nums)
        ])
        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                MLPBlock(input_dim, gate_dnn_units, dropout=dropout),
                nn.Linear(gate_dnn_units[-1], expert_nums, bias=False),
                nn.Softmax(dim=-1),
            )
            for task in all_tasks
        })
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                MLPBlock(expert_dim, task_tower_dnn_units, dropout=dropout),
                nn.Linear(task_tower_dnn_units[-1], 1),
                nn.Sigmoid(),
            )
            for task in all_tasks
        })

    def forward(
        self,
        user_features: Dict[str, torch.Tensor],
        item_features: Dict[str, torch.Tensor],
        embeddings: FeatureEmbedding,
        short_seq=None,
        short_seq_mask=None,
    ) -> Dict[str, torch.Tensor]:
        x = embeddings({**user_features, **item_features})
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        predictions = {}
        for task in self.all_tasks:
            gate = self.gates[task](x)
            tower_input = torch.sum(expert_outputs * gate.unsqueeze(-1), dim=1)
            predictions[task] = self.towers[task](tower_input).squeeze(-1)
        return predictions


class CGCLayer(nn.Module):
    def __init__(
        self,
        task_num: int,
        input_dim: int,
        task_expert_num: int,
        shared_expert_num: int,
        task_expert_dnn_units: List[int],
        shared_expert_dnn_units: List[int],
        task_gate_dnn_units: List[int],
        shared_gate_dnn_units: List[int],
        dropout: float,
        is_last: bool,
    ):
        super().__init__()
        self.task_num = task_num
        self.task_expert_num = task_expert_num
        self.shared_expert_num = shared_expert_num
        self.is_last = is_last
        self.output_dim = task_expert_dnn_units[-1]

        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                MLPBlock(input_dim, task_expert_dnn_units[:-1], self.output_dim, dropout)
                for _ in range(task_expert_num)
            ])
            for _ in range(task_num)
        ])
        self.shared_experts = nn.ModuleList([
            MLPBlock(input_dim, shared_expert_dnn_units[:-1], self.output_dim, dropout)
            for _ in range(shared_expert_num)
        ])
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                MLPBlock(input_dim, task_gate_dnn_units, dropout=dropout),
                nn.Linear(task_gate_dnn_units[-1], task_expert_num + shared_expert_num, bias=False),
                nn.Softmax(dim=-1),
            )
            for _ in range(task_num)
        ])
        if not is_last:
            self.shared_gate = nn.Sequential(
                MLPBlock(input_dim, shared_gate_dnn_units, dropout=dropout),
                nn.Linear(
                    shared_gate_dnn_units[-1],
                    task_num * task_expert_num + shared_expert_num,
                    bias=False,
                ),
                nn.Softmax(dim=-1),
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        task_expert_outputs = []
        for task_idx in range(self.task_num):
            task_outputs = [
                expert(inputs[task_idx])
                for expert in self.task_experts[task_idx]
            ]
            task_expert_outputs.append(task_outputs)

        shared_outputs = [expert(inputs[-1]) for expert in self.shared_experts]

        outputs = []
        for task_idx in range(self.task_num):
            cur_experts = torch.stack(
                task_expert_outputs[task_idx] + shared_outputs,
                dim=1,
            )
            gate = self.task_gates[task_idx](inputs[task_idx])
            outputs.append(torch.sum(cur_experts * gate.unsqueeze(-1), dim=1))

        if not self.is_last:
            all_experts = torch.stack(
                [expert for task_outputs in task_expert_outputs for expert in task_outputs]
                + shared_outputs,
                dim=1,
            )
            shared_gate = self.shared_gate(inputs[-1])
            outputs.append(torch.sum(all_experts * shared_gate.unsqueeze(-1), dim=1))

        return outputs


class PLE(nn.Module):
    """
    PyTorch PLE baseline adapted from fun-rec's TensorFlow CGC implementation.
    """

    def __init__(
        self,
        input_dim: int,
        all_tasks: List[str],
        ple_level_nums: int = 1,
        task_expert_num: int = 4,
        shared_expert_num: int = 2,
        task_expert_dnn_units: List[int] = None,
        shared_expert_dnn_units: List[int] = None,
        task_gate_dnn_units: List[int] = None,
        shared_gate_dnn_units: List[int] = None,
        task_tower_dnn_units: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        task_expert_dnn_units = task_expert_dnn_units or [128, 64]
        shared_expert_dnn_units = shared_expert_dnn_units or [128, 64]
        task_gate_dnn_units = task_gate_dnn_units or [128, 64]
        shared_gate_dnn_units = shared_gate_dnn_units or [128, 64]
        task_tower_dnn_units = task_tower_dnn_units or [128, 64]

        self.all_tasks = all_tasks
        task_num = len(all_tasks)
        layers = []
        level_input_dim = input_dim
        for level in range(ple_level_nums):
            is_last = level == ple_level_nums - 1
            layer = CGCLayer(
                task_num=task_num,
                input_dim=level_input_dim,
                task_expert_num=task_expert_num,
                shared_expert_num=shared_expert_num,
                task_expert_dnn_units=task_expert_dnn_units,
                shared_expert_dnn_units=shared_expert_dnn_units,
                task_gate_dnn_units=task_gate_dnn_units,
                shared_gate_dnn_units=shared_gate_dnn_units,
                dropout=dropout,
                is_last=is_last,
            )
            layers.append(layer)
            level_input_dim = layer.output_dim
        self.cgc_layers = nn.ModuleList(layers)

        tower_input_dim = task_expert_dnn_units[-1]
        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                MLPBlock(tower_input_dim, task_tower_dnn_units, dropout=dropout),
                nn.Linear(task_tower_dnn_units[-1], 1),
                nn.Sigmoid(),
            )
            for task in all_tasks
        })

    def forward(
        self,
        user_features: Dict[str, torch.Tensor],
        item_features: Dict[str, torch.Tensor],
        embeddings: FeatureEmbedding,
        short_seq=None,
        short_seq_mask=None,
    ) -> Dict[str, torch.Tensor]:
        x = embeddings({**user_features, **item_features})
        cgc_inputs = [x] * (len(self.all_tasks) + 1)
        for layer in self.cgc_layers:
            cgc_inputs = layer(cgc_inputs)

        return {
            task: self.towers[task](cgc_inputs[idx]).squeeze(-1)
            for idx, task in enumerate(self.all_tasks)
        }
