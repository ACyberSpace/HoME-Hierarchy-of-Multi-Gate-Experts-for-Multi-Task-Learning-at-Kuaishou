#!/usr/bin/env python3
"""
HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning
快手短视频推荐多任务学习复现

基于论文:
    HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou
    KDD 2025

使用方法:
    python main.py --data_path /path/to/data.pkl --feature_path /path/to/features.pkl
"""

import argparse
import torch
import random
import numpy as np

from config import Config, DataConfig, ModelConfig, TrainConfig
from data import DataLoaderFactory
from model import FeatureEmbedding, HoME, HoMEUncertaintyLoss
from train import HoMETrainer


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description='HoME Multi-Task Learning'
    )

    # 数据参数
    parser.add_argument(
        '--data_path', type=str, default='./KuaiRand-1K/kuairand_train_eval.pkl',
        help='训练/测试数据路径'
    )
    parser.add_argument(
        '--feature_path', type=str, default='./KuaiRand-1K/kuairand_feature_dict.pkl',
        help='特征词表路径'
    )

    # 模型参数
    parser.add_argument(
        '--embed_dim', type=int, default=128,
        help='嵌入维度'
    )
    parser.add_argument(
        '--expert_dim', type=int, default=256,
        help='专家网络维度'
    )
    parser.add_argument(
        '--num_meta_shared', type=int, default=4,
        help='Meta层全局共享专家数量'
    )
    parser.add_argument(
        '--num_meta_category', type=int, default=2,
        help='Meta层类别内共享专家数量'
    )
    parser.add_argument(
        '--num_task_shared', type=int, default=2,
        help='Task层全局共享专家数量'
    )
    parser.add_argument(
        '--num_task_in_category', type=int, default=2,
        help='Task层类别内共享专家数量'
    )
    parser.add_argument(
        '--num_task_specific', type=int, default=1,
        help='Task层任务特定专家数量'
    )
    parser.add_argument(
        '--lora_dim', type=int, default=64,
        help='LoRA维度'
    )
    parser.add_argument(
        '--num_lora', type=int, default=2,
        help='LoRA数量'
    )

    # 训练参数
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='批次大小'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='学习率'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='训练轮数'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:1',
        help='训练设备'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print("="*60)
    print("HoME: Hierarchy of Multi-Gate Experts")
    print("="*60)

    # ========== 1. 配置 ==========
    data_config = DataConfig(
        data_path=args.data_path,
        feature_path=args.feature_path,
        batch_size=args.batch_size
    )

    model_config = ModelConfig(
        embed_dim=args.embed_dim,
        expert_dim=args.expert_dim,
        num_meta_shared_experts=args.num_meta_shared,
        num_meta_category_experts=args.num_meta_category,
        num_task_shared_experts=args.num_task_shared,
        num_task_in_category_experts=args.num_task_in_category,
        num_task_specific_experts=args.num_task_specific,
        lora_dim=args.lora_dim,
        num_lora=args.num_lora
    )

    train_config = TrainConfig(
        learning_rate=args.lr,
        max_epochs=args.epochs,
        device=args.device
    )

    config = Config(
        data=data_config,
        model=model_config,
        train=train_config
    )

    # ========== 2. 数据加载 ==========
    print("\n[1/5] 加载数据...")
    factory = DataLoaderFactory(data_config)
    train_loader, test_loader = factory.get_dataloaders()
    embed_config = factory.get_embedding_config()

    # ========== 3. 模型构建 ==========
    print("\n[2/5] 构建模型...")

    # 特征嵌入
    embeddings = FeatureEmbedding(embed_config)
    print(f"  嵌入维度: {embeddings.total_dim}")
    user_dim = sum([embed_config[x][1] for x in data_config.user_cols])
    item_dim = sum([embed_config[x][1] for x in data_config.item_cols])
    # HoME模型
    # model = HoME(
    #     embedding_dim=embeddings.total_dim,
    #     expert_dim=model_config.expert_dim,
    #     tower_dims=model_config.tower_dims,
    #     num_meta_shared_experts=model_config.num_meta_shared_experts,
    #     num_meta_category_experts=model_config.num_meta_category_experts,
    #     num_task_shared_experts=model_config.num_task_shared_experts,
    #     num_task_in_category_experts=model_config.num_task_in_category_experts,
    #     num_task_specific_experts=model_config.num_task_specific_experts,
    #     lora_dim=model_config.lora_dim,
    #     num_lora=model_config.num_lora,
    #     task_groups=data_config.task_groups,
    #     all_tasks=data_config.label_cols
    # )
    model = HoME(
        user_dim=user_dim,
        item_dim=item_dim,
        expert_dim=model_config.expert_dim,
        tower_dims=model_config.tower_dims,
        num_meta_shared_experts=model_config.num_meta_shared_experts,
        num_meta_category_experts=model_config.num_meta_category_experts,
        num_task_shared_experts=model_config.num_task_shared_experts,
        num_task_in_category_experts=model_config.num_task_in_category_experts,
        num_task_specific_experts=model_config.num_task_specific_experts,
        lora_dim=model_config.lora_dim,
        num_lora=model_config.num_lora,
        task_groups=data_config.task_groups,
        all_tasks=data_config.label_cols
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 损失函数
    criterion = HoMEUncertaintyLoss(
        task_names=data_config.label_cols,
        task_groups=data_config.task_groups,
        sparse_tasks=data_config.sparse_tasks
    )

    # ========== 4. 训练 ==========
    print("\n[3/5] 开始训练...")
    trainer = HoMETrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        config=train_config,
        embeddings=embeddings
    )
    trainer.train()

    # ========== 5. 最终评估 ==========
    print("\n[4/5] 最终评估...")
    final_metrics = trainer.evaluate()
    print("\n最终测试结果:")
    for task in data_config.label_cols:
        print(f"  {task}_AUC: {final_metrics[f'{task}_auc']:.4f}")
        print(f"  {task}_GAUC: {final_metrics[f'{task}_gauc']:.4f}")
    print(f"  平均AUC: {final_metrics['avg_auc']:.4f}")
    print(f"  平均GAUC: {final_metrics['avg_gauc']:.4f}")

    # ========== 6. 保存不确定性参数 ==========
    print("\n[5/5] 学习到的不确定性参数:")
    uncertainties = criterion.get_uncertainties()
    for task, sigma in uncertainties.items():
        weight = 1 / (2 * sigma ** 2)
        print(f"  {task}: σ={sigma:.4f}, 权重={weight:.4f}")

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)


if __name__ == "__main__":
    main()
