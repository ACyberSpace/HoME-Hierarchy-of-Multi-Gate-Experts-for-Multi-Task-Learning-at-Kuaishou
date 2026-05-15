# HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning

快手KDD 2025论文《HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou》的复现实现。

## 项目结构

```
home_project/
├── config.py              # 配置管理
├── main.py                # 入口文件
├── requirements.txt       # 依赖
├── README.md             # 说明文档
├── data/                 # 数据加载模块
│   ├── __init__.py
│   ├── dataset.py        # Dataset类
│   ├── feature_processor.py  # 特征处理
│   └── dataloader.py     # DataLoader工厂
├── model/                # 模型模块
│   ├── __init__.py
│   ├── embedding.py      # 特征嵌入
│   ├── experts.py        # Expert模块
│   ├── gates.py          # 门控机制
│   ├── home.py           # HoME主模型
│   └── loss.py           # 损失函数
└── train/                # 训练模块
    ├── __init__.py
    ├── trainer.py        # 训练器
    └── evaluator.py      # 评估器
```

## 核心特性

### 1. 解决三大问题
- **Expert Collapse**: 使用Batch Normalization + Swish激活
- **Expert Degradation**: 使用Hierarchy Mask机制
- **Expert Underfitting**: 使用Feature-Gate + Self-Gate

### 2. 模型架构
```
Layer 0: Feature-Gate
Layer 1: Meta Expert Layer (全局共享 + 类别内共享)
Layer 2: Task Expert Layer (全局共享 + 类别内共享 + 任务特定)
Task Towers: 多任务预测
```

### 3. 损失函数
基于不确定性的自适应多任务加权（Kendall et al. CVPR 2018）

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
将数据文件放在项目目录下：
- `train_test_data.pkl`: 训练/测试数据
- `features.pkl`: 特征词表

### 3. 运行训练
```bash
python main.py \
    --data_path train_test_data.pkl \
    --feature_path features.pkl \
    --batch_size 4096 \
    --epochs 10
```

### 4. 参数说明
```bash
# 模型参数
--num_meta_shared 4        # Meta层全局共享专家数
--num_meta_category 2      # Meta层类别内共享专家数
--num_task_shared 2        # Task层全局共享专家数
--num_task_in_category 2   # Task层类别内共享专家数
--num_task_specific 1      # Task层任务特定专家数

# 训练参数
--batch_size 4096          # 批次大小
--lr 1e-3                  # 学习率
--epochs 10                # 训练轮数
--device cuda              # 训练设备
```

## 论文引用

```bibtex
@inproceedings{home2025,
  title={HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou},
  author={Kuaishou Research},
  booktitle={KDD},
  year={2025}
}
```
