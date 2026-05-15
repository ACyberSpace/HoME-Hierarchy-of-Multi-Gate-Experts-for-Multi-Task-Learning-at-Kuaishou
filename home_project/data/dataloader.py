import pickle
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from config import DataConfig
from .dataset import KuairandDataset, collate_fn
from .feature_processor import FeatureProcessor


class DataLoaderFactory:
    """
    DataLoader工厂类
    
    负责:
    1. 加载训练/测试数据
    2. 加载特征词表
    3. 创建DataLoader
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_processor = None
        self.data = None
    
    def setup(self) -> None:
        """初始化数据"""
        # 加载训练/测试数据
        print(f"加载数据: {self.config.data_path}")
        with open(self.config.data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # 加载特征词表
        print(f"加载特征词表: {self.config.feature_path}")
        self.feature_processor = FeatureProcessor(self.config)
        all_features = self.feature_processor.load_features(self.config.feature_path)
        self.feature_processor.build_vocab(all_features)
        
        # 打印词表信息
        print("\n词表大小:")
        for feat in (self.config.id_cols + self.config.sparse_cols):
            if feat in self.feature_processor.vocab_sizes:
                size = self.feature_processor.vocab_sizes[feat]
                print(f"    {feat}: {size}")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        获取训练和测试DataLoader
        
        Returns:
            (train_loader, test_loader)
        """
        if self.data is None:
            self.setup()
        
        # 创建Dataset
        train_dataset = KuairandDataset(
            self.data, self.config, mode='train'
        )
        
        test_dataset = KuairandDataset(
            self.data, self.config, mode='test'
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def get_embedding_config(self) -> Dict[str, Tuple[int, int]]:
        """获取嵌入层配置"""
        return self.feature_processor.get_embedding_config()
