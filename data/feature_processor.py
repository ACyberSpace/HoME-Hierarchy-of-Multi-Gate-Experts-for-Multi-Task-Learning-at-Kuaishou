import pickle
from typing import Dict, Tuple

import numpy as np


class FeatureProcessor:
    """
    特征处理器
    
    负责:
    1. 加载特征词表
    2. 构建词表大小映射
    3. 生成嵌入层配置
    """
    
    def __init__(self, config):
        self.config = config
        self.vocab_sizes: Dict[str, int] = {}
        
    def load_features(self, feature_path: str) -> Dict:
        """
        加载特征词表文件
        
        Args:
            feature_path: 特征文件路径
        
        Returns:
            all_features: 特征词表字典
        """
        with open(feature_path, 'rb') as f:
            all_features = pickle.load(f)
        return all_features
    
    def build_vocab(self, all_features: Dict) -> Dict[str, int]:
        """
        构建词表大小映射
        
        Args:
            all_features: 特征词表字典
        
        Returns:
            vocab_sizes: {feature_name: vocab_size}
        """
        for feat_name, feat_info in all_features.items():
            if isinstance(feat_info, dict):
                # 字典格式: {'num_classes': N} 或类似结构
                self.vocab_sizes[feat_name] = feat_info.get('num_classes', 
                                                            feat_info.get('vocab_size', 0))
            elif isinstance(feat_info, (np.int64, float)):
                # 直接是词表大小
                self.vocab_sizes[feat_name] = int(feat_info)
            elif isinstance(feat_info, (list, tuple)):
                # 是词表列表
                self.vocab_sizes[feat_name] = len(feat_info)
            else:
                self.vocab_sizes[feat_name] = 0
        
        return self.vocab_sizes
    
    def get_embedding_config(self) -> Dict[str, Tuple[int, int]]:
        """
        获取嵌入层配置
        
        Returns:
            {feature_name: (vocab_size, embed_dim)}
        """
        embedding_config = {}
        
        all_features = self.config.id_cols + self.config.sparse_cols
        
        for feat_name in all_features:
            if feat_name in self.vocab_sizes and self.vocab_sizes[feat_name] > 0:
                vocab_size = self.vocab_sizes[feat_name]
                # 嵌入维度经验公式: min(6 * cardinality^0.25, 64)
                embed_dim = min(int(6 * (vocab_size ** 0.25)), 64)
                # 确保最小嵌入维度
                embed_dim = max(embed_dim, 4)
                embedding_config[feat_name] = (vocab_size, embed_dim)
        
        return embedding_config
