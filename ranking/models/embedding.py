import torch
import torch.nn as nn
from typing import Dict, Tuple


class FeatureEmbedding(nn.Module):
    """
    特征嵌入层
    
    为所有离散特征创建嵌入层
    """
    
    def __init__(self, embedding_config: Dict[str, Tuple[int, int]]):
        """
        Args:
            embedding_config: {feature_name: (vocab_size, embed_dim)}
        """
        super().__init__()
        
        self.embeddings = nn.ModuleDict()
        for feat_name, (vocab_size, embed_dim) in embedding_config.items():
            self.embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0  # 0作为padding
            )
        
        # 计算总嵌入维度
        self.total_dim = sum(embed_dim for _, embed_dim in embedding_config.values())
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: {特征名: [B]}
        
        Returns:
            embeddings: [B, total_dim]
        """
        embeds = []
        for feat_name, embed_layer in self.embeddings.items():
            if feat_name in batch:
                embeds.append(embed_layer(batch[feat_name]))
        
        return torch.cat(embeds, dim=-1)
