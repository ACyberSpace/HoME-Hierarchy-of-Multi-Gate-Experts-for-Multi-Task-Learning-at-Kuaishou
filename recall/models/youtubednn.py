import torch
import torch.nn as nn
import torch.nn.functional as F


class YouTubeDNN(nn.Module):
    def __init__(self, user_feature_dims: dict, item_vocab_size: int, embedding_dim: int = 64, hidden_dims=None):
        super().__init__()
        self.user_feature_dims = user_feature_dims
        self.item_vocab_size = int(item_vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = hidden_dims or [128, 64]

        self.item_embedding = nn.Embedding(self.item_vocab_size, self.embedding_dim)
        self.user_embedding = nn.ModuleDict()
        for feat_name, dim in user_feature_dims.items():
            self.user_embedding[feat_name] = nn.Embedding(int(dim), self.embedding_dim)

        input_dim = self.embedding_dim * (len(user_feature_dims) + 1)
        layers = []
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, self.embedding_dim))
        self.user_tower = nn.Sequential(*layers)

    def get_user_repr(self, user_features):
        user_embeds = []
        for feat_name, embedding in self.user_embedding.items():
            if feat_name in user_features:
                user_embeds.append(embedding(user_features[feat_name]))

        short_seq = user_features["short_seq"]
        short_mask = user_features["short_mask"].float()
        seq_emb = self.item_embedding(short_seq)
        seq_emb = seq_emb * short_mask.unsqueeze(-1)
        seq_len = short_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        hist_repr = seq_emb.sum(dim=1) / seq_len
        user_embeds.append(hist_repr)

        user_input = torch.cat(user_embeds, dim=1)
        return self.user_tower(user_input)

    def get_item_repr(self, item_features):
        return self.item_embedding(item_features["video_id"])

    def forward(self, user_features, item_features):
        user_repr = self.get_user_repr(user_features)
        item_repr = self.get_item_repr(item_features)
        return F.cosine_similarity(user_repr, item_repr)


def build_youtubednn_model(config: dict, user_feature_dims: dict, item_feature_dims: dict):
    return YouTubeDNN(
        user_feature_dims=user_feature_dims,
        item_vocab_size=item_feature_dims.get("video_id", 1),
        embedding_dim=config.get("embedding_dim", 64),
        hidden_dims=config.get("youtubednn_hidden_dims", [128, 64]),
    )
