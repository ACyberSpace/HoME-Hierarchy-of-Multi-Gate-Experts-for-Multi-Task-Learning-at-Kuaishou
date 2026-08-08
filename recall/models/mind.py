import torch
import torch.nn as nn
import torch.nn.functional as F


class MIND(nn.Module):
    def __init__(self, user_feature_dims: dict, item_feature_dims: dict, embedding_dim: int = 64, num_interests: int = 4):
        super(MIND, self).__init__()
        self.user_feature_dims = user_feature_dims
        self.item_feature_dims = item_feature_dims
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests

        self.item_embeddings = nn.ModuleDict()
        for feat_name, dim in item_feature_dims.items():
            self.item_embeddings[feat_name] = nn.Embedding(dim, embedding_dim)

        self.user_embedding = nn.ModuleDict()
        for feat_name, dim in user_feature_dims.items():
            self.user_embedding[feat_name] = nn.Embedding(dim, embedding_dim)

        self.capsule_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim * num_interests),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim * len(item_feature_dims), embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, user_features, item_features):
        short_seq = user_features["short_seq"]
        short_mask = user_features["short_mask"]

        seq_embedding = self.item_embeddings["video_id"](short_seq)
        seq_embedding = seq_embedding * short_mask.unsqueeze(-1)

        seq_len = short_mask.sum(dim=1).unsqueeze(-1)
        seq_len = torch.max(seq_len, torch.ones_like(seq_len))
        avg_embedding = seq_embedding.sum(dim=1) / seq_len

        capsule_output = self.capsule_network(avg_embedding)
        interest_embeddings = capsule_output.view(-1, self.num_interests, self.embedding_dim)

        item_embeds = []
        for feat_name in self.item_feature_dims:
            if feat_name in item_features:
                item_embeds.append(self.item_embeddings[feat_name](item_features[feat_name]))

        item_embeds = torch.cat(item_embeds, dim=1)
        item_repr = self.item_tower(item_embeds)

        item_repr = item_repr.unsqueeze(1)
        scores = F.cosine_similarity(interest_embeddings, item_repr, dim=-1)
        max_score, _ = torch.max(scores, dim=1)

        return max_score

    def get_user_interests(self, user_features):
        short_seq = user_features["short_seq"]
        short_mask = user_features["short_mask"]

        seq_embedding = self.item_embeddings["video_id"](short_seq)
        seq_embedding = seq_embedding * short_mask.unsqueeze(-1)

        seq_len = short_mask.sum(dim=1).unsqueeze(-1)
        seq_len = torch.max(seq_len, torch.ones_like(seq_len))
        avg_embedding = seq_embedding.sum(dim=1) / seq_len

        capsule_output = self.capsule_network(avg_embedding)
        interest_embeddings = capsule_output.view(-1, self.num_interests, self.embedding_dim)

        return interest_embeddings


def build_mind_model(config: dict, user_feature_dims: dict, item_feature_dims: dict):
    model = MIND(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        embedding_dim=config.get("embedding_dim", 64),
        num_interests=config.get("num_interests", 4),
    )
    return model
