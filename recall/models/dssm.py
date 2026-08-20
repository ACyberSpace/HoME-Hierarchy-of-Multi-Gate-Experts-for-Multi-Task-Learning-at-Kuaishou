import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(input_dim, hidden_dims, output_dim, dropout=0.0):
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DSSM(nn.Module):
    def __init__(
        self,
        user_feature_dims: dict,
        item_feature_dims: dict,
        embedding_dim: int = 64,
        user_hidden_dims=None,
        item_hidden_dims=None,
        dropout: float = 0.0,
    ):
        super(DSSM, self).__init__()
        self.user_feature_dims = user_feature_dims
        self.item_feature_dims = item_feature_dims
        self.embedding_dim = embedding_dim
        user_hidden_dims = user_hidden_dims or [embedding_dim * 2]
        item_hidden_dims = item_hidden_dims or [embedding_dim * 2]

        self.user_embedding = nn.ModuleDict()
        for feat_name, dim in user_feature_dims.items():
            self.user_embedding[feat_name] = nn.Embedding(dim, embedding_dim)

        self.item_embedding = nn.ModuleDict()
        for feat_name, dim in item_feature_dims.items():
            self.item_embedding[feat_name] = nn.Embedding(dim, embedding_dim)

        self.user_tower = _make_mlp(
            embedding_dim * len(user_feature_dims),
            user_hidden_dims,
            embedding_dim,
            dropout,
        )
        self.item_tower = _make_mlp(
            embedding_dim * len(item_feature_dims),
            item_hidden_dims,
            embedding_dim,
            dropout,
        )

    def forward(self, user_features, item_features):
        user_embeds = []
        for feat_name, embedding in self.user_embedding.items():
            if feat_name in user_features:
                user_embeds.append(embedding(user_features[feat_name]))
        if len(user_embeds) == 0:
            user_embeds = [torch.zeros(user_features[list(user_features.keys())[0]].shape[0], self.embedding_dim)]

        user_embeds = torch.cat(user_embeds, dim=1)
        user_repr = self.user_tower(user_embeds)

        item_embeds = []
        for feat_name, embedding in self.item_embedding.items():
            if feat_name in item_features:
                item_embeds.append(embedding(item_features[feat_name]))
        if len(item_embeds) == 0:
            item_embeds = [torch.zeros(item_features[list(item_features.keys())[0]].shape[0], self.embedding_dim)]

        item_embeds = torch.cat(item_embeds, dim=1)
        item_repr = self.item_tower(item_embeds)

        score = F.cosine_similarity(user_repr, item_repr)
        return score

    def get_user_repr(self, user_features):
        user_embeds = []
        for feat_name, embedding in self.user_embedding.items():
            if feat_name in user_features:
                user_embeds.append(embedding(user_features[feat_name]))
        if len(user_embeds) == 0:
            user_embeds = [torch.zeros(user_features[list(user_features.keys())[0]].shape[0], self.embedding_dim)]

        user_embeds = torch.cat(user_embeds, dim=1)
        return self.user_tower(user_embeds)

    def get_item_repr(self, item_features):
        item_embeds = []
        for feat_name, embedding in self.item_embedding.items():
            if feat_name in item_features:
                item_embeds.append(embedding(item_features[feat_name]))
        if len(item_embeds) == 0:
            item_embeds = [torch.zeros(item_features[list(item_features.keys())[0]].shape[0], self.embedding_dim)]

        item_embeds = torch.cat(item_embeds, dim=1)
        return self.item_tower(item_embeds)


def build_dssm_model(config: dict, user_feature_dims: dict, item_feature_dims: dict):
    model = DSSM(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        embedding_dim=config.get("embedding_dim", 64),
        user_hidden_dims=config.get("dssm_user_hidden_dims"),
        item_hidden_dims=config.get("dssm_item_hidden_dims"),
        dropout=config.get("dssm_dropout", 0.0),
    )
    return model
