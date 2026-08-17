import torch
import torch.nn as nn
import torch.nn.functional as F


class SDM(nn.Module):
    def __init__(self, user_feature_dims: dict, item_feature_dims: dict, embedding_dim: int = 64, num_heads: int = 4):
        super(SDM, self).__init__()
        self.user_feature_dims = user_feature_dims
        self.item_feature_dims = item_feature_dims
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.item_embeddings = nn.ModuleDict()
        for feat_name, dim in item_feature_dims.items():
            self.item_embeddings[feat_name] = nn.Embedding(dim, embedding_dim)

        self.user_embedding = nn.ModuleDict()
        for feat_name, dim in user_feature_dims.items():
            self.user_embedding[feat_name] = nn.Embedding(dim, embedding_dim)

        self.short_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=True)
        self.short_attention = nn.MultiheadAttention(embedding_dim * 2, num_heads, batch_first=True)
        self.short_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        self.long_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim * len(user_feature_dims), embedding_dim),
            nn.ReLU(),
        )

        self.gate_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.Sigmoid(),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim * len(item_feature_dims), embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, user_features, item_features):
        short_seq = user_features["short_seq"]
        short_mask = user_features["short_mask"]
        long_seq = user_features["long_seq"]
        long_mask = user_features["long_mask"]

        short_embedding = self.item_embeddings["video_id"](short_seq)
        long_embedding = self.item_embeddings["video_id"](long_seq)

        short_lstm_output, _ = self.short_lstm(short_embedding)
        short_lstm_output = short_lstm_output * short_mask.unsqueeze(-1)
        short_attention_output, _ = self.short_attention(short_lstm_output, short_lstm_output, short_lstm_output)
        short_interest = self.short_proj(short_attention_output.mean(dim=1))

        long_embedding = long_embedding * long_mask.unsqueeze(-1)
        long_attention_output, _ = self.long_attention(long_embedding, long_embedding, long_embedding)
        long_interest = long_attention_output.mean(dim=1)

        user_embeds = []
        for feat_name, embedding in self.user_embedding.items():
            if feat_name in user_features:
                user_embeds.append(embedding(user_features[feat_name]))
        if user_embeds:
            user_repr = self.user_tower(torch.cat(user_embeds, dim=1))
        else:
            user_repr = torch.zeros(short_seq.shape[0], self.embedding_dim, device=short_seq.device)

        gate_input = torch.cat([short_interest, long_interest, user_repr], dim=-1)
        gate = self.gate_layer(gate_input)

        final_interest = gate * short_interest + (1 - gate) * long_interest

        item_embeds = []
        for feat_name in self.item_feature_dims:
            if feat_name in item_features:
                item_embeds.append(self.item_embeddings[feat_name](item_features[feat_name]))

        item_embeds = torch.cat(item_embeds, dim=1)
        item_repr = self.item_tower(item_embeds)

        score = F.cosine_similarity(final_interest, item_repr)
        return score

    def get_user_repr(self, user_features):
        short_seq = user_features["short_seq"]
        short_mask = user_features["short_mask"]
        long_seq = user_features["long_seq"]
        long_mask = user_features["long_mask"]

        short_embedding = self.item_embeddings["video_id"](short_seq)
        long_embedding = self.item_embeddings["video_id"](long_seq)

        short_lstm_output, _ = self.short_lstm(short_embedding)
        short_lstm_output = short_lstm_output * short_mask.unsqueeze(-1)
        short_attention_output, _ = self.short_attention(short_lstm_output, short_lstm_output, short_lstm_output)
        short_interest = self.short_proj(short_attention_output.mean(dim=1))

        long_embedding = long_embedding * long_mask.unsqueeze(-1)
        long_attention_output, _ = self.long_attention(long_embedding, long_embedding, long_embedding)
        long_interest = long_attention_output.mean(dim=1)

        user_embeds = []
        for feat_name, embedding in self.user_embedding.items():
            if feat_name in user_features:
                user_embeds.append(embedding(user_features[feat_name]))
        if user_embeds:
            user_repr = self.user_tower(torch.cat(user_embeds, dim=1))
        else:
            user_repr = torch.zeros(short_seq.shape[0], self.embedding_dim, device=short_seq.device)

        gate_input = torch.cat([short_interest, long_interest, user_repr], dim=-1)
        gate = self.gate_layer(gate_input)

        final_interest = gate * short_interest + (1 - gate) * long_interest
        return final_interest

    def get_item_repr(self, item_features):
        item_embeds = []
        for feat_name in self.item_feature_dims:
            if feat_name in item_features:
                item_embeds.append(self.item_embeddings[feat_name](item_features[feat_name]))
        item_embeds = torch.cat(item_embeds, dim=1)
        return self.item_tower(item_embeds)


def build_sdm_model(config: dict, user_feature_dims: dict, item_feature_dims: dict):
    model = SDM(
        user_feature_dims=user_feature_dims,
        item_feature_dims=item_feature_dims,
        embedding_dim=config.get("embedding_dim", 64),
        num_heads=config.get("num_heads", 4),
    )
    return model
