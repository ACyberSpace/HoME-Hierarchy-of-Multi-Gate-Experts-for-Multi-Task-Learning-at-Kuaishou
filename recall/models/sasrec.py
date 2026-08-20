import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        embedding_dim: int = 64,
        max_seq_len: int = 200,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.item_vocab_size = int(item_vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)

        self.item_embedding = nn.Embedding(
            self.item_vocab_size, self.embedding_dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=self.embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def get_user_repr(self, user_features):
        seq = user_features["long_seq"]
        mask = user_features["long_mask"].bool()
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        seq_emb = self.item_embedding(seq) + self.position_embedding(positions)
        seq_emb = self.dropout(seq_emb)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(
            seq_emb,
            mask=causal_mask,
            src_key_padding_mask=~mask,
        )
        encoded = self.layer_norm(encoded)

        seq_lens = mask.long().sum(dim=1).clamp_min(1)
        last_indices = (seq_lens - 1).view(-1, 1, 1).expand(-1, 1, self.embedding_dim)
        return encoded.gather(dim=1, index=last_indices).squeeze(1)

    def get_item_repr(self, item_features):
        return self.item_embedding(item_features["video_id"])

    def forward(self, user_features, item_features):
        user_repr = self.get_user_repr(user_features)
        item_repr = self.get_item_repr(item_features)
        return F.cosine_similarity(user_repr, item_repr)


def build_sasrec_model(config: dict, item_feature_dims: dict):
    return SASRec(
        item_vocab_size=item_feature_dims.get("video_id", 1),
        embedding_dim=config.get("embedding_dim", 64),
        max_seq_len=config.get("sasrec_max_seq_len", 200),
        num_heads=config.get("sasrec_num_heads", 2),
        num_layers=config.get("sasrec_num_layers", 2),
        dropout=config.get("sasrec_dropout", 0.2),
    )
