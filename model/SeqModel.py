import torch
import torch.nn as nn
import torch.nn.functional as F


class UserAttention(nn.Module):
    """
    用户个性化注意力层（候选感知）

    用user_embedding作为query，对序列做加权聚合
    公式: attention = softmax(user_emb · seq.T) · seq
    """

    def __init__(self, input_dim, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim, bias=False)

    def forward(
            self,
            user_emb: torch.Tensor,  # [B, embed_dim]
            item_seq: torch.Tensor,  # [B, seq_len, embed_dim]
            mask: torch.Tensor = None  # [B, seq_len]
    ) -> torch.Tensor:
        """
        Returns:
            weighted_seq: [B, embed_dim] 加权聚合后的序列表示
        """
        # 计算注意力分数: [B, embed_dim] · [B, embed_dim, seq_len] -> [B, seq_len]
        attn_scores = torch.bmm(
            self.proj(user_emb).unsqueeze(1),  # [B, 1, embed_dim]
            item_seq.transpose(1, 2)  # [B, embed_dim, seq_len]
        ).squeeze(1)  # [B, seq_len]

        # 应用mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, seq_len]

        # 加权聚合
        weighted_seq = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, seq_len]
            item_seq  # [B, seq_len, embed_dim]
        ).squeeze(1)  # [B, embed_dim]

        return weighted_seq


class GatedFusion(nn.Module):
    """
    门控融合层（更正版）

    融合方式: output = (1 - gate) * long_term + gate * short_term
    门控计算: gate = sigmoid(user·W1 + short·W2 + long·W3 + b)
    """

    def __init__(self, user_dim, embed_dim: int):
        super().__init__()
        self.W1 = nn.Linear(user_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(embed_dim))

        # 初始化
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)

    def forward(
            self,
            user_emb: torch.Tensor,  # [B, embed_dim]
            short_term: torch.Tensor,  # [B, embed_dim]
            long_term: torch.Tensor  # [B, embed_dim]
    ) -> torch.Tensor:
        """
        Returns:
            fused: [B, embed_dim]
        """
        # 计算门控: sigmoid(W1·user + W2·short + W3·long + b)
        gate = torch.sigmoid(
            self.W1(user_emb) +
            self.W2(short_term) +
            self.W3(long_term) +
            self.b
        )  # [B, embed_dim]

        # 融合: (1-gate)·long + gate·short
        output = (1 - gate) * long_term + gate * short_term

        return output


class SDMInterestNetwork(nn.Module):
    """
    SDM长短期兴趣网络

    短期兴趣: LSTM + MultiHeadAttention + UserAttention
    长期兴趣: 多维度UserAttention聚合 + Dense
    融合: GatedFusion
    """

    def __init__(
            self,
            input_dim: int,
            embed_dim: int = 64,
            num_heads: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ========== 短期兴趣建模 ==========
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if dropout > 0 else 0
        )

        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm
        self.lstm_ln = nn.LayerNorm(embed_dim)
        self.attn_ln = nn.LayerNorm(embed_dim)

        # 短期兴趣注意力
        self.short_term_attention = UserAttention(input_dim, embed_dim)

        # ========== 长期兴趣建模 ==========
        # 长期兴趣使用单层注意力（简化版）
        self.long_term_attention = UserAttention(input_dim, embed_dim)

        # ========== 融合层 ==========
        self.gated_fusion = GatedFusion(input_dim, embed_dim)

        # ========== 辅助损失（可选） ==========
        # SDM原始使用sampled softmax，这里不需要，因为我们直接做CTR
        self.use_aux_loss = False

    # def forward(
    #         self,
    #         short_seq_emb: torch.Tensor,  # [B, short_seq_len, embed_dim]
    #         short_seq_mask: torch.Tensor,  # [B, short_seq_len]
    #         long_seq_emb: torch.Tensor,  # [B, long_seq_len, embed_dim]
    #         long_seq_mask: torch.Tensor,  # [B, long_seq_len]
    #         user_emb: torch.Tensor = None  # [B, embed_dim] 可选
    # ) -> dict:
    #     """
    #     Args:
    #         short_seq_emb: 短期序列嵌入 [B, short_len, embed_dim]
    #         short_seq_mask: 短期序列mask [B, short_len]
    #         long_seq_emb: 长期序列嵌入 [B, long_len, embed_dim]
    #         long_seq_mask: 长期序列mask [B, long_len]
    #         user_emb: 用户表征嵌入 [B, embed_dim]，如果为None则用序列均值
    #
    #     Returns:
    #         {
    #             'short_interest': [B, embed_dim] 短期兴趣,
    #             'long_interest': [B, embed_dim] 长期兴趣,
    #             'fused_interest': [B, embed_dim] 融合兴趣,
    #             'user_emb': [B, embed_dim] 用户表征
    #         }
    #     """
    #     B, short_len, D = short_seq_emb.shape
    #
    #     # 检测全零序列，替换为极小值，避免LSTM/Attention产生NaN
    #     eps = 1e-8
    #
    #     short_zero_mask = (short_seq_emb.abs().sum(dim=-1) == 0)  # [B, short_len]
    #     if short_zero_mask.any():
    #         short_seq_emb = short_seq_emb + eps  # 全零位置变为eps
    #
    #     long_zero_mask = (long_seq_emb.abs().sum(dim=-1) == 0)  # [B, short_len]
    #     if long_zero_mask.any():
    #         long_seq_emb = long_seq_emb + eps  # 全零位置变为eps
    #
    #     # 如果没有提供user_emb，用序列均值
    #     if user_emb is None:
    #         # 用短期序列的均值作为user_emb（简化处理）
    #         mask_expanded = short_seq_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
    #         user_emb = (short_seq_emb * mask_expanded).sum(dim=1) / (
    #                 mask_expanded.sum(dim=1) + 1e-8
    #         )
    #
    #     # ========== 短期兴趣建模 ==========
    #     # Step 1: LSTM处理序列
    #     lstm_out, _ = self.lstm(short_seq_emb)  # [B, short_len, embed_dim]
    #     lstm_out = self.lstm_ln(lstm_out)
    #
    #     # Step 2: 多头自注意力（需要padding mask）
    #     # 创建attention mask: True表示需要mask的位置
    #     attn_mask = (short_seq_mask == 0)  # [B, short_len]
    #
    #     attn_out, _ = self.multihead_attn(
    #         lstm_out, lstm_out, lstm_out,
    #         key_padding_mask=attn_mask
    #     )  # [B, short_len, embed_dim]
    #     attn_out = self.attn_ln(attn_out)
    #
    #     # Step 3: UserAttention聚合（候选感知）
    #     short_interest = self.short_term_attention(
    #         user_emb, attn_out, short_seq_mask
    #     )  # [B, embed_dim]
    #
    #     # ========== 长期兴趣建模 ==========
    #     long_interest = self.long_term_attention(
    #         user_emb, long_seq_emb, long_seq_mask
    #     )  # [B, embed_dim]
    #
    #     # ========== 融合 ==========
    #     fused_interest = self.gated_fusion(user_emb, short_interest, long_interest)
    #
    #     return {
    #         'short_interest': short_interest,
    #         'long_interest': long_interest,
    #         'fused_interest': fused_interest,
    #         'user_emb': user_emb
    #     }
    def forward(self, short_seq_emb, short_seq_mask, long_seq_emb, long_seq_mask, user_emb=None):
        B = short_seq_emb.size(0)
        device = short_seq_emb.device

        # ========== 检测哪些用户有有效历史 ==========
        short_valid = short_seq_mask.sum(dim=-1) > 0  # [B]
        long_valid = long_seq_mask.sum(dim=-1) > 0  # [B]

        # 短期兴趣
        short_interest = torch.zeros(B, self.embed_dim, device=device)
        if short_valid.any():
            valid_idx = short_valid.nonzero(as_tuple=True)[0]
            valid_short_emb = short_seq_emb[valid_idx]
            valid_short_mask = short_seq_mask[valid_idx]
            valid_user_emb = user_emb[valid_idx] if user_emb is not None else None

            lstm_out, _ = self.lstm(valid_short_emb)
            lstm_out = self.lstm_ln(lstm_out)

            attn_mask = (valid_short_mask == 0)
            attn_out, _ = self.multihead_attn(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=attn_mask
            )
            attn_out = self.attn_ln(attn_out)

            if valid_user_emb is None:
                mask_exp = valid_short_mask.unsqueeze(-1).float()
                valid_user_emb = (valid_short_emb * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)

            short_interest[valid_idx] = self.short_term_attention(
                valid_user_emb, attn_out, valid_short_mask
            )

        # 长期兴趣
        long_interest = torch.zeros(B, self.embed_dim, device=device)
        if long_valid.any():
            valid_idx = long_valid.nonzero(as_tuple=True)[0]
            valid_long_emb = long_seq_emb[valid_idx]
            valid_long_mask = long_seq_mask[valid_idx]
            valid_user_emb = user_emb[valid_idx] if user_emb is not None else None

            if valid_user_emb is None:
                mask_exp = valid_long_mask.unsqueeze(-1).float()
                valid_user_emb = (valid_long_emb * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)

            long_interest[valid_idx] = self.long_term_attention(
                valid_user_emb, valid_long_emb, valid_long_mask
            )

        # 融合
        fused_interest = self.gated_fusion(
            user_emb if user_emb is not None else short_interest,
            short_interest,
            long_interest
        )

        return {
            'short_interest': short_interest,
            'long_interest': long_interest,
            'fused_interest': fused_interest,
            'user_emb': user_emb if user_emb is not None else short_interest
        }


"""
DIN (Deep Interest Network) - PyTorch Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceActivation(nn.Module):
    """Dice激活函数：自适应的PReLU变体"""

    def __init__(self, num_features, eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x_reshaped = x.view(-1, features)
            norm_x = self.bn(x_reshaped)
            norm_x = norm_x.view(batch_size, seq_len, features)
        else:
            norm_x = self.bn(x)

        p = torch.sigmoid(norm_x)
        return p * x + (1 - p) * self.alpha * x


class FeedForwardLayer(nn.Module):
    """前馈神经网络层，支持Dice激活"""

    def __init__(self, hidden_units, activation='dice'):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if activation == 'dice':
                self.layers.append(DiceActivation(hidden_units[i + 1]))
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'prelu':
                self.layers.append(nn.PReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DinAttentionLayer(nn.Module):
    """
    DIN Attention Layer - PyTorch版本

    Reference: Deep Interest Network for Click-Through Rate Prediction
    """

    def __init__(
            self,
            ffn_hidden_units=[80, 40],
            ffn_activation='dice',
            query_ffn=False,
            query_activation='prelu'
    ):
        super().__init__()
        self.query_ffn = query_ffn
        self.query_activation = query_activation
        self.query_ffn_layer = None

        self.ffn_input_dim = None
        self.ffn_hidden_units = ffn_hidden_units
        self.ffn_activation = ffn_activation
        self.ffn_layer = None
        self.dense = None

    def _build(self, embed_dim, device):
        """延迟构建，根据输入维度初始化网络"""
        if self.ffn_layer is not None:
            return

        self.ffn_input_dim = embed_dim * 4
        ffn_units = [self.ffn_input_dim] + self.ffn_hidden_units
        self.ffn_layer = FeedForwardLayer(ffn_units, self.ffn_activation)
        self.ffn_layer.to(device)
        self.dense = nn.Linear(self.ffn_hidden_units[-1], 1)
        self.dense.to(device)

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: (batch_size, embed_dim) - 目标/候选物品embedding
            keys: (batch_size, seq_len, embed_dim) - 历史序列embeddings
            mask: (batch_size, seq_len) - 有效位置为1，无效为0

        Returns:
            output: (batch_size, embed_dim) - 注意力加权后的用户兴趣
        """
        batch_size, seq_len, embed_dim = keys.shape

        # 延迟构建网络
        if self.ffn_layer is None:
            self._build(embed_dim, device=keys.device)

        # 可选：对query进行前馈变换
        if self.query_ffn and self.query_ffn_layer is not None:
            query = self.query_ffn_layer(query)

        # 扩展query维度
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)

        # 构建注意力输入: [query, keys, query-keys, query*keys]
        att_inputs = torch.cat([
            query_expanded,
            keys,
            query_expanded - keys,
            query_expanded * keys
        ], dim=-1)

        # 前馈网络处理
        hidden_layer = self.ffn_layer(att_inputs)

        # 计算注意力分数
        scores = self.dense(hidden_layer).squeeze(-1)

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 缩放 + Softmax
        scores = scores / (embed_dim ** 0.5)
        scores = F.softmax(scores, dim=-1)

        # 加权求和
        scores_expanded = scores.unsqueeze(1)
        att_outputs = torch.bmm(scores_expanded, keys)

        return att_outputs.squeeze(1)


class DIN(nn.Module):
    """
    Deep Interest Network模型

    输入：target_item_id + 历史item_id序列
    输出：用户兴趣表征
    """

    def __init__(
            self,
            embed_dim=64,
            attention_hidden_units=[80, 40],
            attention_activation='dice',
            padding_idx=0
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # DIN注意力层
        self.attention = DinAttentionLayer(
            ffn_hidden_units=attention_hidden_units,
            ffn_activation=attention_activation
        )

    def forward(self, target_item, history_item, history_mask=None):
        """
        Args:
            target_item_ids: (batch_size,) - 目标/候选物品ID
            history_item_ids: (batch_size, seq_len) - 历史物品ID序列
            history_mask: (batch_size, seq_len) - 有效位置为1

        Returns:
            user_interest: (batch_size, embed_dim) - 用户兴趣表征
        """
        # Embedding
        query = target_item
        keys = history_item

        # DIN注意力聚合
        user_interest = self.attention(query, keys, history_mask)

        return user_interest

    def get_output_dim(self):
        return self.embed_dim