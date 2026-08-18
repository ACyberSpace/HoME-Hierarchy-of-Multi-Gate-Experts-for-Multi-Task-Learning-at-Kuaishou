from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FeatureColumn:
    name: str
    emb_dim: int = 16
    vocab_size: int = 1
    group: List[str] = field(default_factory=list)
    feature_type: str = "sparse"
    max_len: int = 1
    combiner: str = "mean"
    l2_reg: float = 0.0
    dtype: str = "int32"


FEATURE_GROUPS = {
    "user": [
        "user_id",
        "user_active_degree",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat3",
        "onehot_feat4",
        "onehot_feat5",
        "onehot_feat6",
        "onehot_feat7",
        "onehot_feat8",
        "onehot_feat9",
        "onehot_feat10",
        "onehot_feat11",
        "onehot_feat12",
        "onehot_feat13",
        "onehot_feat14",
        "onehot_feat15",
        "onehot_feat16",
        "onehot_feat17",
    ],
    "item": [
        "video_id",
        "author_id",
        "music_id",
        "video_type",
        "tag",
        "tab",
    ],
    "raw_hist_seq": [
        "short_seq",
        "short_mask",
        "long_seq",
        "long_mask",
    ],
    "target_item": [
        "video_id",
        "author_id",
        "music_id",
    ],
}


RECALL_USER_FEATURES = [
    "user_id",
    "user_active_degree",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
]

RECALL_ITEM_FEATURES = [
    "video_id",
    "author_id",
    "music_id",
    "tag",
    "video_type",
    "upload_type",
    "visible_status",
    "music_type",
]

RECALL_SEQ_FEATURES = [
    "short_seq",
    "short_mask",
    "long_seq",
    "long_mask",
]
