from typing import Dict, Iterable

import numpy as np


RECALL_POSITIVE_LABELS = (
    "is_click",
    "long_view",
    "is_like",
    "is_comment",
    "is_forward",
    "is_follow",
)


def build_recall_positive_mask(
    data: Dict,
    positive_labels: Iterable[str] = RECALL_POSITIVE_LABELS,
    negative_label: str = "is_hate",
) -> np.ndarray:
    """Return items with any positive feedback, excluding explicit hate feedback."""
    sample_count = len(data["user_id"])
    mask = np.zeros(sample_count, dtype=bool)

    for label in positive_labels:
        if label in data:
            mask |= np.asarray(data[label]) == 1

    if negative_label in data:
        mask &= np.asarray(data[negative_label]) != 1

    return mask


def build_user_positive_items(data: Dict) -> Dict[int, set]:
    positive_mask = build_recall_positive_mask(data)
    user_positive_items = {}

    for user_id, video_id, is_positive in zip(
        data["user_id"], data["video_id"], positive_mask
    ):
        if not is_positive:
            continue
        user_positive_items.setdefault(int(user_id), set()).add(int(video_id))

    return user_positive_items
