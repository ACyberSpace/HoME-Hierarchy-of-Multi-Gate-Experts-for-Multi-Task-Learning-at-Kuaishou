from collections import defaultdict

import numpy as np


class PopularityRecall:
    def __init__(self, weights=None):
        self.weights = weights or {
            "is_click": 1.0,
            "long_view": 0.7,
            "is_like": 2.0,
            "is_comment": 3.0,
            "is_forward": 3.0,
            "is_follow": 3.0,
        }
        self.popular_items = []

    def fit(self, train_data):
        scores = defaultdict(float)
        video_ids = train_data["video_id"]

        for label, weight in self.weights.items():
            if label not in train_data:
                continue
            label_values = np.asarray(train_data[label])
            positive_indices = np.flatnonzero(label_values == 1)
            for idx in positive_indices:
                scores[int(video_ids[idx])] += float(weight)

        if not scores:
            counts = defaultdict(float)
            for item_id in video_ids:
                counts[int(item_id)] += 1.0
            scores = counts

        self.popular_items = [
            item_id
            for item_id, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        ]

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        hist_set = set(int(item_id) for item_id in user_hist_items)
        candidates = []

        for item_id in self.popular_items:
            if item_id in hist_set:
                continue
            candidates.append(item_id)
            if len(candidates) >= top_k:
                break

        return candidates


def build_popularity_model(config: dict):
    return PopularityRecall(weights=config.get("popularity_weights"))
