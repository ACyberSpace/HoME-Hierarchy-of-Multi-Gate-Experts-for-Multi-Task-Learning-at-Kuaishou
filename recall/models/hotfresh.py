from collections import defaultdict

import numpy as np
import pandas as pd


class HotFreshRecall:
    def __init__(self, video_info: pd.DataFrame, weights=None, half_life_days=3.0):
        self.video_info = video_info
        self.weights = weights or {
            "is_click": 1.0,
            "long_view": 0.7,
            "is_like": 2.0,
            "is_comment": 3.0,
            "is_forward": 3.0,
            "is_follow": 3.0,
        }
        self.half_life_days = float(half_life_days)
        self.ranked_items = []

    def _build_upload_time(self):
        if self.video_info is None or "video_id" not in self.video_info.columns:
            return {}
        if "upload_timestamp" not in self.video_info.columns:
            return {}
        return {
            int(row.video_id): float(row.upload_timestamp)
            for row in self.video_info[["video_id", "upload_timestamp"]].itertuples(index=False)
            if not pd.isna(row.upload_timestamp) and float(row.upload_timestamp) > 0
        }

    def fit(self, train_data):
        popularity = defaultdict(float)
        video_ids = train_data["video_id"]
        for label, weight in self.weights.items():
            if label not in train_data:
                continue
            label_values = np.asarray(train_data[label])
            for idx in np.flatnonzero(label_values == 1):
                popularity[int(video_ids[idx])] += float(weight)

        if not popularity:
            for item_id in video_ids:
                popularity[int(item_id)] += 1.0

        upload_time = self._build_upload_time()
        latest_ts = max(upload_time.values()) if upload_time else 0.0
        half_life_seconds = max(self.half_life_days, 1e-6) * 86400.0

        scores = {}
        for item_id, pop_score in popularity.items():
            ts = upload_time.get(int(item_id), latest_ts)
            age_seconds = max(latest_ts - ts, 0.0) if latest_ts > 0 else 0.0
            freshness_weight = 0.5 ** (age_seconds / half_life_seconds)
            scores[int(item_id)] = float(pop_score) * float(freshness_weight)

        self.ranked_items = [
            item_id
            for item_id, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        ]

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        hist_set = set(int(item_id) for item_id in user_hist_items)
        candidates = []
        for item_id in self.ranked_items:
            if item_id in hist_set:
                continue
            candidates.append(int(item_id))
            if len(candidates) >= top_k:
                break
        return candidates


def build_hotfresh_model(config: dict, video_info: pd.DataFrame):
    return HotFreshRecall(
        video_info=video_info,
        weights=config.get("hotfresh_weights"),
        half_life_days=config.get("hotfresh_half_life_days", 3.0),
    )
