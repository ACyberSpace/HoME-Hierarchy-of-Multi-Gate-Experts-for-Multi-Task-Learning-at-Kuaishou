import numpy as np
import pandas as pd


class FreshnessRecall:
    def __init__(self, video_info: pd.DataFrame):
        self.video_info = video_info
        self.fresh_items = []

    def fit(self):
        if "upload_timestamp" in self.video_info.columns:
            sorted_videos = self.video_info.sort_values("upload_timestamp", ascending=False)
            self.fresh_items = sorted_videos["video_id"].tolist()
        else:
            self.fresh_items = self.video_info["video_id"].tolist()

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        candidates = []
        for item_id in self.fresh_items:
            if item_id not in user_hist_items:
                candidates.append(item_id)
            if len(candidates) >= top_k:
                break
        return candidates


def build_freshness_model(config: dict, video_info: pd.DataFrame):
    model = FreshnessRecall(video_info=video_info)
    return model
