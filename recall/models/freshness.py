import pandas as pd


class FreshnessRecall:
    def __init__(self, video_info: pd.DataFrame):
        self.video_info = video_info
        self.fresh_items = []

    def fit(self):
        if "upload_timestamp" in self.video_info.columns and (self.video_info["upload_timestamp"] > 0).any():
            sorted_videos = self.video_info.sort_values(
                ["upload_timestamp", "video_id"], ascending=[False, True]
            )
            self.fresh_items = sorted_videos["video_id"].tolist()
        else:
            self.fresh_items = self.video_info["video_id"].tolist()

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        candidates = []
        hist_set = set(int(item_id) for item_id in user_hist_items)
        for item_id in self.fresh_items:
            item_id = int(item_id)
            if item_id in hist_set:
                continue
            candidates.append(item_id)
            if len(candidates) >= top_k:
                break
        return candidates


def build_freshness_model(config: dict, video_info: pd.DataFrame):
    model = FreshnessRecall(video_info=video_info)
    return model
