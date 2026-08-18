from collections import defaultdict

from gensim.models import Word2Vec
from tqdm import tqdm

from recall.data.feature_column import RECALL_ITEM_FEATURES
from recall.data.labels import build_recall_positive_mask


class EGESRecall:
    def __init__(
        self,
        embedding_dim=64,
        window=5,
        epochs=5,
        workers=4,
        max_user_items=50,
        min_count=1,
    ):
        self.embedding_dim = int(embedding_dim)
        self.window = int(window)
        self.epochs = int(epochs)
        self.workers = int(workers)
        self.max_user_items = int(max_user_items)
        self.min_count = int(min_count)
        self.model = None
        self.item_feature_index = {}

    @staticmethod
    def _item_token(item_id):
        return f"i_{int(item_id)}"

    @staticmethod
    def _side_token(feature_name, value):
        return f"{feature_name}_{int(value)}"

    def _build_item_feature_index(self, train_data):
        item_feature_index = {}
        for idx, item_id in enumerate(train_data["video_id"]):
            item_id = int(item_id)
            if item_id in item_feature_index:
                continue
            item_feature_index[item_id] = {
                feat_name: int(train_data[feat_name][idx])
                for feat_name in RECALL_ITEM_FEATURES
                if feat_name in train_data
            }
        return item_feature_index

    def _item_tokens_with_side_info(self, item_id):
        item_id = int(item_id)
        tokens = [self._item_token(item_id)]
        features = self.item_feature_index.get(item_id, {})
        for feat_name in ("author_id", "music_id", "tag", "video_type", "upload_type", "music_type"):
            value = int(features.get(feat_name, 0))
            if value > 0:
                tokens.append(self._side_token(feat_name, value))
        return tokens

    def fit(self, train_data):
        self.item_feature_index = self._build_item_feature_index(train_data)
        positive_mask = build_recall_positive_mask(train_data)
        user_items = defaultdict(list)

        for user_id, item_id, is_positive in zip(
            train_data["user_id"], train_data["video_id"], positive_mask
        ):
            if not is_positive:
                continue
            user_items[int(user_id)].append(int(item_id))

        sentences = []
        for items in tqdm(user_items.values(), desc="Build EGES walks"):
            if len(items) < 2:
                continue
            items = items[-self.max_user_items:]
            sentence = []
            for item_id in items:
                sentence.extend(self._item_tokens_with_side_info(item_id))
            if len(sentence) >= 2:
                sentences.append(sentence)

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,
            epochs=self.epochs,
        )

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        if self.model is None:
            return []

        positive_tokens = [
            self._item_token(item_id)
            for item_id in user_hist_items[-self.max_user_items:]
            if int(item_id) != 0 and self._item_token(item_id) in self.model.wv
        ]
        if not positive_tokens:
            return []

        hist_set = set(positive_tokens)
        candidate_scores = defaultdict(float)
        topn = min(len(self.model.wv), top_k * 20 + len(hist_set))
        for hist_rank, token in enumerate(reversed(positive_tokens), start=1):
            weight = 1.0 / hist_rank
            raw_candidates = self.model.wv.most_similar(positive=[token], topn=topn)
            for candidate_token, score in raw_candidates:
                if not candidate_token.startswith("i_") or candidate_token in hist_set:
                    continue
                candidate_scores[int(candidate_token[2:])] += float(score) * weight

        candidates = []
        for item_id, _ in sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0])):
            candidates.append(int(item_id))
            if len(candidates) >= top_k:
                break
        return candidates


def build_eges_model(config: dict):
    return EGESRecall(
        embedding_dim=config.get("embedding_dim", 64),
        window=config.get("eges_window", 5),
        epochs=config.get("eges_epochs", config.get("epochs", 5)),
        workers=config.get("workers", 4),
        max_user_items=config.get("eges_max_user_items", 50),
        min_count=config.get("eges_min_count", 1),
    )
