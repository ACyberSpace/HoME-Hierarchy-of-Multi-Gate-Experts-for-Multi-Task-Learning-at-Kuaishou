from collections import defaultdict
from tqdm import tqdm


class Swing:
    def __init__(
        self,
        item_vocab_size: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        max_user_items: int = 200,
        max_sim_items: int = 500,
    ):
        self.item_vocab_size = item_vocab_size
        self.alpha = alpha
        self.beta = beta
        self.max_user_items = max_user_items
        self.max_sim_items = max_sim_items
        self.item_similarity = {}

    def fit(self, user_item_pairs):
        user_item_dict = defaultdict(list)
        for user_id, item_id, _ in user_item_pairs:
            user_item_dict[int(user_id)].append(int(item_id))

        similarity = defaultdict(lambda: defaultdict(float))
        for _, items in tqdm(user_item_dict.items(), desc="计算 Swing 共现相似度"):
            dedup_items = list(dict.fromkeys(items))[-self.max_user_items:]
            degree = len(dedup_items)
            if degree < 2:
                continue

            weight = 1.0 / (self.alpha + degree)
            for left_idx, left_item in enumerate(dedup_items):
                for right_item in dedup_items[left_idx + 1:]:
                    if left_item == right_item:
                        continue
                    similarity[left_item][right_item] += weight
                    similarity[right_item][left_item] += weight

        self.item_similarity = {}
        for item_id, neighbors in tqdm(similarity.items(), desc="裁剪 Swing top 相似物品"):
            ranked = sorted(neighbors.items(), key=lambda x: (-x[1], x[0]))
            self.item_similarity[item_id] = ranked[:self.max_sim_items]

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        candidate_scores = defaultdict(float)
        hist_set = set(int(item_id) for item_id in user_hist_items)

        for item_id in user_hist_items:
            for candidate_id, score in self.item_similarity.get(int(item_id), []):
                if candidate_id in hist_set:
                    continue
                candidate_scores[candidate_id] += score

        candidates = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [item_id for item_id, score in candidates]


def build_swing_model(config: dict, item_vocab_size: int):
    model = Swing(
        item_vocab_size=item_vocab_size,
        alpha=config.get("alpha", 0.5),
        beta=config.get("beta", 0.5),
        max_user_items=config.get("max_user_items", 200),
        max_sim_items=config.get("max_sim_items", 500),
    )
    return model
