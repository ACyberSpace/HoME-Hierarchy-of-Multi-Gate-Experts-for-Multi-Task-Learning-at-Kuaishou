from collections import defaultdict
from itertools import combinations
from tqdm import tqdm


class Swing:
    def __init__(
        self,
        item_vocab_size: int,
        alpha1: float = 5.0,
        alpha2: float = 1.0,
        beta: float = 0.3,
        min_user_items: int = 2,
        max_user_items: int = 600,
        max_user_per_item: int = 700,
        max_pair_users: int = 200,
        max_sim_items: int = 200,
    ):
        self.item_vocab_size = item_vocab_size
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.min_user_items = min_user_items
        self.max_user_items = max_user_items
        self.max_user_per_item = max_user_per_item
        self.max_pair_users = max_pair_users
        self.max_sim_items = max_sim_items
        self.item_similarity = {}

    def fit(self, user_item_pairs):
        user_item_dict = defaultdict(list)
        for user_id, item_id, _ in user_item_pairs:
            user_item_dict[int(user_id)].append(int(item_id))

        user_items = {}
        item_users = defaultdict(list)
        for user_id, items in tqdm(user_item_dict.items(), desc="构建 Swing 用户序列"):
            dedup_items = list(dict.fromkeys(items))[-self.max_user_items:]
            if len(dedup_items) < self.min_user_items:
                continue
            user_items[user_id] = set(dedup_items)
            for item_id in dedup_items:
                if len(item_users[item_id]) < self.max_user_per_item:
                    item_users[item_id].append(user_id)

        valid_item_users = {
            item_id: set(users)
            for item_id, users in item_users.items()
            if users
        }
        user_weight = {
            user_id: 1.0 / ((len(items) + self.alpha1) ** self.beta)
            for user_id, items in user_items.items()
        }

        pair_users = defaultdict(list)
        for user_id, items in tqdm(user_items.items(), desc="收集 Swing 共同用户"):
            ordered_items = list(items)
            for left_idx, left_item in enumerate(ordered_items):
                for right_item in ordered_items[left_idx + 1:]:
                    if left_item == right_item:
                        continue
                    if user_id not in valid_item_users.get(left_item, set()):
                        continue
                    if user_id not in valid_item_users.get(right_item, set()):
                        continue
                    key = (left_item, right_item) if left_item < right_item else (right_item, left_item)
                    if len(pair_users[key]) < self.max_pair_users:
                        pair_users[key].append(user_id)

        similarity = defaultdict(lambda: defaultdict(float))
        for (left_item, right_item), users in tqdm(pair_users.items(), desc="计算 Swing 用户对惩罚"):
            if len(users) < 2:
                continue
            score = 0.0
            for left_user, right_user in combinations(users, 2):
                common_items = user_items[left_user] & user_items[right_user]
                score += (
                    user_weight[left_user]
                    * user_weight[right_user]
                    / (len(common_items) + self.alpha2)
                )
            if score <= 0:
                continue
            similarity[left_item][right_item] += score
            similarity[right_item][left_item] += score

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
        alpha1=config.get("alpha1", config.get("alpha", 5.0)),
        alpha2=config.get("alpha2", 1.0),
        beta=config.get("beta", 0.3),
        min_user_items=config.get("min_user_items", 2),
        max_user_items=config.get("max_user_items", 600),
        max_user_per_item=config.get("max_user_per_item", 700),
        max_pair_users=config.get("max_pair_users", 200),
        max_sim_items=config.get("max_sim_items", 200),
    )
    return model
