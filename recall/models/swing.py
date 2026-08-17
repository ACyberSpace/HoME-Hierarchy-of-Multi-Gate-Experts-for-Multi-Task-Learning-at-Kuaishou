import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm


class Swing:
    def __init__(self, item_vocab_size: int, alpha: float = 0.5, beta: float = 0.5):
        self.item_vocab_size = item_vocab_size
        self.alpha = alpha
        self.beta = beta
        self.item_similarity = None

    def fit(self, user_item_pairs):
        user_item_dict = defaultdict(list)
        for user_id, item_id, _ in user_item_pairs:
            user_item_dict[user_id].append(item_id)

        all_items = []
        all_users = []
        for user_id, items in user_item_dict.items():
            for item_id in items:
                all_users.append(user_id)
                all_items.append(item_id)

        unique_users = sorted(set(all_users))
        unique_items = sorted(set(all_items))

        user_to_idx = {u: i for i, u in enumerate(unique_users)}
        item_to_idx = {item: i for i, item in enumerate(unique_items)}

        row = [user_to_idx[u] for u in all_users]
        col = [item_to_idx[item] for item in all_items]
        data = [1] * len(row)

        user_item_matrix = csr_matrix((data, (row, col)), shape=(len(unique_users), len(unique_items)))
        user_item_matrix = normalize(user_item_matrix, norm='l2', axis=1)

        item_user_matrix = user_item_matrix.T
        similarity_matrix = item_user_matrix @ user_item_matrix

        user_degree = np.array(user_item_matrix.sum(axis=1)).flatten()
        user_weight = 1 / (self.alpha + user_degree)

        for i in tqdm(range(len(unique_items)), desc="计算 Swing 物品相似度"):
            for j in range(len(unique_items)):
                if i != j:
                    users_i = user_item_matrix[:, i].nonzero()[0]
                    users_j = user_item_matrix[:, j].nonzero()[0]
                    common_users = set(users_i) & set(users_j)
                    weight_sum = np.sum(user_weight[list(common_users)])
                    similarity_matrix[i, j] *= weight_sum

        self.item_similarity = similarity_matrix
        self.item_to_idx = item_to_idx
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        candidates = []
        for item_id in user_hist_items:
            if item_id in self.item_to_idx:
                idx = self.item_to_idx[item_id]
                sim_scores = self.item_similarity[idx].toarray().flatten()
                top_indices = np.argsort(sim_scores)[::-1][:top_k]
                for idx_ in top_indices:
                    if idx_ in self.idx_to_item:
                        candidates.append((self.idx_to_item[idx_], sim_scores[idx_]))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        return [item_id for item_id, score in candidates]


def build_swing_model(config: dict, item_vocab_size: int):
    model = Swing(item_vocab_size=item_vocab_size)
    return model
