from gensim.models import Word2Vec
import numpy as np


class Item2Vec:
    def __init__(self, embedding_dim: int = 64, window: int = 5, sg: int = 1):
        self.embedding_dim = embedding_dim
        self.window = window
        self.sg = sg
        self.model = None

    def fit(self, sequences):
        self.model = Word2Vec(
            sentences=sequences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=1,
            workers=4,
            sg=self.sg,
            epochs=10,
        )

    def get_item_embedding(self, item_id):
        if str(item_id) in self.model.wv:
            return self.model.wv[str(item_id)]
        return np.zeros(self.embedding_dim)

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        hist_embeddings = []
        for item_id in user_hist_items:
            if item_id != 0:
                hist_embeddings.append(self.get_item_embedding(item_id))

        if len(hist_embeddings) == 0:
            return []

        user_embedding = np.mean(hist_embeddings, axis=0)

        scores = []
        for item_id in all_item_ids:
            if item_id != 0:
                item_embedding = self.get_item_embedding(item_id)
                score = np.dot(user_embedding, item_embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding) + 1e-8
                )
                scores.append((item_id, score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [item_id for item_id, score in scores]


def build_item2vec_model(config: dict, item_vocab_size: int):
    model = Item2Vec(
        embedding_dim=config.get("embedding_dim", 64),
        window=config.get("window", 5),
        sg=config.get("sg", 1),
    )
    return model
