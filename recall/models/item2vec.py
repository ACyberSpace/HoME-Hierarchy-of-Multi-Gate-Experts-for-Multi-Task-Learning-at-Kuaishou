from gensim.models import Word2Vec


class Item2Vec:
    def __init__(
        self,
        embedding_dim: int = 64,
        window: int = 5,
        sg: int = 1,
        epochs: int = 10,
        workers: int = 4,
        max_user_items: int = 50,
    ):
        self.embedding_dim = embedding_dim
        self.window = window
        self.sg = sg
        self.epochs = epochs
        self.workers = workers
        self.max_user_items = max_user_items
        self.model = None

    def fit(self, sequences):
        self.model = Word2Vec(
            sentences=sequences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=1,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
        )

    def get_item_embedding(self, item_id):
        if str(item_id) in self.model.wv:
            return self.model.wv[str(item_id)]
        return np.zeros(self.embedding_dim)

    def generate_recall_candidates(self, user_id, user_hist_items, all_item_ids, top_k=100):
        hist_tokens = [
            str(int(item_id))
            for item_id in user_hist_items[-self.max_user_items:]
            if int(item_id) != 0 and str(int(item_id)) in self.model.wv
        ]
        if not hist_tokens:
            return []

        hist_set = set(hist_tokens)
        raw_candidates = self.model.wv.most_similar(
            positive=hist_tokens,
            topn=top_k + len(hist_set),
        )
        candidates = [
            int(item_id)
            for item_id, _ in raw_candidates
            if item_id not in hist_set
        ]
        return candidates[:top_k]


def build_item2vec_model(config: dict, item_vocab_size: int):
    model = Item2Vec(
        embedding_dim=config.get("embedding_dim", 64),
        window=config.get("window", 5),
        sg=config.get("sg", 1),
        epochs=config.get("item2vec_epochs", config.get("epochs", 10)),
        workers=config.get("workers", 4),
        max_user_items=config.get("item2vec_max_user_items", 50),
    )
    return model
