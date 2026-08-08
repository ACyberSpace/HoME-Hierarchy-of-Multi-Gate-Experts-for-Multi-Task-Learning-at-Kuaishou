import numpy as np
import torch
from tqdm import tqdm


def evaluate_recall(recall_results, test_data, top_k_list=[10, 50, 100]):
    user_ids = test_data["user_id"]
    video_ids = test_data["video_id"]
    is_click = test_data["is_click"]

    user_pos_items = {}
    for user_id, video_id, click in zip(user_ids, video_ids, is_click):
        if click == 1:
            if user_id not in user_pos_items:
                user_pos_items[user_id] = []
            user_pos_items[user_id].append(video_id)

    metrics = {}
    for top_k in top_k_list:
        hit_count = 0
        total_count = 0
        for user_id in user_pos_items:
            if user_id in recall_results:
                candidates = recall_results[user_id][:top_k]
                pos_items = user_pos_items[user_id]
                hits = len(set(candidates) & set(pos_items))
                hit_count += hits
                total_count += len(pos_items)
        metrics[f"Recall@{top_k}"] = hit_count / total_count if total_count > 0 else 0

    return metrics


def generate_recall_candidates_dssm(model, user_sequences, all_item_ids, top_k=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    recall_results = {}
    user_ids = user_sequences["user_id"]

    all_item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long).to(device)

    for i in tqdm(range(0, len(user_ids), batch_size), desc="生成召回候选"):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_short_seq = user_sequences["short_seq"][i:i+batch_size]
        batch_short_mask = user_sequences["short_mask"][i:i+batch_size]

        user_features = {
            "short_seq": torch.tensor(batch_short_seq, dtype=torch.long).to(device),
            "short_mask": torch.tensor(batch_short_mask, dtype=torch.long).to(device),
        }

        user_repr = model.get_user_repr(user_features)
        item_repr = model.get_item_repr({"video_id": all_item_ids_tensor})

        scores = torch.matmul(user_repr, item_repr.T)
        top_indices = torch.topk(scores, top_k, dim=1)[1]

        for j, user_id in enumerate(batch_user_ids):
            candidates = [all_item_ids[idx.item()] for idx in top_indices[j]]
            recall_results[user_id] = candidates

    return recall_results


def generate_recall_candidates_mind(model, user_sequences, all_item_ids, top_k=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    recall_results = {}
    user_ids = user_sequences["user_id"]

    all_item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long).to(device)

    for i in tqdm(range(0, len(user_ids), batch_size), desc="生成召回候选"):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_short_seq = user_sequences["short_seq"][i:i+batch_size]
        batch_short_mask = user_sequences["short_mask"][i:i+batch_size]

        user_features = {
            "short_seq": torch.tensor(batch_short_seq, dtype=torch.long).to(device),
            "short_mask": torch.tensor(batch_short_mask, dtype=torch.long).to(device),
        }

        interest_embeddings = model.get_user_interests(user_features)
        item_embedding = model.item_embedding(all_item_ids_tensor)

        scores = torch.matmul(interest_embeddings, item_embedding.T)
        max_scores, _ = torch.max(scores, dim=1)
        top_indices = torch.topk(max_scores, top_k, dim=1)[1]

        for j, user_id in enumerate(batch_user_ids):
            candidates = [all_item_ids[idx.item()] for idx in top_indices[j]]
            recall_results[user_id] = candidates

    return recall_results


def generate_recall_candidates_sdm(model, user_sequences, all_item_ids, top_k=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    recall_results = {}
    user_ids = user_sequences["user_id"]

    all_item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long).to(device)

    for i in tqdm(range(0, len(user_ids), batch_size), desc="生成召回候选"):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_short_seq = user_sequences["short_seq"][i:i+batch_size]
        batch_short_mask = user_sequences["short_mask"][i:i+batch_size]
        batch_long_seq = user_sequences["long_seq"][i:i+batch_size]
        batch_long_mask = user_sequences["long_mask"][i:i+batch_size]

        user_features = {
            "short_seq": torch.tensor(batch_short_seq, dtype=torch.long).to(device),
            "short_mask": torch.tensor(batch_short_mask, dtype=torch.long).to(device),
            "long_seq": torch.tensor(batch_long_seq, dtype=torch.long).to(device),
            "long_mask": torch.tensor(batch_long_mask, dtype=torch.long).to(device),
        }

        user_repr = model.get_user_repr(user_features)
        item_embedding = model.item_embedding(all_item_ids_tensor)

        scores = torch.matmul(user_repr, item_embedding.T)
        top_indices = torch.topk(scores, top_k, dim=1)[1]

        for j, user_id in enumerate(batch_user_ids):
            candidates = [all_item_ids[idx.item()] for idx in top_indices[j]]
            recall_results[user_id] = candidates

    return recall_results
