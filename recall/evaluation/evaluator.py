import numpy as np
import torch
from tqdm import tqdm

from recall.data.labels import build_recall_positive_mask


def evaluate_recall(recall_results, test_data, top_k_list=[10, 50, 100]):
    user_ids = test_data["user_id"]
    video_ids = test_data["video_id"]
    positive_mask = build_recall_positive_mask(test_data)

    user_pos_items = {}
    for user_id, video_id, is_positive in zip(user_ids, video_ids, positive_mask):
        if is_positive:
            if user_id not in user_pos_items:
                user_pos_items[user_id] = set()
            user_pos_items[user_id].add(int(video_id))

    metrics = {}
    for top_k in top_k_list:
        hit_count = 0
        total_count = 0
        user_recalls = []
        covered_items = set()

        for user_id in user_pos_items:
            if user_id in recall_results:
                candidates = [int(x) for x in recall_results[user_id][:top_k]]
                pos_items = user_pos_items[user_id]
                hits = len(set(candidates) & set(pos_items))
                hit_count += hits
                total_count += len(pos_items)
                user_recalls.append(hits / len(pos_items) if pos_items else 0.0)
                covered_items.update(candidates)
            else:
                total_count += len(user_pos_items[user_id])
                user_recalls.append(0.0)

        metrics[f"Recall@{top_k}"] = hit_count / total_count if total_count > 0 else 0
        metrics[f"MicroRecall@{top_k}"] = metrics[f"Recall@{top_k}"]
        metrics[f"UserAvgRecall@{top_k}"] = float(np.mean(user_recalls)) if user_recalls else 0.0
        metrics[f"ItemCoverage@{top_k}"] = len(covered_items)

    return metrics


def evaluate_recall_channels(channel_results, test_data, top_k=100):
    """Evaluate each recall channel and its overlap against the fused candidates."""
    channel_metrics = {}
    for channel, results in channel_results.items():
        channel_metrics[channel] = evaluate_recall(results, test_data, [top_k])
    return channel_metrics


def compute_channel_overlap(channel_results, top_k=100):
    """Return pairwise user-averaged overlap rates between recall channels."""
    channels = list(channel_results.keys())
    overlap = {}

    for i, left in enumerate(channels):
        for right in channels[i + 1:]:
            user_ids = set(channel_results[left]) | set(channel_results[right])
            rates = []
            for user_id in user_ids:
                left_items = set(channel_results[left].get(user_id, [])[:top_k])
                right_items = set(channel_results[right].get(user_id, [])[:top_k])
                denom = min(len(left_items), len(right_items))
                if denom == 0:
                    continue
                rates.append(len(left_items & right_items) / denom)
            overlap[f"{left}__{right}"] = float(np.mean(rates)) if rates else 0.0

    return overlap


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
