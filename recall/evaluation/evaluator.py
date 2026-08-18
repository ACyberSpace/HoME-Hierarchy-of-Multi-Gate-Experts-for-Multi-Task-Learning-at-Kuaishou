import numpy as np
import torch
from tqdm import tqdm

from recall.data.labels import build_recall_positive_mask
from recall.data.feature_column import RECALL_ITEM_FEATURES


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
        candidate_counts = []

        for user_id in user_pos_items:
            if user_id in recall_results:
                candidates = [int(x) for x in recall_results[user_id][:top_k]]
                candidate_counts.append(len(candidates))
                pos_items = user_pos_items[user_id]
                hits = len(set(candidates) & set(pos_items))
                hit_count += hits
                total_count += len(pos_items)
                user_recalls.append(hits / len(pos_items) if pos_items else 0.0)
                covered_items.update(candidates)
            else:
                total_count += len(user_pos_items[user_id])
                user_recalls.append(0.0)
                candidate_counts.append(0)

        metrics[f"Recall@{top_k}"] = hit_count / total_count if total_count > 0 else 0
        metrics[f"MicroRecall@{top_k}"] = metrics[f"Recall@{top_k}"]
        metrics[f"UserAvgRecall@{top_k}"] = float(np.mean(user_recalls)) if user_recalls else 0.0
        metrics[f"ItemCoverage@{top_k}"] = len(covered_items)
        metrics[f"AvgCandidates@{top_k}"] = float(np.mean(candidate_counts)) if candidate_counts else 0.0
        metrics[f"FullCandidateRate@{top_k}"] = (
            float(np.mean([count >= top_k for count in candidate_counts]))
            if candidate_counts else 0.0
        )

    return metrics


def evaluate_recall_channels(channel_results, test_data, top_k=100):
    """Evaluate each recall channel and its overlap against the fused candidates."""
    channel_metrics = {}
    for channel, results in channel_results.items():
        channel_metrics[channel] = evaluate_recall(results, test_data, [top_k])
    return channel_metrics


def compute_test_pool_overlap(train_item_ids, test_data):
    """Measure how many test positive targets are reachable from the train item pool."""
    train_item_set = set(int(item_id) for item_id in train_item_ids)
    test_item_ids = [int(item_id) for item_id in test_data["video_id"]]
    positive_mask = build_recall_positive_mask(test_data)

    positive_events = [
        (int(user_id), int(item_id))
        for user_id, item_id, is_positive in zip(
            test_data["user_id"], test_item_ids, positive_mask
        )
        if is_positive
    ]
    positive_items = {item_id for _, item_id in positive_events}
    positive_user_items = set(positive_events)

    overlap_events = sum(1 for _, item_id in positive_events if item_id in train_item_set)
    overlap_items = len(positive_items & train_item_set)
    overlap_user_items = sum(
        1 for _, item_id in positive_user_items if item_id in train_item_set
    )

    total_events = len(positive_events)
    total_items = len(positive_items)
    total_user_items = len(positive_user_items)

    def ratio(count, total):
        return count / total if total else 0.0

    return {
        "train_unique_items": len(train_item_set),
        "test_unique_items": len(set(test_item_ids)),
        "test_positive_events": total_events,
        "test_positive_unique_items": total_items,
        "test_positive_user_items": total_user_items,
        "overlap_positive_events": overlap_events,
        "non_overlap_positive_events": total_events - overlap_events,
        "overlap_positive_event_rate": ratio(overlap_events, total_events),
        "non_overlap_positive_event_rate": ratio(total_events - overlap_events, total_events),
        "overlap_positive_unique_items": overlap_items,
        "non_overlap_positive_unique_items": total_items - overlap_items,
        "overlap_positive_unique_item_rate": ratio(overlap_items, total_items),
        "non_overlap_positive_unique_item_rate": ratio(total_items - overlap_items, total_items),
        "overlap_positive_user_items": overlap_user_items,
        "non_overlap_positive_user_items": total_user_items - overlap_user_items,
        "overlap_positive_user_item_rate": ratio(overlap_user_items, total_user_items),
        "non_overlap_positive_user_item_rate": ratio(
            total_user_items - overlap_user_items, total_user_items
        ),
    }


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


def _build_item_feature_tensors(all_item_ids, item_feature_index, device):
    features = {"video_id": torch.tensor(all_item_ids, dtype=torch.long).to(device)}
    if not item_feature_index:
        return features

    for feat_name in RECALL_ITEM_FEATURES:
        if feat_name == "video_id":
            continue
        values = [
            int(item_feature_index.get(int(item_id), {}).get(feat_name, 0))
            for item_id in all_item_ids
        ]
        features[feat_name] = torch.tensor(values, dtype=torch.long).to(device)

    return features


def generate_recall_candidates_dssm(model, user_sequences, all_item_ids, item_feature_index=None, top_k=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    recall_results = {}
    user_ids = user_sequences["user_id"]

    item_features = _build_item_feature_tensors(all_item_ids, item_feature_index, device)
    with torch.no_grad():
        item_repr = model.get_item_repr(item_features)

    for i in tqdm(range(0, len(user_ids), batch_size), desc="生成召回候选"):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_short_seq = user_sequences["short_seq"][i:i+batch_size]
        batch_short_mask = user_sequences["short_mask"][i:i+batch_size]

        user_features = {
            "user_id": torch.tensor(batch_user_ids, dtype=torch.long).to(device),
            "short_seq": torch.tensor(batch_short_seq, dtype=torch.long).to(device),
            "short_mask": torch.tensor(batch_short_mask, dtype=torch.long).to(device),
        }

        with torch.no_grad():
            user_repr = model.get_user_repr(user_features)
            scores = torch.matmul(user_repr, item_repr.T)
            top_indices = torch.topk(scores, top_k, dim=1)[1]

        for j, user_id in enumerate(batch_user_ids):
            candidates = [all_item_ids[idx.item()] for idx in top_indices[j]]
            recall_results[user_id] = candidates

    return recall_results


def generate_recall_candidates_mind(model, user_sequences, all_item_ids, item_feature_index=None, top_k=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    recall_results = {}
    user_ids = user_sequences["user_id"]

    item_features = _build_item_feature_tensors(all_item_ids, item_feature_index, device)
    with torch.no_grad():
        item_repr = model.get_item_repr(item_features)

    for i in tqdm(range(0, len(user_ids), batch_size), desc="生成召回候选"):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_short_seq = user_sequences["short_seq"][i:i+batch_size]
        batch_short_mask = user_sequences["short_mask"][i:i+batch_size]

        user_features = {
            "user_id": torch.tensor(batch_user_ids, dtype=torch.long).to(device),
            "short_seq": torch.tensor(batch_short_seq, dtype=torch.long).to(device),
            "short_mask": torch.tensor(batch_short_mask, dtype=torch.long).to(device),
        }

        with torch.no_grad():
            interest_embeddings = model.get_user_interests(user_features)
            scores = torch.matmul(interest_embeddings, item_repr.T)
            max_scores, _ = torch.max(scores, dim=1)
            top_indices = torch.topk(max_scores, top_k, dim=1)[1]

        for j, user_id in enumerate(batch_user_ids):
            candidates = [all_item_ids[idx.item()] for idx in top_indices[j]]
            recall_results[user_id] = candidates

    return recall_results


def generate_recall_candidates_sdm(model, user_sequences, all_item_ids, item_feature_index=None, top_k=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    recall_results = {}
    user_ids = user_sequences["user_id"]

    item_features = _build_item_feature_tensors(all_item_ids, item_feature_index, device)
    with torch.no_grad():
        item_repr = model.get_item_repr(item_features)

    for i in tqdm(range(0, len(user_ids), batch_size), desc="生成召回候选"):
        batch_user_ids = user_ids[i:i+batch_size]
        batch_short_seq = user_sequences["short_seq"][i:i+batch_size]
        batch_short_mask = user_sequences["short_mask"][i:i+batch_size]
        batch_long_seq = user_sequences["long_seq"][i:i+batch_size]
        batch_long_mask = user_sequences["long_mask"][i:i+batch_size]

        user_features = {
            "user_id": torch.tensor(batch_user_ids, dtype=torch.long).to(device),
            "short_seq": torch.tensor(batch_short_seq, dtype=torch.long).to(device),
            "short_mask": torch.tensor(batch_short_mask, dtype=torch.long).to(device),
            "long_seq": torch.tensor(batch_long_seq, dtype=torch.long).to(device),
            "long_mask": torch.tensor(batch_long_mask, dtype=torch.long).to(device),
        }

        with torch.no_grad():
            user_repr = model.get_user_repr(user_features)
            scores = torch.matmul(user_repr, item_repr.T)
            top_indices = torch.topk(scores, top_k, dim=1)[1]

        for j, user_id in enumerate(batch_user_ids):
            candidates = [all_item_ids[idx.item()] for idx in top_indices[j]]
            recall_results[user_id] = candidates

    return recall_results
