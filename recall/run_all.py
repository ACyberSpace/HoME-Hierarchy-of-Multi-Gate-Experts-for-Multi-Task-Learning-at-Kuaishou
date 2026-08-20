import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import FullPipeline
from recall.evaluation.evaluator import (
    compute_channel_overlap,
    compute_test_pool_overlap,
    evaluate_recall,
    evaluate_recall_channels,
)
from recall.manager import RecallManager


ALL_CHANNELS = [
    "hotfresh",
    "eges",
    "youtubednn",
    "sasrec",
    "sdm",
    "dssm",
    "popularity",
    "freshness",
    "item2vec",
    "swing",
    "mind",
]
CHANNEL_RUN_ORDER = [
    "hotfresh",
    "eges",
    "youtubednn",
    "sasrec",
    "sdm",
    "dssm",
    "popularity",
    "freshness",
    "item2vec",
    "swing",
    "mind",
]
DEFAULT_CHANNELS = ["hotfresh", "eges", "youtubednn", "sasrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train, evaluate, and fuse recall channels.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--channels", nargs="*", default=DEFAULT_CHANNELS, choices=ALL_CHANNELS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--softmax_temperature", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--rank_base", type=float, default=0.0)
    parser.add_argument("--min_quota", type=int, default=5)
    parser.add_argument("--swing_alpha1", type=float, default=5.0)
    parser.add_argument("--swing_alpha2", type=float, default=1.0)
    parser.add_argument("--swing_beta", type=float, default=0.3)
    parser.add_argument("--swing_max_user_items", type=int, default=600)
    parser.add_argument("--swing_max_user_per_item", type=int, default=700)
    parser.add_argument("--swing_max_pair_users", type=int, default=200)
    parser.add_argument("--swing_max_sim_items", type=int, default=200)
    parser.add_argument("--item2vec_max_user_items", type=int, default=50)
    parser.add_argument("--eges_window", type=int, default=5)
    parser.add_argument("--eges_max_user_items", type=int, default=50)
    parser.add_argument("--eges_min_count", type=int, default=1)
    parser.add_argument("--hotfresh_half_life_days", type=float, default=3.0)
    parser.add_argument("--youtubednn_hidden_dims", type=str, default="128,64")
    parser.add_argument("--youtubednn_dropout", type=float, default=0.0)
    parser.add_argument("--dssm_user_hidden_dims", type=str, default="")
    parser.add_argument("--dssm_item_hidden_dims", type=str, default="")
    parser.add_argument("--dssm_dropout", type=float, default=0.0)
    parser.add_argument("--sdm_num_heads", type=int, default=4)
    parser.add_argument("--sdm_lstm_layers", type=int, default=1)
    parser.add_argument("--sdm_dropout", type=float, default=0.0)
    parser.add_argument("--sdm_item_hidden_dims", type=str, default="")
    parser.add_argument("--sasrec_max_seq_len", type=int, default=200)
    parser.add_argument("--sasrec_num_heads", type=int, default=2)
    parser.add_argument("--sasrec_num_layers", type=int, default=2)
    parser.add_argument("--sasrec_dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate_batch_size", type=int, default=32)
    parser.add_argument("--item_batch_size", type=int, default=50000)
    parser.add_argument("--output_path", type=str, default="checkpoints/recall/all_recall_metrics.json")
    parser.add_argument("--save_candidates", action="store_true")
    parser.add_argument("--candidates_path", type=str, default="checkpoints/recall/all_recall_candidates.pkl")
    parser.add_argument("--channel_cache_dir", type=str, default="checkpoints/recall/channel_candidates")
    parser.add_argument("--force_recall", action="store_true")
    parser.add_argument("--eval_seen_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fusion_mode", choices=["rank", "union"], default="rank")
    parser.add_argument("--fusion_top_k", type=int, default=None)
    return parser.parse_args()


def parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    value = str(value).strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def order_channels(channels):
    selected = list(dict.fromkeys(channels))
    order_rank = {channel: idx for idx, channel in enumerate(CHANNEL_RUN_ORDER)}
    return sorted(selected, key=lambda channel: order_rank.get(channel, len(order_rank)))


def get_channel_cache_path(cache_dir, channel, top_k):
    return Path(cache_dir) / f"{channel}_top{top_k}_candidates.pkl"


def build_data_signature(data_dir, train_data, test_data):
    metadata_path = Path(data_dir) / "preprocess_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    return {
        "train_rows": int(len(train_data["user_id"])),
        "test_rows": int(len(test_data["user_id"])),
        "train_items": int(len(set(int(item_id) for item_id in train_data["video_id"]))),
        "metadata": metadata,
    }


def fuse_by_union(channel_results, channels, user_ids, top_k):
    fused_results = {}
    metadata = {}

    for user_id in user_ids:
        selected = []
        selected_set = set()
        user_metadata = {}
        for channel in channels:
            for rank, item_id in enumerate(channel_results[channel].get(user_id, []), start=1):
                item_id = int(item_id)
                if item_id not in user_metadata:
                    user_metadata[item_id] = {
                        "sources": [],
                        "channel_ranks": {},
                        "fusion_mode": "union",
                    }
                user_metadata[item_id]["sources"].append(channel)
                user_metadata[item_id]["channel_ranks"][channel] = rank
                if item_id not in selected_set:
                    selected.append(item_id)
                    selected_set.add(item_id)
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        for item_meta in user_metadata.values():
            item_meta["sources"] = sorted(set(item_meta["sources"]))
            item_meta["hit_count"] = len(item_meta["sources"])

        fused_results[user_id] = selected
        metadata[user_id] = {
            item_id: user_metadata[item_id]
            for item_id in selected
            if item_id in user_metadata
        }

    return fused_results, metadata


def build_base_config(args, feature_dims, model_type):
    item_feature_names = [
        "video_id",
        "author_id",
        "music_id",
        "tag",
        "video_type",
        "upload_type",
        "visible_status",
        "music_type",
    ]
    user_feature_names = [
        "user_id",
        "user_active_degree",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]
    return {
        "model_type": model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "embedding_dim": args.embedding_dim,
        "num_negatives": args.num_negatives,
        "softmax_temperature": args.softmax_temperature,
        "item_feature_dims": {
            feature_name: feature_dims[feature_name]
            for feature_name in item_feature_names
            if feature_name in feature_dims
        },
        "user_feature_dims": {
            feature_name: feature_dims[feature_name]
            for feature_name in user_feature_names
            if feature_name in feature_dims
        },
        "alpha1": args.swing_alpha1,
        "alpha2": args.swing_alpha2,
        "beta": args.swing_beta,
        "max_user_items": args.swing_max_user_items,
        "max_user_per_item": args.swing_max_user_per_item,
        "max_pair_users": args.swing_max_pair_users,
        "max_sim_items": args.swing_max_sim_items,
        "item2vec_max_user_items": args.item2vec_max_user_items,
        "eges_window": args.eges_window,
        "eges_max_user_items": args.eges_max_user_items,
        "eges_min_count": args.eges_min_count,
        "hotfresh_half_life_days": args.hotfresh_half_life_days,
        "youtubednn_hidden_dims": parse_int_list(args.youtubednn_hidden_dims) or [128, 64],
        "youtubednn_dropout": args.youtubednn_dropout,
        "dssm_user_hidden_dims": parse_int_list(args.dssm_user_hidden_dims),
        "dssm_item_hidden_dims": parse_int_list(args.dssm_item_hidden_dims),
        "dssm_dropout": args.dssm_dropout,
        "sdm_num_heads": args.sdm_num_heads,
        "sdm_lstm_layers": args.sdm_lstm_layers,
        "sdm_dropout": args.sdm_dropout,
        "sdm_item_hidden_dims": parse_int_list(args.sdm_item_hidden_dims),
        "sasrec_max_seq_len": args.sasrec_max_seq_len,
        "sasrec_num_heads": args.sasrec_num_heads,
        "sasrec_num_layers": args.sasrec_num_layers,
        "sasrec_dropout": args.sasrec_dropout,
        "workers": args.workers,
        "candidate_batch_size": args.candidate_batch_size,
        "item_batch_size": args.item_batch_size,
    }


def run_or_load_channel(
    channel,
    args,
    feature_dims,
    train_data,
    test_data,
    user_sequences,
    video_info,
    all_item_ids,
    cache_dir,
    data_signature,
):
    cache_path = get_channel_cache_path(cache_dir, channel, args.top_k)
    channel_start = time.time()
    config = build_base_config(args, feature_dims, channel)

    print("=" * 80)
    print(f"Recall channel: {channel}")
    print("=" * 80)

    if cache_path.exists() and not args.force_recall:
        cached = joblib.load(cache_path)
        cache_signature = cached.get("data_signature") if isinstance(cached, dict) else None
        cache_config = cached.get("config") if isinstance(cached, dict) else None
        if cache_signature == data_signature and cache_config == config:
            print(f"[{channel}] cache found, skip training and generation: {cache_path}")
            results = cached["results"] if isinstance(cached, dict) and "results" in cached else cached
        else:
            print(f"[{channel}] cache exists but data/config signature changed, rerun: {cache_path}")
            cached = None
    else:
        cached = None

    if cached is None:
        manager = RecallManager(config)
        manager.train(train_data, user_sequences, video_info)
        results = manager.generate_candidates(user_sequences, all_item_ids, top_k=args.top_k)
        joblib.dump(
            {
                "channel": channel,
                "top_k": args.top_k,
                "results": results,
                "config": config,
                "data_signature": data_signature,
            },
            cache_path,
            compress=3,
        )
        print(f"[{channel}] candidates saved: {cache_path}")

    eval_item_ids = all_item_ids if args.eval_seen_only else None
    metrics = evaluate_recall(results, test_data, [args.top_k], candidate_item_ids=eval_item_ids)
    elapsed = time.time() - channel_start
    print(f"[{channel}] {metrics}")
    return results, metrics, elapsed, str(cache_path)


def main():
    args = parse_args()
    start_time = time.time()
    data_dir = Path(args.data_dir)
    channels = order_channels(args.channels)
    fusion_top_k = args.fusion_top_k if args.fusion_top_k is not None else args.top_k
    channel_cache_dir = Path(args.channel_cache_dir)
    channel_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading recall data...")
    train_eval_dict = joblib.load(data_dir / "kuairand_train_eval.pkl")
    user_sequences = joblib.load(data_dir / "user_sequences.pkl")
    video_info = joblib.load(data_dir / "video_info.pkl")
    feature_dims = joblib.load(data_dir / "kuairand_feature_dict.pkl")

    train_data = train_eval_dict["train"]
    test_data = train_eval_dict["test"]
    all_item_ids = list(set(train_data["video_id"]))
    data_signature = build_data_signature(data_dir, train_data, test_data)
    test_pool_overlap = compute_test_pool_overlap(all_item_ids, test_data)

    print("Test positives vs train candidate pool overlap:")
    print(json.dumps(test_pool_overlap, ensure_ascii=False, indent=2))
    print(f"Requested channels: {args.channels}")
    print(f"Run order: {channels}")

    channel_results = {}
    channel_metrics = {}
    channel_times = {}
    channel_cache_paths = {}

    for channel in tqdm(channels, desc="Recall channel progress"):
        results, metrics, elapsed, cache_path = run_or_load_channel(
            channel,
            args,
            feature_dims,
            train_data,
            test_data,
            user_sequences,
            video_info,
            all_item_ids,
            channel_cache_dir,
            data_signature,
        )
        channel_results[channel] = results
        channel_metrics[channel] = metrics
        channel_times[channel] = elapsed
        channel_cache_paths[channel] = cache_path

    fusion_config = {
        "weights": {channel: 1.0 for channel in channels},
        "rank_base": args.rank_base,
        "min_quota_per_channel": {channel: args.min_quota for channel in channels},
        "fusion_mode": args.fusion_mode,
        "fusion_top_k": fusion_top_k,
    }
    pipeline = FullPipeline(
        data_path=str(data_dir),
        recall_checkpoints={},
        rank_checkpoint="",
        recall_fusion_config=fusion_config,
    )
    test_user_ids = list(set(test_data["user_id"]))
    if args.fusion_mode == "union":
        fused_results, fusion_metadata = fuse_by_union(
            channel_results, channels, test_user_ids, top_k=fusion_top_k
        )
        pipeline.last_recall_fusion_metadata = fusion_metadata
    else:
        fused_results = pipeline._fuse_recall_results(
            [(channel, channel_results[channel]) for channel in channels],
            test_user_ids,
            top_k=fusion_top_k,
        )
    eval_item_ids = all_item_ids if args.eval_seen_only else None
    fused_top_k_list = sorted(set([args.top_k, fusion_top_k]))
    fused_metrics = evaluate_recall(
        fused_results, test_data, fused_top_k_list, candidate_item_ids=eval_item_ids
    )
    overlap = compute_channel_overlap(channel_results, top_k=args.top_k)

    channel_metrics = evaluate_recall_channels(
        channel_results, test_data, top_k=args.top_k, candidate_item_ids=eval_item_ids
    )

    output = {
        "config": {
            "data_dir": str(data_dir),
            "channels": channels,
            "requested_channels": args.channels,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "embedding_dim": args.embedding_dim,
            "num_negatives": args.num_negatives,
            "softmax_temperature": args.softmax_temperature,
            "top_k": args.top_k,
            "fusion_mode": args.fusion_mode,
            "fusion_top_k": fusion_top_k,
            "workers": args.workers,
            "candidate_batch_size": args.candidate_batch_size,
            "item_batch_size": args.item_batch_size,
            "channel_cache_dir": str(channel_cache_dir),
            "force_recall": args.force_recall,
            "eval_seen_only": args.eval_seen_only,
            "swing_alpha1": args.swing_alpha1,
            "swing_alpha2": args.swing_alpha2,
            "swing_beta": args.swing_beta,
            "swing_max_user_items": args.swing_max_user_items,
            "swing_max_user_per_item": args.swing_max_user_per_item,
            "swing_max_pair_users": args.swing_max_pair_users,
            "swing_max_sim_items": args.swing_max_sim_items,
            "item2vec_max_user_items": args.item2vec_max_user_items,
            "eges_window": args.eges_window,
            "eges_max_user_items": args.eges_max_user_items,
            "eges_min_count": args.eges_min_count,
            "hotfresh_half_life_days": args.hotfresh_half_life_days,
            "youtubednn_hidden_dims": parse_int_list(args.youtubednn_hidden_dims) or [128, 64],
            "youtubednn_dropout": args.youtubednn_dropout,
            "dssm_user_hidden_dims": parse_int_list(args.dssm_user_hidden_dims),
            "dssm_item_hidden_dims": parse_int_list(args.dssm_item_hidden_dims),
            "dssm_dropout": args.dssm_dropout,
            "sdm_num_heads": args.sdm_num_heads,
            "sdm_lstm_layers": args.sdm_lstm_layers,
            "sdm_dropout": args.sdm_dropout,
            "sdm_item_hidden_dims": parse_int_list(args.sdm_item_hidden_dims),
            "sasrec_max_seq_len": args.sasrec_max_seq_len,
            "sasrec_num_heads": args.sasrec_num_heads,
            "sasrec_num_layers": args.sasrec_num_layers,
            "sasrec_dropout": args.sasrec_dropout,
            "fusion": fusion_config,
        },
        "channel_metrics": channel_metrics,
        "fused_metrics": fused_metrics,
        "test_pool_overlap": test_pool_overlap,
        "channel_overlap": overlap,
        "channel_times_seconds": channel_times,
        "channel_cache_paths": channel_cache_paths,
        "elapsed_seconds": time.time() - start_time,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved: {output_path}")
    print(f"[fused] {fused_metrics}")

    if args.save_candidates:
        candidates_path = Path(args.candidates_path)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "channels": channel_results,
                "fused": fused_results,
                "fusion_metadata": pipeline.last_recall_fusion_metadata,
            },
            candidates_path,
            compress=3,
        )
        print(f"Fused candidates saved: {candidates_path}")


if __name__ == "__main__":
    main()
