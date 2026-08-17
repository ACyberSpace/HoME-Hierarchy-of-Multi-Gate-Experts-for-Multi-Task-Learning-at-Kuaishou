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
    evaluate_recall,
    evaluate_recall_channels,
)
from recall.manager import RecallManager


DEFAULT_CHANNELS = ["swing", "item2vec", "dssm", "mind", "sdm", "freshness"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train, evaluate, and fuse all recall channels.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--channels", nargs="*", default=DEFAULT_CHANNELS, choices=DEFAULT_CHANNELS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=64)
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="checkpoints/recall/all_recall_metrics.json")
    parser.add_argument("--save_candidates", action="store_true")
    parser.add_argument("--candidates_path", type=str, default="checkpoints/recall/all_recall_candidates.pkl")
    return parser.parse_args()


def build_base_config(args, feature_dims, model_type):
    return {
        "model_type": model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "embedding_dim": args.embedding_dim,
        "item_feature_dims": {
            "video_id": feature_dims["video_id"],
            "author_id": feature_dims["author_id"],
            "music_id": feature_dims["music_id"],
        },
        "user_feature_dims": {
            "user_id": feature_dims["user_id"],
        },
        "alpha1": args.swing_alpha1,
        "alpha2": args.swing_alpha2,
        "beta": args.swing_beta,
        "max_user_items": args.swing_max_user_items,
        "max_user_per_item": args.swing_max_user_per_item,
        "max_pair_users": args.swing_max_pair_users,
        "max_sim_items": args.swing_max_sim_items,
        "item2vec_max_user_items": args.item2vec_max_user_items,
        "workers": args.workers,
    }


def main():
    args = parse_args()
    start_time = time.time()
    data_dir = Path(args.data_dir)

    print("加载召回数据...")
    train_eval_dict = joblib.load(data_dir / "kuairand_train_eval.pkl")
    user_sequences = joblib.load(data_dir / "user_sequences.pkl")
    video_info = joblib.load(data_dir / "video_info.pkl")
    feature_dims = joblib.load(data_dir / "kuairand_feature_dict.pkl")

    train_data = train_eval_dict["train"]
    test_data = train_eval_dict["test"]
    all_item_ids = list(set(train_data["video_id"]))

    channel_results = {}
    channel_metrics = {}
    channel_times = {}

    for channel in tqdm(args.channels, desc="六路召回总进度"):
        channel_start = time.time()
        print("=" * 80)
        print(f"训练并评估召回通道: {channel}")
        print("=" * 80)

        config = build_base_config(args, feature_dims, channel)
        manager = RecallManager(config)
        manager.train(train_data, user_sequences, video_info)

        results = manager.generate_candidates(user_sequences, all_item_ids, top_k=args.top_k)
        metrics = evaluate_recall(results, test_data, [args.top_k])

        channel_results[channel] = results
        channel_metrics[channel] = metrics
        channel_times[channel] = time.time() - channel_start

        print(f"[{channel}] {metrics}")

    fusion_config = {
        "weights": {channel: 1.0 for channel in args.channels},
        "rank_base": args.rank_base,
        "min_quota_per_channel": {channel: args.min_quota for channel in args.channels},
    }
    pipeline = FullPipeline(
        data_path=str(data_dir),
        recall_checkpoints={},
        rank_checkpoint="",
        recall_fusion_config=fusion_config,
    )
    test_user_ids = list(set(test_data["user_id"]))
    fused_results = pipeline._fuse_recall_results(
        [(channel, channel_results[channel]) for channel in args.channels],
        test_user_ids,
        top_k=args.top_k,
    )
    fused_metrics = evaluate_recall(fused_results, test_data, [args.top_k])
    overlap = compute_channel_overlap(channel_results, top_k=args.top_k)

    # Keep the direct helper output in the JSON so each channel has the same schema.
    channel_metrics = evaluate_recall_channels(channel_results, test_data, top_k=args.top_k)

    output = {
        "config": {
            "data_dir": str(data_dir),
            "channels": args.channels,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "embedding_dim": args.embedding_dim,
            "top_k": args.top_k,
            "workers": args.workers,
            "swing_alpha1": args.swing_alpha1,
            "swing_alpha2": args.swing_alpha2,
            "swing_beta": args.swing_beta,
            "swing_max_user_items": args.swing_max_user_items,
            "swing_max_user_per_item": args.swing_max_user_per_item,
            "swing_max_pair_users": args.swing_max_pair_users,
            "swing_max_sim_items": args.swing_max_sim_items,
            "item2vec_max_user_items": args.item2vec_max_user_items,
            "fusion": fusion_config,
        },
        "channel_metrics": channel_metrics,
        "fused_metrics": fused_metrics,
        "channel_overlap": overlap,
        "channel_times_seconds": channel_times,
        "elapsed_seconds": time.time() - start_time,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"指标已保存: {output_path}")
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
        print(f"候选已保存: {candidates_path}")


if __name__ == "__main__":
    main()
