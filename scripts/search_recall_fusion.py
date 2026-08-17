#!/usr/bin/env python3
"""Search recall fusion weights on a validation split.

The script expects a joblib/pickle file containing:
{
    "swing": {user_id: [item_id, ...]},
    "item2vec": {user_id: [item_id, ...]},
    ...
}
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import joblib

from pipeline import FullPipeline
from recall.evaluation.evaluator import evaluate_recall


DEFAULT_CHANNELS = ["swing", "item2vec", "dssm", "mind", "sdm", "freshness"]


def load_object(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def sample_weights(channels, low, high):
    weights = {channel: random.uniform(low, high) for channel in channels}
    total = sum(weights.values()) or 1.0
    return {channel: value / total * len(channels) for channel, value in weights.items()}


def main():
    parser = argparse.ArgumentParser(description="Search weighted reciprocal-rank recall fusion.")
    parser.add_argument("--channel_results", required=True)
    parser.add_argument("--validation_data", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--rank_base", type=float, default=0.0)
    parser.add_argument("--min_quota", type=int, default=5)
    parser.add_argument("--weight_low", type=float, default=0.25)
    parser.add_argument("--weight_high", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    channel_results = load_object(args.channel_results)
    validation_data = load_object(args.validation_data)
    channels = [channel for channel in DEFAULT_CHANNELS if channel in channel_results]
    user_ids = list({int(uid) for results in channel_results.values() for uid in results})

    best = None
    trials = []
    pipeline = FullPipeline(
        data_path=".",
        recall_checkpoints={},
        rank_checkpoint="",
        recall_fusion_config={},
    )

    for trial_idx in range(args.trials):
        weights = sample_weights(channels, args.weight_low, args.weight_high)
        pipeline.recall_fusion_config = {
            "weights": weights,
            "rank_base": args.rank_base,
            "min_quota_per_channel": {channel: args.min_quota for channel in channels},
        }
        fused = pipeline._fuse_recall_results(
            [(channel, channel_results[channel]) for channel in channels],
            user_ids,
            top_k=args.top_k,
        )
        metrics = evaluate_recall(fused, validation_data, [args.top_k])
        score = metrics[f"MicroRecall@{args.top_k}"]
        trial = {"trial": trial_idx, "weights": weights, "metrics": metrics}
        trials.append(trial)
        if best is None or score > best["metrics"][f"MicroRecall@{args.top_k}"]:
            best = trial

    output = {
        "search": {
            "seed": args.seed,
            "trials": args.trials,
            "top_k": args.top_k,
            "rank_base": args.rank_base,
            "min_quota": args.min_quota,
            "weight_low": args.weight_low,
            "weight_high": args.weight_high,
            "channels": channels,
        },
        "best": best,
        "trials": trials,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
