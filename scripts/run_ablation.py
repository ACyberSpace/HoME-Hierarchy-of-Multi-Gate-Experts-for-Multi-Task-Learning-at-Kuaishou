#!/usr/bin/env python3
"""
Run reproducible HoME ablation experiments.

Each experiment delegates to main.py and writes its own checkpoint directory and
final_metrics.json. Keep this script small: it is an experiment orchestrator, not
another training implementation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "home_bce": {
        "description": "Original HoME with equal-weight BCE; this is the primary single-variable baseline.",
        "args": {"--model_name": "home", "--loss_name": "bce"},
    },
    "home_adaptive": {
        "description": "Original HoME with adaptive uncertainty-weighted loss; same structure as home_bce.",
        "args": {"--model_name": "home", "--loss_name": "uncertainty"},
    },
    "mmoe": {
        "description": "PyTorch MMoE baseline ported from fun-rec and trained on the same KuaiRand features.",
        "args": {"--model_name": "mmoe", "--loss_name": "bce"},
    },
    "ple": {
        "description": "PyTorch PLE baseline ported from fun-rec CGC and trained on the same KuaiRand features.",
        "args": {"--model_name": "ple", "--loss_name": "bce"},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run HoME ablation matrix")
    parser.add_argument("--data_path", default="./data/kuairand_train_eval.pkl")
    parser.add_argument("--feature_path", default="./data/kuairand_feature_dict.pkl")
    parser.add_argument("--output_dir", default="checkpoints/ablation")
    parser.add_argument("--epochs", default="10")
    parser.add_argument("--batch_size", default="1024")
    parser.add_argument("--lr", default="1e-3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default="42")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Subset of experiments to run.",
    )
    return parser.parse_args()


def flatten_extra_args(extra_args):
    flattened = []
    for key, value in extra_args.items():
        flattened.extend([key, value])
    return flattened


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for name in args.experiments:
        spec = EXPERIMENTS[name]
        save_dir = root / name
        metrics_path = save_dir / "final_metrics.json"
        command = [
            sys.executable,
            "main.py",
            "--data_path",
            args.data_path,
            "--feature_path",
            args.feature_path,
            "--epochs",
            args.epochs,
            "--batch_size",
            args.batch_size,
            "--lr",
            args.lr,
            "--device",
            args.device,
            "--seed",
            args.seed,
            "--save_dir",
            str(save_dir),
            "--metrics_path",
            str(metrics_path),
            *flatten_extra_args(spec["args"]),
        ]

        print("=" * 80)
        print(f"Experiment: {name}")
        print(spec["description"])
        print("Command:", " ".join(command))
        print("=" * 80)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
