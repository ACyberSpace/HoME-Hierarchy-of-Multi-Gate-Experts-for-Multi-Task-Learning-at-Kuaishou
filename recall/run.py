import argparse
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recall.manager import RecallManager


def main():
    parser = argparse.ArgumentParser(description="训练召回模型")
    parser.add_argument("--model_type", type=str, default="sdm",
                        choices=["swing", "item2vec", "dssm", "mind", "sdm", "freshness"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=100)

    args = parser.parse_args()

    config = {
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "embedding_dim": args.embedding_dim,
    }

    print("加载数据...")
    import joblib
    train_eval_dict = joblib.load("../data/kuairand_train_eval.pkl")
    user_sequences = joblib.load("../data/user_sequences.pkl")
    video_info = joblib.load("../data/video_info.pkl")

    train_data = train_eval_dict["train"]
    test_data = train_eval_dict["test"]

    feature_dims = joblib.load("../data/kuairand_feature_dict.pkl")
    config["item_feature_dims"] = {
        "video_id": feature_dims["video_id"],
        "author_id": feature_dims["author_id"],
        "music_id": feature_dims["music_id"],
    }
    config["user_feature_dims"] = {
        "user_id": feature_dims["user_id"],
    }

    all_item_ids = list(set(train_data["video_id"]))

    manager = RecallManager(config)

    print(f"训练 {args.model_type} 模型...")
    manager.train(train_data, user_sequences, video_info)

    print(f"生成召回候选...")
    recall_results = manager.generate_candidates(user_sequences, all_item_ids, top_k=args.top_k)

    print(f"评估召回结果...")
    metrics = manager.evaluate(recall_results, test_data)
    print(f"评估指标: {metrics}")

    print("保存模型...")
    manager.save_model(f"checkpoints/recall/{args.model_type}_model.pth")


if __name__ == "__main__":
    main()
