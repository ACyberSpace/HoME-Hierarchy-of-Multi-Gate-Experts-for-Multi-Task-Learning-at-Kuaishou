# HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning

PyTorch implementation of a short-video multi-task ranking system inspired by Kuaishou's HoME paper. The repository includes HoME, MMoE and PLE ranking models, KuaiRand-1K preprocessing utilities, recall modules, unified training/evaluation, and ablation scripts.

## Project Structure

```
home_project/
├── config.py                  # Shared data/model/training configuration
├── main.py                    # Ranking model training entry point
├── pipeline.py                # Recall-to-rank offline inference pipeline
├── requirements.txt
├── data/
│   └── preprocess.py          # KuaiRand-1K preprocessing script
├── ranking/
│   ├── data/                  # Dataset, feature processor, dataloader
│   ├── models/                # HoME, MMoE, PLE, gates, experts, losses
│   └── training/              # Trainer and evaluator
├── recall/
│   ├── data/
│   ├── models/                # Swing, Item2Vec, DSSM, MIND, SDM, freshness
│   ├── training/
│   └── evaluation/
├── scripts/
│   └── run_ablation.py        # Reproducible experiment matrix
└── docs/
    └── home_story.md          # Project narrative and interview notes
```

Large data files, checkpoints and local IDE files are intentionally ignored by Git. Put generated KuaiRand files under `data/` locally, but do not commit them.

## Features

- HoME reproduction with Feature-Gate, Self-Gate, hierarchical expert routing, Gated Cross Experts and uncertainty-weighted multi-task loss.
- PyTorch MMoE and PLE baselines adapted from fun-rec's TensorFlow implementations.
- Unified KuaiRand-1K feature embedding, training, checkpointing, AUC and GAUC evaluation.
- Ablation script for HoME, HoME+BCE, MMoE, PLE, smaller expert capacity and task-specific expert variants.
- Recall modules for Swing, Item2Vec, DSSM, MIND, SDM and freshness-based recall.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Expected local files:

```text
data/
├── kuairand_train_eval.pkl
├── kuairand_feature_dict.pkl
├── user_sequences.pkl
└── video_info.pkl
```

These files are generated artifacts and are ignored by Git. If starting from the raw KuaiRand-1K files, run:

```bash
python data/preprocess.py
```

## Train Ranking Models

```bash
python main.py --model_name home --loss_name uncertainty --epochs 10 --batch_size 1024
python main.py --model_name mmoe --loss_name bce --epochs 10 --batch_size 1024
python main.py --model_name ple --loss_name bce --epochs 10 --batch_size 1024
```

Final metrics are saved to `checkpoints/final_metrics.json` by default. The trainer evaluates the best checkpoint according to `avg_gauc` when GAUC is available.

## Run Ablations

```bash
python scripts/run_ablation.py --epochs 10 --batch_size 1024 --device cuda
```

## Notes

- `pipeline.py` wires current recall managers to HoME/MMoE/PLE-style ranking models. It expects trained recall checkpoints plus a constructed ranking model and `FeatureEmbedding`; pass `rank_checkpoint` to load saved ranking and embedding weights.
- Sequence fields are loaded by the ranking dataloader, but the current ranking models mainly use tabular sparse/id features.

## Citation

```bibtex
@inproceedings{home2025,
  title={HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou},
  author={Kuaishou Research},
  booktitle={KDD},
  year={2025}
}
```
