"""Train and evaluate popularity baseline and PyG LightGCN recommenders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.recommender_pipeline import run_full_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Book-Crossing recommenders.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_full_pipeline(
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        k=args.top_k,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
