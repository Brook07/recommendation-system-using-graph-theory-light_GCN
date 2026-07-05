"""Training and recommendation pipeline for the reduced Book-Crossing graph.

This module keeps preprocessing inputs in ``data/processed`` and writes all
training outputs to ``data/processed/model_training``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


PROCESSED_DIR = Path("data/processed")
TRAINING_DIR = PROCESSED_DIR / "model_training"
MODELS_DIR = TRAINING_DIR / "models"
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"
EMBEDDINGS_DIR = TRAINING_DIR / "embeddings"
METRICS_DIR = TRAINING_DIR / "metrics"
PLOTS_DIR = TRAINING_DIR / "plots"
LOGS_DIR = TRAINING_DIR / "logs"
RECOMMENDATIONS_DIR = TRAINING_DIR / "recommendations"
REPORTS_DIR = TRAINING_DIR / "reports"
SPLITS_DIR = TRAINING_DIR / "splits"

K = 10


@dataclass
class PipelineArtifacts:
    user_mapping: pd.DataFrame
    book_mapping: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def ensure_training_dirs() -> None:
    """Create the model-training output tree."""

    for directory in [
        MODELS_DIR,
        CHECKPOINTS_DIR,
        EMBEDDINGS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        RECOMMENDATIONS_DIR,
        REPORTS_DIR,
        SPLITS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_reduced_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the reduced interactions and node mappings created by preprocessing."""

    interactions = pd.read_csv(PROCESSED_DIR / "cleaned_ratings.csv")
    user_mapping = pd.read_csv(PROCESSED_DIR / "user_mapping.csv")
    book_mapping = pd.read_csv(PROCESSED_DIR / "book_mapping.csv")
    books = pd.read_csv(PROCESSED_DIR / "books_reduced_lightgcn.csv", low_memory=False)
    return interactions, user_mapping, book_mapping, books


def split_interactions_per_user(
    interactions: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each user's interactions into train, validation, and test rows."""

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _, group in interactions.groupby("User-ID", sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        if n < 3:
            train_indices.extend(indices.tolist())
            continue

        n_test = max(1, int(math.floor(n * test_frac)))
        n_val = max(1, int(math.floor(n * val_frac)))
        test_indices.extend(indices[:n_test].tolist())
        val_indices.extend(indices[n_test : n_test + n_val].tolist())
        train_indices.extend(indices[n_test + n_val :].tolist())

    train_df = interactions.loc[train_indices].reset_index(drop=True)
    val_df = interactions.loc[val_indices].reset_index(drop=True)
    test_df = interactions.loc[test_indices].reset_index(drop=True)
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train, validation, and test splits."""

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train_interactions.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "validation_interactions.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test_interactions.csv", index=False)


def build_user_seen_items(interactions: pd.DataFrame) -> dict[int, set[str]]:
    """Map each user to ISBNs already observed in the given interaction table."""

    return {
        int(user_id): set(group["ISBN"].astype(str))
        for user_id, group in interactions.groupby("User-ID")
    }


def build_user_ground_truth(interactions: pd.DataFrame) -> dict[int, set[str]]:
    """Map each user to held-out relevant ISBNs."""

    return build_user_seen_items(interactions)


def dcg_at_k(hits: Iterable[int]) -> float:
    """Discounted cumulative gain for a binary hit list."""

    return float(sum(hit / math.log2(index + 2) for index, hit in enumerate(hits)))


def average_precision_at_k(recommended: list[str], ground_truth: set[str], k: int = K) -> float:
    """Average precision at k for one user."""

    if not ground_truth:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, isbn in enumerate(recommended[:k], start=1):
        if isbn in ground_truth:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(ground_truth), k)


def evaluate_recommendations(
    recommendations: dict[int, list[str]],
    ground_truth: dict[int, set[str]],
    total_items: int,
    k: int = K,
) -> dict[str, float]:
    """Compute ranking metrics for top-k recommendation lists."""

    hit_rates: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    maps: list[float] = []
    recommended_items: set[str] = set()

    for user_id, true_items in ground_truth.items():
        recs = recommendations.get(user_id, [])[:k]
        recommended_items.update(recs)
        hits = [1 if isbn in true_items else 0 for isbn in recs]
        hit_count = sum(hits)

        hit_rates.append(1.0 if hit_count > 0 else 0.0)
        precisions.append(hit_count / k)
        recalls.append(hit_count / max(1, len(true_items)))
        ideal_hits = [1] * min(len(true_items), k)
        ideal_dcg = dcg_at_k(ideal_hits)
        ndcgs.append(dcg_at_k(hits) / ideal_dcg if ideal_dcg else 0.0)
        maps.append(average_precision_at_k(recs, true_items, k))

    return {
        "HitRate@10": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "Precision@10": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "MAP@10": float(np.mean(maps)) if maps else 0.0,
        "Coverage": len(recommended_items) / total_items if total_items else 0.0,
    }


def save_recommendations(name: str, recommendations: dict[int, list[str]]) -> None:
    """Save recommendation lists in CSV and JSON formats."""

    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"User-ID": user_id, "rank": rank, "ISBN": isbn}
        for user_id, isbns in recommendations.items()
        for rank, isbn in enumerate(isbns, start=1)
    ]
    pd.DataFrame(rows).to_csv(RECOMMENDATIONS_DIR / f"{name}_recommendations.csv", index=False)
    with (RECOMMENDATIONS_DIR / f"{name}_recommendations.json").open("w", encoding="utf-8") as output:
        json.dump({str(user_id): isbns for user_id, isbns in recommendations.items()}, output, indent=2)


def popularity_recommendations(
    train_df: pd.DataFrame,
    candidate_users: Iterable[int],
    all_items: list[str],
    seen_by_user: dict[int, set[str]],
    k: int = K,
) -> dict[int, list[str]]:
    """Recommend globally popular books from the training set."""

    popularity = (
        train_df.groupby("ISBN")["Book-Rating"]
        .agg(["count", "mean"])
        .sort_values(["count", "mean"], ascending=[False, False])
    )
    ranked_items = popularity.index.astype(str).tolist()
    ranked_items.extend([isbn for isbn in all_items if isbn not in set(ranked_items)])

    recommendations: dict[int, list[str]] = {}
    for user_id in candidate_users:
        seen = seen_by_user.get(int(user_id), set())
        recommendations[int(user_id)] = [isbn for isbn in ranked_items if isbn not in seen][:k]
    return recommendations


def build_node_edge_index(
    train_df: pd.DataFrame,
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
) -> torch.Tensor:
    """Build a bidirectional edge index tensor from the training split."""

    user_lookup = user_mapping.set_index("User-ID")["node_index"].to_dict()
    book_lookup = book_mapping.set_index("ISBN")["node_index"].to_dict()
    user_nodes = train_df["User-ID"].map(user_lookup).to_numpy()
    book_nodes = train_df["ISBN"].map(book_lookup).to_numpy()
    forward = np.vstack([user_nodes, book_nodes])
    reverse = np.vstack([book_nodes, user_nodes])
    return torch.tensor(np.hstack([forward, reverse]), dtype=torch.long)


class PyGLightGCN(nn.Module):
    """LightGCN implemented with PyTorch Geometric LGConv layers."""

    def __init__(self, num_nodes: int, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        try:
            from torch_geometric.nn import LGConv
        except ImportError as error:  # pragma: no cover - depends on optional package
            raise ImportError(
                "torch_geometric is required for PyG LightGCN training. "
                "Install it with the PyTorch-compatible wheel before running training."
            ) from error

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.convs = nn.ModuleList([LGConv(normalize=True) for _ in range(num_layers)])
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Return propagated node embeddings."""

        embeddings = self.embedding.weight
        layer_outputs = [embeddings]
        for conv in self.convs:
            embeddings = conv(embeddings, edge_index)
            layer_outputs.append(embeddings)
        return torch.stack(layer_outputs, dim=0).mean(dim=0)


def sample_negative_items(
    users: np.ndarray,
    user_positive_items: dict[int, set[int]],
    all_item_nodes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one unseen negative item node for each user node."""

    negatives = []
    for user_node in users:
        positives = user_positive_items.get(int(user_node), set())
        while True:
            candidate = int(rng.choice(all_item_nodes))
            if candidate not in positives:
                negatives.append(candidate)
                break
    return np.array(negatives, dtype=np.int64)


def bpr_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""

    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_lightgcn_pyg(
    artifacts: PipelineArtifacts,
    embedding_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
) -> tuple[PyGLightGCN, dict[str, list[float]], torch.Tensor]:
    """Train PyG LightGCN and save model, embeddings, logs, and checkpoints."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_nodes = int(max(artifacts.user_mapping["node_index"].max(), artifacts.book_mapping["node_index"].max()) + 1)
    edge_index = build_node_edge_index(artifacts.train_df, artifacts.user_mapping, artifacts.book_mapping).to(resolved_device)

    user_lookup = artifacts.user_mapping.set_index("User-ID")["node_index"].to_dict()
    book_lookup = artifacts.book_mapping.set_index("ISBN")["node_index"].to_dict()
    train_pairs = artifacts.train_df[["User-ID", "ISBN"]].copy()
    train_pairs["user_node"] = train_pairs["User-ID"].map(user_lookup)
    train_pairs["book_node"] = train_pairs["ISBN"].map(book_lookup)
    train_pairs = train_pairs.dropna(subset=["user_node", "book_node"]).astype({"user_node": int, "book_node": int})

    val_pairs = artifacts.val_df[["User-ID", "ISBN"]].copy()
    val_pairs["user_node"] = val_pairs["User-ID"].map(user_lookup)
    val_pairs["book_node"] = val_pairs["ISBN"].map(book_lookup)
    val_pairs = val_pairs.dropna(subset=["user_node", "book_node"]).astype({"user_node": int, "book_node": int})

    user_positive_items: dict[int, set[int]] = {}
    for row in train_pairs.itertuples(index=False):
        user_positive_items.setdefault(int(row.user_node), set()).add(int(row.book_node))

    all_item_nodes = artifacts.book_mapping["node_index"].astype(int).to_numpy()
    model = PyGLightGCN(num_nodes, embedding_dim, num_layers).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_user_nodes = train_pairs["user_node"].to_numpy()
    train_book_nodes = train_pairs["book_node"].to_numpy()
    val_user_nodes = val_pairs["user_node"].to_numpy()
    val_book_nodes = val_pairs["book_node"].to_numpy()
    history = {"epoch": [], "train_loss": [], "validation_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_pairs))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            users = train_user_nodes[batch_idx]
            positives = train_book_nodes[batch_idx]
            negatives = sample_negative_items(users, user_positive_items, all_item_nodes, rng)

            embeddings = model(edge_index)
            loss = bpr_loss(
                embeddings[torch.tensor(users, dtype=torch.long, device=resolved_device)],
                embeddings[torch.tensor(positives, dtype=torch.long, device=resolved_device)],
                embeddings[torch.tensor(negatives, dtype=torch.long, device=resolved_device)],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_idx)

        model.eval()
        with torch.no_grad():
            embeddings = model(edge_index)
            val_negatives = sample_negative_items(val_user_nodes, user_positive_items, all_item_nodes, rng)
            validation_loss = bpr_loss(
                embeddings[torch.tensor(val_user_nodes, dtype=torch.long, device=resolved_device)],
                embeddings[torch.tensor(val_book_nodes, dtype=torch.long, device=resolved_device)],
                embeddings[torch.tensor(val_negatives, dtype=torch.long, device=resolved_device)],
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / max(1, len(train_pairs)))
        history["validation_loss"].append(float(validation_loss.item()))

        if epoch == epochs or epoch % 10 == 0:
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": embedding_dim,
                    "num_layers": num_layers,
                    "num_nodes": num_nodes,
                    "epoch": epoch,
                },
                CHECKPOINTS_DIR / f"lightgcn_epoch_{epoch}.pt",
            )

    with torch.no_grad():
        final_embeddings = model(edge_index).detach().cpu()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_nodes": num_nodes,
        },
        MODELS_DIR / "lightgcn_pyg.pt",
    )
    torch.save(
        {
            "user_embeddings": final_embeddings[artifacts.user_mapping["node_index"].astype(int).to_numpy()],
            "user_mapping": artifacts.user_mapping,
        },
        EMBEDDINGS_DIR / "lightgcn_user_embeddings.pt",
    )
    torch.save(
        {
            "book_embeddings": final_embeddings[artifacts.book_mapping["node_index"].astype(int).to_numpy()],
            "book_mapping": artifacts.book_mapping,
        },
        EMBEDDINGS_DIR / "lightgcn_book_embeddings.pt",
    )
    torch.save({"node_embeddings": final_embeddings}, EMBEDDINGS_DIR / "lightgcn_node_embeddings.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(LOGS_DIR / "lightgcn_training_history.csv", index=False)
    plot_training_history(history_df)
    return model, history, final_embeddings


def generate_embedding_recommendations(
    embeddings: torch.Tensor,
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
    users: Iterable[int],
    seen_by_user: dict[int, set[str]],
    k: int = K,
) -> dict[int, list[str]]:
    """Generate top-k recommendations from learned LightGCN embeddings."""

    user_lookup = user_mapping.set_index("User-ID")["node_index"].to_dict()
    book_lookup = book_mapping.set_index("ISBN")["node_index"].to_dict()
    inv_book_lookup = {node: isbn for isbn, node in book_lookup.items()}
    item_nodes = np.array(sorted(book_lookup.values()))
    item_embeddings = embeddings[item_nodes]
    recommendations: dict[int, list[str]] = {}

    for user_id in users:
        user_node = user_lookup.get(int(user_id))
        if user_node is None:
            continue
        scores = torch.matmul(item_embeddings, embeddings[int(user_node)])
        order = torch.argsort(scores, descending=True).cpu().numpy()
        seen = seen_by_user.get(int(user_id), set())
        recs = []
        for index in order:
            isbn = inv_book_lookup[int(item_nodes[index])]
            if isbn in seen:
                continue
            recs.append(isbn)
            if len(recs) >= k:
                break
        recommendations[int(user_id)] = recs
    return recommendations


def plot_training_history(history_df: pd.DataFrame) -> None:
    """Save LightGCN training and validation loss curve."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    ax.plot(history_df["epoch"], history_df["validation_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BPR loss")
    ax.set_title("LightGCN Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lightgcn_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame) -> None:
    """Save a grouped bar chart comparing baseline and LightGCN metrics."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_cols = ["HitRate@10", "Precision@10", "Recall@10", "NDCG@10", "MAP@10", "Coverage"]
    plot_df = metrics_df.set_index("model")[metric_cols]
    ax = plot_df.T.plot(kind="bar", figsize=(10, 5), rot=30)
    ax.set_ylabel("Score")
    ax.set_title("Recommendation Model Comparison")
    ax.legend(title="Model")
    ax.figure.tight_layout()
    ax.figure.savefig(PLOTS_DIR / "model_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def write_report(metrics_df: pd.DataFrame) -> None:
    """Write a short report explaining metric differences."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = metrics_df[metrics_df["model"] == "PopularityBaseline"].iloc[0]
    report_lines = [
        "# Model Evaluation Report",
        "",
        "The popularity baseline ranks books by global training-set popularity and excludes books already seen by each user.",
        "LightGCN learns user and book embeddings from the bipartite graph and ranks books by embedding similarity.",
        "",
        "## Metrics",
        "",
        "```text",
        metrics_df.to_string(index=False),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if "LightGCN" in set(metrics_df["model"]):
        lightgcn = metrics_df[metrics_df["model"] == "LightGCN"].iloc[0]
        for metric in ["HitRate@10", "Precision@10", "Recall@10", "NDCG@10", "MAP@10", "Coverage"]:
            delta = lightgcn[metric] - baseline[metric]
            report_lines.append(f"- {metric}: LightGCN changes by {delta:.4f} versus popularity baseline.")
        report_lines.append(
            "- Improvements indicate that graph-based collaborative filtering is capturing personalized user-book structure beyond global popularity."
        )
    else:
        report_lines.append("- LightGCN metrics are not available because PyTorch Geometric training was not run.")
    (REPORTS_DIR / "model_evaluation_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_popularity_baseline(artifacts: PipelineArtifacts, books: pd.DataFrame, k: int = K) -> dict[str, float]:
    """Evaluate and save the popularity baseline."""

    all_items = artifacts.book_mapping["ISBN"].astype(str).tolist()
    train_seen = build_user_seen_items(artifacts.train_df)
    test_truth = build_user_ground_truth(artifacts.test_df)
    recs = popularity_recommendations(artifacts.train_df, test_truth.keys(), all_items, train_seen, k)
    save_recommendations("popularity_baseline", recs)
    metrics = evaluate_recommendations(recs, test_truth, total_items=len(all_items), k=k)
    metrics["model"] = "PopularityBaseline"
    pd.DataFrame([metrics]).to_csv(METRICS_DIR / "popularity_baseline_metrics.csv", index=False)
    with (METRICS_DIR / "popularity_baseline_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)
    return metrics


def run_full_pipeline(
    epochs: int = 50,
    embedding_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    seed: int = 42,
    k: int = K,
) -> pd.DataFrame:
    """Run baseline evaluation, PyG LightGCN training, evaluation, and reporting."""

    ensure_training_dirs()
    interactions, user_mapping, book_mapping, books = load_reduced_artifacts()
    train_df, val_df, test_df = split_interactions_per_user(interactions, seed=seed)
    save_splits(train_df, val_df, test_df)
    artifacts = PipelineArtifacts(user_mapping, book_mapping, train_df, val_df, test_df)

    metrics_rows = [run_popularity_baseline(artifacts, books, k=k)]

    model, _, embeddings = train_lightgcn_pyg(
        artifacts,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    del model
    train_seen = build_user_seen_items(train_df)
    test_truth = build_user_ground_truth(test_df)
    lightgcn_recs = generate_embedding_recommendations(
        embeddings,
        user_mapping,
        book_mapping,
        test_truth.keys(),
        train_seen,
        k=k,
    )
    save_recommendations("lightgcn", lightgcn_recs)
    lightgcn_metrics = evaluate_recommendations(lightgcn_recs, test_truth, total_items=len(book_mapping), k=k)
    lightgcn_metrics["model"] = "LightGCN"
    metrics_rows.append(lightgcn_metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_DIR / "model_comparison_metrics.csv", index=False)
    with (METRICS_DIR / "model_comparison_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(metrics_rows, output, indent=2)
    plot_metric_comparison(metrics_df)
    write_report(metrics_df)
    return metrics_df


def recommend(user_id: int, top_k: int = K) -> list[dict[str, object]]:
    """Return recommended books with scores for UI integration.

    The function uses saved LightGCN embeddings when available. Unknown users or
    missing embeddings fall back to the popularity baseline recommendations.
    """

    interactions, user_mapping, book_mapping, books = load_reduced_artifacts()
    books = books.drop_duplicates("ISBN").set_index("ISBN")
    train_path = SPLITS_DIR / "train_interactions.csv"
    train_df = pd.read_csv(train_path) if train_path.exists() else interactions
    seen_by_user = build_user_seen_items(train_df)

    embedding_path = EMBEDDINGS_DIR / "lightgcn_node_embeddings.pt"
    used_lightgcn = embedding_path.exists() and int(user_id) in set(user_mapping["User-ID"])
    if used_lightgcn:
        payload = torch.load(embedding_path, map_location="cpu")
        embeddings = payload["node_embeddings"]
        recs = generate_embedding_recommendations(
            embeddings,
            user_mapping,
            book_mapping,
            [int(user_id)],
            seen_by_user,
            k=top_k,
        ).get(int(user_id), [])
        user_node = int(user_mapping.set_index("User-ID").loc[int(user_id), "node_index"])
        book_lookup = book_mapping.set_index("ISBN")["node_index"].to_dict()
        scores = {
            isbn: float(torch.dot(embeddings[user_node], embeddings[int(book_lookup[isbn])]))
            for isbn in recs
        }
    else:
        all_items = book_mapping["ISBN"].astype(str).tolist()
        recs = popularity_recommendations(train_df, [int(user_id)], all_items, seen_by_user, top_k).get(int(user_id), [])
        popularity_scores = train_df["ISBN"].value_counts(normalize=True)
        scores = {isbn: float(popularity_scores.get(isbn, 0.0)) for isbn in recs}

    results = []
    for isbn in recs:
        metadata = books.loc[isbn] if isbn in books.index else {}
        year = metadata.get("Year-Of-Publication", "Unknown") if hasattr(metadata, "get") else "Unknown"
        results.append(
            {
                "ISBN": isbn,
                "Book-Title": metadata.get("Book-Title", "Unknown Title") if hasattr(metadata, "get") else "Unknown Title",
                "Book-Author": metadata.get("Book-Author", "Unknown Author") if hasattr(metadata, "get") else "Unknown Author",
                "Year-Of-Publication": str(year),
                "Publisher": metadata.get("Publisher", "Unknown") if hasattr(metadata, "get") else "Unknown",
                "Image-URL-L": metadata.get("Image-URL-L", "") if hasattr(metadata, "get") else "",
                "Score": float(scores.get(isbn, 0.0)),
                "Is-Fallback": not used_lightgcn,
            }
        )
    return results
