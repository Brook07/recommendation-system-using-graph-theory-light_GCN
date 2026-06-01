"""Graph construction helpers for LightGCN book recommendation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
CLEANED_RATINGS_PATH = PROCESSED_DIR / "cleaned_ratings.csv"


@dataclass
class GraphArtifacts:
    cleaned_ratings: pd.DataFrame
    user_mapping: pd.DataFrame
    book_mapping: pd.DataFrame
    edge_index: pd.DataFrame
    edge_weight: pd.DataFrame
    num_users: int
    num_books: int
    num_nodes: int
    num_edges: int
    average_degree: float
    graph_density: float


def load_cleaned_ratings(file_path: Path = CLEANED_RATINGS_PATH) -> pd.DataFrame:
    """Load the cleaned interaction table used for graph construction."""

    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")
    return pd.read_csv(file_path)


def create_node_mappings(cleaned_ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create contiguous integer IDs for users and books."""

    unique_users = sorted(cleaned_ratings["User-ID"].unique())
    unique_books = sorted(cleaned_ratings["ISBN"].unique())

    user_mapping = pd.DataFrame(
        {
            "User-ID": unique_users,
            "node_index": range(len(unique_users)),
            "node_type": "user",
        }
    )

    book_offset = len(unique_users)
    book_mapping = pd.DataFrame(
        {
            "ISBN": unique_books,
            "node_index": range(book_offset, book_offset + len(unique_books)),
            "node_type": "book",
        }
    )

    return user_mapping, book_mapping


def build_edge_tables(
    cleaned_ratings: pd.DataFrame,
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build PyG-style edge_index and edge_weight tables."""

    user_lookup = user_mapping.set_index("User-ID")["node_index"].to_dict()
    book_lookup = book_mapping.set_index("ISBN")["node_index"].to_dict()

    edges = cleaned_ratings[["User-ID", "ISBN", "Book-Rating"]].copy()
    edges["user_node"] = edges["User-ID"].map(user_lookup)
    edges["book_node"] = edges["ISBN"].map(book_lookup)

    forward_edges = pd.DataFrame({"source": edges["user_node"], "target": edges["book_node"]})
    reverse_edges = pd.DataFrame({"source": edges["book_node"], "target": edges["user_node"]})
    edge_index = pd.concat([forward_edges, reverse_edges], ignore_index=True)

    edge_weight = pd.DataFrame({"edge_weight": pd.concat([edges["Book-Rating"], edges["Book-Rating"]], ignore_index=True)})
    return edge_index, edge_weight


def compute_graph_statistics(
    cleaned_ratings: pd.DataFrame,
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
    edge_index: pd.DataFrame,
) -> tuple[int, int, int, int, float, float]:
    """Compute graph statistics for reporting and validation."""

    num_users = len(user_mapping)
    num_books = len(book_mapping)
    num_nodes = num_users + num_books
    num_edges = len(cleaned_ratings)
    average_degree = (2 * num_edges) / num_nodes
    graph_density = num_edges / (num_users * num_books)
    return num_users, num_books, num_nodes, num_edges, average_degree, graph_density


def save_tabular_outputs(
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
    edge_index: pd.DataFrame,
    edge_weight: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Save the graph tables for downstream training."""

    output_dir.mkdir(parents=True, exist_ok=True)
    user_mapping.to_csv(output_dir / "user_mapping.csv", index=False)
    book_mapping.to_csv(output_dir / "book_mapping.csv", index=False)
    edge_index.to_csv(output_dir / "edge_index.csv", index=False)
    edge_weight.to_csv(output_dir / "edge_weight.csv", index=False)


def save_lightgcn_tensors(
    edge_index: pd.DataFrame,
    edge_weight: pd.DataFrame,
    num_nodes: int,
    output_dir: Path = PROCESSED_DIR,
) -> Optional[Path]:
    """Save a PyTorch tensor artifact if torch is available."""

    if torch is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_index_tensor = torch.tensor(edge_index[["source", "target"]].to_numpy().T, dtype=torch.long)
    edge_weight_tensor = torch.tensor(edge_weight["edge_weight"].to_numpy(), dtype=torch.float)
    output_path = output_dir / "lightgcn_graph.pt"
    torch.save(
        {
            "edge_index": edge_index_tensor,
            "edge_weight": edge_weight_tensor,
            "num_nodes": num_nodes,
        },
        output_path,
    )
    return output_path


def plot_degree_distribution(edge_index: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> Path:
    """Save a degree-distribution plot for the bipartite graph."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    degree_series = pd.concat([edge_index["source"], edge_index["target"]]).value_counts()

    plt.figure(figsize=(10, 5))
    plt.hist(degree_series.values, bins=30, color="#2A6F97", edgecolor="white")
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()

    output_path = figures_dir / "graph_degree_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return output_path


def plot_sample_graph(
    cleaned_ratings: pd.DataFrame,
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
    max_users: int = 25,
    max_books: int = 25,
) -> Path:
    """Render a sampled bipartite graph for inspection."""

    figures_dir.mkdir(parents=True, exist_ok=True)

    sampled_users = user_mapping.head(max_users)
    sampled_books = book_mapping.head(max_books)
    sampled_df = cleaned_ratings[
        cleaned_ratings["User-ID"].isin(sampled_users["User-ID"])
        & cleaned_ratings["ISBN"].isin(sampled_books["ISBN"])
    ].copy()

    user_lookup = user_mapping.set_index("User-ID")["node_index"].to_dict()
    book_lookup = book_mapping.set_index("ISBN")["node_index"].to_dict()

    graph = nx.Graph()
    for _, row in sampled_users.iterrows():
        graph.add_node(f"u_{row['node_index']}", bipartite=0)
    for _, row in sampled_books.iterrows():
        graph.add_node(f"b_{row['node_index']}", bipartite=1)

    for _, row in sampled_df.iterrows():
        graph.add_edge(
            f"u_{user_lookup[row['User-ID']]}",
            f"b_{book_lookup[row['ISBN']]}",
            weight=row["Book-Rating"],
        )

    plt.figure(figsize=(12, 8))
    positions = nx.spring_layout(graph, seed=42, k=0.6)
    nx.draw_networkx_nodes(graph, positions, node_size=150, node_color="#4C78A8")
    nx.draw_networkx_edges(graph, positions, alpha=0.35)
    nx.draw_networkx_labels(graph, positions, font_size=7)
    plt.title("Sample Bipartite Graph")
    plt.axis("off")
    plt.tight_layout()

    output_path = figures_dir / "sample_bipartite_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return output_path


def build_lightgcn_ready_graph(file_path: Path = CLEANED_RATINGS_PATH) -> GraphArtifacts:
    """Build all graph artifacts needed for LightGCN training."""

    cleaned_ratings = load_cleaned_ratings(file_path)
    user_mapping, book_mapping = create_node_mappings(cleaned_ratings)
    edge_index, edge_weight = build_edge_tables(cleaned_ratings, user_mapping, book_mapping)
    num_users, num_books, num_nodes, num_edges, average_degree, graph_density = compute_graph_statistics(
        cleaned_ratings, user_mapping, book_mapping, edge_index
    )

    save_tabular_outputs(user_mapping, book_mapping, edge_index, edge_weight)
    save_lightgcn_tensors(edge_index, edge_weight, num_nodes)
    plot_degree_distribution(edge_index)
    plot_sample_graph(cleaned_ratings, user_mapping, book_mapping)

    return GraphArtifacts(
        cleaned_ratings=cleaned_ratings,
        user_mapping=user_mapping,
        book_mapping=book_mapping,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_users=num_users,
        num_books=num_books,
        num_nodes=num_nodes,
        num_edges=num_edges,
        average_degree=average_degree,
        graph_density=graph_density,
    )


def print_graph_summary(artifacts: GraphArtifacts) -> None:
    """Print a concise ready-for-training summary."""

    print("\nGraph Summary")
    print(f"Number of users: {artifacts.num_users}")
    print(f"Number of books: {artifacts.num_books}")
    print(f"Number of nodes: {artifacts.num_nodes}")
    print(f"Number of edges: {artifacts.num_edges}")
    print(f"Average degree: {artifacts.average_degree:.4f}")
    print(f"Graph density: {artifacts.graph_density:.6f}")
    print("\nGraph is ready for recommendation model training.")
