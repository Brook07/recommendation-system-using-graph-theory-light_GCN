"""Export EDA-derived graph metadata and a reusable NetworkX graph artifact."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


PROCESSED_DIR = Path("data/processed")


def main() -> None:
    """Write graph statistics, an artifact manifest, and a pickled graph."""

    ratings = pd.read_csv(PROCESSED_DIR / "ratings_clean_eda.csv", low_memory=False)
    user_mapping = pd.read_csv(PROCESSED_DIR / "user_mapping.csv")
    book_mapping = pd.read_csv(PROCESSED_DIR / "book_mapping.csv")
    edge_index = pd.read_csv(PROCESSED_DIR / "edge_index.csv")

    num_users = len(user_mapping)
    num_books = len(book_mapping)
    num_nodes = num_users + num_books
    num_interactions = len(ratings)
    average_degree = (2 * num_interactions) / num_nodes
    graph_density = num_interactions / (num_users * num_books)

    stats = pd.DataFrame(
        {
            "metric": [
                "user_nodes",
                "book_nodes",
                "total_nodes",
                "interaction_edges",
                "directed_lightgcn_edges",
                "average_degree",
                "graph_density",
                "implicit_zero_edges",
                "explicit_rating_edges",
            ],
            "value": [
                num_users,
                num_books,
                num_nodes,
                num_interactions,
                len(edge_index),
                average_degree,
                graph_density,
                int((ratings["Book-Rating"] == 0).sum()),
                int((ratings["Book-Rating"] > 0).sum()),
            ],
        }
    )
    stats.to_csv(PROCESSED_DIR / "graph_statistics_eda.csv", index=False)

    graph = nx.Graph()
    for row in user_mapping.itertuples(index=False):
        graph.add_node(
            f"user_{row[0]}",
            node_type="user",
            user_id=int(row[0]),
            node_index=int(row.node_index),
        )

    for row in book_mapping.itertuples(index=False):
        graph.add_node(
            f"book_{row.ISBN}",
            node_type="book",
            isbn=str(row.ISBN),
            node_index=int(row.node_index),
        )

    for row in ratings.itertuples(index=False):
        graph.add_edge(
            f"user_{row[0]}",
            f"book_{row.ISBN}",
            weight=int(row[2]),
        )

    graph_path = PROCESSED_DIR / "book_bipartite_graph.gpickle"
    with graph_path.open("wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)

    artifact_names = [
        "books_clean_eda.csv",
        "users_clean_eda.csv",
        "ratings_clean_eda.csv",
        "merged_eda.csv",
        "cleaned_ratings.csv",
        "cleaned_ratings_from_eda.csv",
        "user_mapping.csv",
        "book_mapping.csv",
        "edge_index.csv",
        "edge_weight.csv",
        "lightgcn_graph.pt",
        "graph_statistics_eda.csv",
        "book_bipartite_graph.gpickle",
    ]
    manifest = pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(PROCESSED_DIR / name),
                "exists": (PROCESSED_DIR / name).exists(),
                "size_bytes": (PROCESSED_DIR / name).stat().st_size
                if (PROCESSED_DIR / name).exists()
                else None,
            }
            for name in artifact_names
        ]
    )
    manifest.to_csv(PROCESSED_DIR / "artifact_manifest_eda.csv", index=False)

    print(stats.to_string(index=False))
    print(
        "Saved NetworkX graph:",
        graph_path,
        f"nodes={graph.number_of_nodes()}",
        f"edges={graph.number_of_edges()}",
    )


if __name__ == "__main__":
    main()
