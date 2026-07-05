"""Reduce the EDA-cleaned Book-Crossing data for LightGCN training."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.graph_construction import (  # noqa: E402
    compute_graph_statistics,
    create_node_mappings,
    save_lightgcn_tensors,
    save_tabular_outputs,
)


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MIN_USER_INTERACTIONS = 10
MIN_BOOK_INTERACTIONS = 10
MIN_POSITIVE_RATING = 5


def graph_stats(stage: str, interactions: pd.DataFrame) -> dict[str, float | int | str]:
    """Return recommender graph statistics for a user-book interaction table."""

    users = interactions["User-ID"].nunique()
    books = interactions["ISBN"].nunique()
    ratings = len(interactions)
    possible_edges = users * books
    density = ratings / possible_edges if possible_edges else 0.0
    return {
        "stage": stage,
        "users": users,
        "books": books,
        "ratings": ratings,
        "graph_density": density,
        "sparsity": 1 - density if possible_edges else 1.0,
        "avg_user_degree": ratings / users if users else 0.0,
        "avg_book_degree": ratings / books if books else 0.0,
        "implicit_zero_ratings": int((interactions["Book-Rating"] == 0).sum())
        if "Book-Rating" in interactions
        else 0,
        "explicit_ratings": int((interactions["Book-Rating"] > 0).sum())
        if "Book-Rating" in interactions
        else 0,
    }


def iterative_k_core(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_book_interactions: int,
) -> tuple[pd.DataFrame, int]:
    """Iteratively retain users and books that satisfy the interaction thresholds."""

    reduced = interactions.copy()
    previous_shape: tuple[int, int] | None = None
    iterations = 0

    while previous_shape != reduced.shape:
        previous_shape = reduced.shape
        iterations += 1

        user_counts = reduced["User-ID"].value_counts()
        retained_users = user_counts[user_counts >= min_user_interactions].index
        reduced = reduced[reduced["User-ID"].isin(retained_users)].copy()

        book_counts = reduced["ISBN"].value_counts()
        retained_books = book_counts[book_counts >= min_book_interactions].index
        reduced = reduced[reduced["ISBN"].isin(retained_books)].copy()

    return reduced.reset_index(drop=True), iterations


def build_candidate_table(base_interactions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate candidate thresholds so the final choice is evidence-based."""

    rows: list[dict[str, float | int | str]] = []
    for minimum_rating in [1, MIN_POSITIVE_RATING, 6]:
        positive = base_interactions[base_interactions["Book-Rating"] >= minimum_rating].copy()
        for min_user, min_book in [(5, 10), (10, 10), (10, 20)]:
            reduced, iterations = iterative_k_core(positive, min_user, min_book)
            stats = graph_stats(
                f"rating>={minimum_rating}, user>={min_user}, book>={min_book}",
                reduced,
            )
            stats.update(
                {
                    "min_positive_rating": minimum_rating,
                    "min_user_interactions": min_user,
                    "min_book_interactions": min_book,
                    "k_core_iterations": iterations,
                }
            )
            rows.append(stats)
    return pd.DataFrame(rows)


def build_bidirectional_edges(
    interactions: pd.DataFrame,
    user_mapping: pd.DataFrame,
    book_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build LightGCN edge tables using binary positive feedback weights."""

    user_lookup = user_mapping.set_index("User-ID")["node_index"].to_dict()
    book_lookup = book_mapping.set_index("ISBN")["node_index"].to_dict()

    edges = interactions[["User-ID", "ISBN"]].copy()
    edges["user_node"] = edges["User-ID"].map(user_lookup)
    edges["book_node"] = edges["ISBN"].map(book_lookup)

    forward_edges = pd.DataFrame({"source": edges["user_node"], "target": edges["book_node"]})
    reverse_edges = pd.DataFrame({"source": edges["book_node"], "target": edges["user_node"]})
    edge_index = pd.concat([forward_edges, reverse_edges], ignore_index=True)
    edge_weight = pd.DataFrame({"edge_weight": 1.0}, index=edge_index.index)
    return edge_index, edge_weight


def main() -> None:
    """Create the reduced, graph-ready LightGCN dataset."""

    books = pd.read_csv(PROCESSED_DIR / "books_clean_eda.csv", low_memory=False)
    users = pd.read_csv(PROCESSED_DIR / "users_clean_eda.csv", low_memory=False)
    ratings = pd.read_csv(PROCESSED_DIR / "ratings_clean_eda.csv", low_memory=False)

    stats_rows = [graph_stats("eda_cleaned_ratings_all_valid_scores", ratings)]

    valid = ratings.drop_duplicates(subset=["User-ID", "ISBN", "Book-Rating"]).copy()
    valid = valid[valid["Book-Rating"].between(0, 10)].copy()
    valid = valid[
        valid["User-ID"].isin(users["User-ID"])
        & valid["ISBN"].isin(books["ISBN"])
    ].copy()
    stats_rows.append(graph_stats("metadata_matched_valid_unique", valid))

    positive = valid[valid["Book-Rating"] >= MIN_POSITIVE_RATING].copy()
    positive = positive.drop_duplicates(subset=["User-ID", "ISBN"], keep="last")
    stats_rows.append(graph_stats("explicit_positive_rating_ge_5", positive))

    candidate_table = build_candidate_table(valid)
    reduced, iterations = iterative_k_core(
        positive,
        MIN_USER_INTERACTIONS,
        MIN_BOOK_INTERACTIONS,
    )
    reduced = reduced.sort_values(["User-ID", "ISBN"]).reset_index(drop=True)
    final_stats = graph_stats("final_lightgcn_k_core", reduced)
    final_stats["k_core_iterations"] = iterations
    stats_rows.append(final_stats)

    books_reduced = books[books["ISBN"].isin(reduced["ISBN"])].copy()
    users_reduced = users[users["User-ID"].isin(reduced["User-ID"])].copy()
    merged_reduced = (
        reduced.merge(books_reduced, on="ISBN", how="left")
        .merge(
            users_reduced[["User-ID", "Location", "Age", "Age_Clean", "Country"]],
            on="User-ID",
            how="left",
        )
    )

    final_interactions = reduced.rename(columns={"Book-Rating": "original_book_rating"})
    final_interactions["interaction"] = 1
    final_interactions["implicit_source"] = False
    final_interactions["preference_label"] = "explicit_positive"

    user_mapping, book_mapping = create_node_mappings(final_interactions)
    edge_index, edge_weight = build_bidirectional_edges(final_interactions, user_mapping, book_mapping)
    num_users, num_books, num_nodes, num_edges, average_degree, graph_density = compute_graph_statistics(
        final_interactions,
        user_mapping,
        book_mapping,
        edge_index,
    )

    graph_summary = pd.DataFrame(
        {
            "metric": [
                "user_nodes",
                "book_nodes",
                "total_nodes",
                "interaction_edges",
                "directed_lightgcn_edges",
                "average_degree",
                "graph_density",
                "sparsity",
                "min_user_interactions",
                "min_book_interactions",
                "min_positive_rating",
            ],
            "value": [
                num_users,
                num_books,
                num_nodes,
                num_edges,
                len(edge_index),
                average_degree,
                graph_density,
                1 - graph_density,
                MIN_USER_INTERACTIONS,
                MIN_BOOK_INTERACTIONS,
                MIN_POSITIVE_RATING,
            ],
        }
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    books_reduced.to_csv(PROCESSED_DIR / "books_reduced_lightgcn.csv", index=False)
    users_reduced.to_csv(PROCESSED_DIR / "users_reduced_lightgcn.csv", index=False)
    final_interactions.to_csv(PROCESSED_DIR / "ratings_reduced_lightgcn.csv", index=False)
    merged_reduced.to_csv(PROCESSED_DIR / "merged_reduced_lightgcn.csv", index=False)
    candidate_table.to_csv(PROCESSED_DIR / "threshold_candidate_analysis.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(PROCESSED_DIR / "data_reduction_statistics.csv", index=False)
    graph_summary.to_csv(PROCESSED_DIR / "graph_statistics_reduced_lightgcn.csv", index=False)

    # Canonical artifacts consumed by the current graph construction and training scripts.
    final_interactions.rename(columns={"original_book_rating": "Book-Rating"})[
        ["User-ID", "ISBN", "Book-Rating"]
    ].to_csv(PROCESSED_DIR / "cleaned_ratings.csv", index=False)
    save_tabular_outputs(user_mapping, book_mapping, edge_index, edge_weight, output_dir=PROCESSED_DIR)
    save_lightgcn_tensors(edge_index, edge_weight, num_nodes, output_dir=PROCESSED_DIR)

    user_mapping.to_csv(PROCESSED_DIR / "user_mapping_reduced_lightgcn.csv", index=False)
    book_mapping.to_csv(PROCESSED_DIR / "book_mapping_reduced_lightgcn.csv", index=False)
    edge_index.to_csv(PROCESSED_DIR / "edge_index_reduced_lightgcn.csv", index=False)
    edge_weight.to_csv(PROCESSED_DIR / "edge_weight_reduced_lightgcn.csv", index=False)

    artifact_names = [
        "books_reduced_lightgcn.csv",
        "users_reduced_lightgcn.csv",
        "ratings_reduced_lightgcn.csv",
        "merged_reduced_lightgcn.csv",
        "threshold_candidate_analysis.csv",
        "data_reduction_statistics.csv",
        "graph_statistics_reduced_lightgcn.csv",
        "cleaned_ratings.csv",
        "user_mapping.csv",
        "book_mapping.csv",
        "edge_index.csv",
        "edge_weight.csv",
        "lightgcn_graph.pt",
        "user_mapping_reduced_lightgcn.csv",
        "book_mapping_reduced_lightgcn.csv",
        "edge_index_reduced_lightgcn.csv",
        "edge_weight_reduced_lightgcn.csv",
    ]
    pd.DataFrame(
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
    ).to_csv(PROCESSED_DIR / "artifact_manifest_reduced_lightgcn.csv", index=False)

    print("Threshold candidate analysis:")
    print(candidate_table.to_string(index=False))
    print("\nBefore/after reduction statistics:")
    print(pd.DataFrame(stats_rows).to_string(index=False))
    print("\nFinal graph summary:")
    print(graph_summary.to_string(index=False))


if __name__ == "__main__":
    main()
