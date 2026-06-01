"""Run the SVD baseline and save evaluation metrics."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.baseline_svd import (
    BaselineArtifacts,
    MatrixFactorization,
    build_mappings,
    evaluate,
    load_cleaned_ratings,
    plot_sample_recommendations,
    prepare_interactions,
    recommend_top_n,
    save_metrics,
    train_test_split_per_user,
    train_test_split_per_user as split_func,
    build_mappings as mapping_func,
)


def main() -> None:
    cleaned = load_cleaned_ratings()
    train_df, test_df = train_test_split_per_user(cleaned, test_frac=0.2, seed=42)
    user_map, item_map, inv_user_map, inv_item_map = build_mappings(train_df)

    interactions = prepare_interactions(train_df, user_map, item_map)

    model = MatrixFactorization(n_users=len(user_map), n_items=len(item_map), n_factors=64, lr=0.01, reg=0.02, epochs=10)
    model.fit(interactions)

    # prepare train items per user for masking
    train_items_by_user = {}
    for _, row in train_df.iterrows():
        train_items_by_user.setdefault(row["User-ID"], set()).add(row["ISBN"])

    # recommendations
    # adapt recommend_top_n signature
    recs = recommend_top_n(model, user_map, inv_item_map, train_items_by_user, n=10)

    # evaluate
    metrics = evaluate(model, train_df, test_df, user_map, item_map, inv_item_map, k=10)
    out = save_metrics(metrics)
    print(f"Saved baseline metrics to: {out.resolve()}")

    # display sample users
    sample_users = list(train_items_by_user.keys())[:5]
    for u in sample_users:
        print(f"\nTop-10 for user {u}: {recs.get(u, [])}")

    plot_sample_recommendations(recs, sample_users)


if __name__ == "__main__":
    main()
