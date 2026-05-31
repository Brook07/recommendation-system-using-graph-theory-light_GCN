"""Run the Book-Crossing preprocessing pipeline from the command line."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.preprocessing import (  # noqa: E402
    build_summary_statistics,
    drop_duplicate_rows,
    filter_explicit_ratings,
    load_raw_datasets,
    plot_preprocessing_visualizations,
    prepare_cleaned_ratings,
    report_dataset_overview,
    report_missing_values,
    save_cleaned_ratings,
)


def main() -> None:
    """Execute the preprocessing workflow and persist the cleaned CSV."""

    books, ratings, users = load_raw_datasets()
    datasets = {"Books": books, "Ratings": ratings, "Users": users}

    report_dataset_overview(datasets)
    report_missing_values(datasets)

    books, ratings, users = drop_duplicate_rows(books, ratings, users)
    print("\nShapes after removing duplicate rows:")
    for name, dataframe in {"Books": books, "Ratings": ratings, "Users": users}.items():
        print(f"{name}: {dataframe.shape}")

    ratings = filter_explicit_ratings(ratings)
    print("\nRatings shape after keeping explicit ratings only:", ratings.shape)
    print("Unique ratings remaining:", ratings["Book-Rating"].value_counts().sort_index().to_dict())

    cleaned_ratings = prepare_cleaned_ratings(books, ratings)
    print("Shape after removing sparse users and books:", cleaned_ratings.shape)

    summary_stats = build_summary_statistics(cleaned_ratings)
    print("\nSummary statistics:")
    print(summary_stats.to_string(index=False))

    plot_preprocessing_visualizations(cleaned_ratings)

    output_path = save_cleaned_ratings(cleaned_ratings)
    print(f"Cleaned dataset saved to: {output_path.resolve()}")
    print("\nFinal cleaned dataset shape:", cleaned_ratings.shape)
    print(cleaned_ratings.head())


if __name__ == "__main__":
    main()
