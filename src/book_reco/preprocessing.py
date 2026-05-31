"""Reusable preprocessing helpers for the Book-Crossing dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd

RAW_DATA_DIR = Path("data/raw-dataset-books")
PROCESSED_DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")


def load_raw_datasets(data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw Books, Ratings, and Users tables from disk."""

    books = pd.read_csv(data_dir / "Books.csv", low_memory=False)
    ratings = pd.read_csv(data_dir / "Ratings.csv", low_memory=False)
    users = pd.read_csv(data_dir / "Users.csv", low_memory=False)
    return books, ratings, users


def report_dataset_overview(datasets: Mapping[str, pd.DataFrame]) -> None:
    """Print the shape and column names for each dataset."""

    print("Dataset shapes:")
    for name, dataframe in datasets.items():
        print(f"{name}: {dataframe.shape}")

    print("\nColumn names:")
    for name, dataframe in datasets.items():
        print(f"{name}: {dataframe.columns.tolist()}")


def report_missing_values(datasets: Mapping[str, pd.DataFrame]) -> None:
    """Print missing-value counts for each dataset."""

    for name, dataframe in datasets.items():
        missing_values = dataframe.isna().sum()
        missing_values = missing_values[missing_values > 0]
        print(f"\n{name} missing values:")
        if missing_values.empty:
            print("No missing values found.")
        else:
            print(missing_values.sort_values(ascending=False))


def drop_duplicate_rows(books: pd.DataFrame, ratings: pd.DataFrame, users: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove duplicate rows from the three raw tables."""

    return books.drop_duplicates().copy(), ratings.drop_duplicates().copy(), users.drop_duplicates().copy()


def filter_explicit_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep only explicit ratings, where Book-Rating is greater than zero."""

    return ratings[ratings["Book-Rating"] > 0].copy()


def prepare_cleaned_ratings(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    minimum_user_ratings: int = 5,
    minimum_book_ratings: int = 5,
) -> pd.DataFrame:
    """Build the cleaned interaction table used for graph-based recommendation."""

    cleaned_ratings = ratings.merge(books, on="ISBN", how="inner")

    previous_shape: tuple[int, int] | None = None
    while previous_shape != cleaned_ratings.shape:
        previous_shape = cleaned_ratings.shape

        user_counts = cleaned_ratings["User-ID"].value_counts()
        active_users = user_counts[user_counts >= minimum_user_ratings].index
        cleaned_ratings = cleaned_ratings[cleaned_ratings["User-ID"].isin(active_users)].copy()

        book_counts = cleaned_ratings["ISBN"].value_counts()
        active_books = book_counts[book_counts >= minimum_book_ratings].index
        cleaned_ratings = cleaned_ratings[cleaned_ratings["ISBN"].isin(active_books)].copy()

    return cleaned_ratings


def build_summary_statistics(cleaned_ratings: pd.DataFrame) -> pd.DataFrame:
    """Create the summary statistics requested for the cleaned dataset."""

    return pd.DataFrame(
        {
            "Metric": [
                "Number of users",
                "Number of books",
                "Number of ratings",
                "Average rating",
            ],
            "Value": [
                cleaned_ratings["User-ID"].nunique(),
                cleaned_ratings["ISBN"].nunique(),
                len(cleaned_ratings),
                round(cleaned_ratings["Book-Rating"].mean(), 3),
            ],
        }
    )


def plot_preprocessing_visualizations(
    cleaned_ratings: pd.DataFrame,
    figures_dir: Path | None = FIGURES_DIR,
) -> None:
    """Render and optionally save the preprocessing charts."""

    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(cleaned_ratings["Book-Rating"], bins=10, color="#2A6F97", edgecolor="white")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(figures_dir / "rating_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    top_books = cleaned_ratings["Book-Title"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top_books.index[::-1], top_books.values[::-1], color="#A23B72")
    ax.set_title("Top 20 Most Rated Books")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Book Title")
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(figures_dir / "top_20_most_rated_books.png", dpi=150, bbox_inches="tight")
    plt.show()

    top_users = cleaned_ratings["User-ID"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top_users.index.astype(str)[::-1], top_users.values[::-1], color="#F18F01")
    ax.set_title("Top 20 Most Active Users")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("User ID")
    plt.tight_layout()
    if figures_dir is not None:
        fig.savefig(figures_dir / "top_20_most_active_users.png", dpi=150, bbox_inches="tight")
    plt.show()


def save_cleaned_ratings(cleaned_ratings: pd.DataFrame, output_path: Path = PROCESSED_DATA_DIR / "cleaned_ratings.csv") -> Path:
    """Save the cleaned interaction table and return the final file path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_ratings.to_csv(output_path, index=False)
    return output_path
