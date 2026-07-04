"""Append the LightGCN data-reduction section to the preprocessing notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


NOTEBOOK_PATH = Path("notebooks/01_book_crossing_preprocessing.ipynb")
SECTION_MARKER = "## LightGCN Data Reduction"


MARKDOWN = """## LightGCN Data Reduction

The EDA showed that the full Book-Crossing interaction graph is extremely sparse and dominated by implicit zero ratings. For LightGCN with BPR ranking, retained edges should represent positive user-book preference signals rather than ambiguous non-ratings.

Reduction policy:

- Remove duplicate and invalid interaction records.
- Keep only ratings whose users and ISBNs still exist in the cleaned user/book tables.
- Treat `Book-Rating = 0` as implicit exposure or missing preference, not as a positive edge.
- Keep explicit positive interactions with `Book-Rating >= 5`.
- Compare candidate k-core thresholds before selecting the final graph.
- Use an iterative `user >= 10` and `book >= 10` k-core for the final LightGCN dataset.

The EDA suggested trying `book >= 20`, but the candidate analysis below shows that after removing implicit zero ratings this threshold removes the entire explicit-preference graph. The final `book >= 10` threshold is therefore used because it preserves a trainable graph while still requiring each retained book to have enough evidence for collaborative filtering.
"""


CODE = """from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "src").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reduce_book_crossing_for_lightgcn import main as reduce_for_lightgcn

# Rebuild the reduced graph-ready artifacts from the EDA-cleaned tables.
reduce_for_lightgcn()

processed_dir = REPO_ROOT / "data" / "processed"

reduction_stats = pd.read_csv(processed_dir / "data_reduction_statistics.csv")
candidate_stats = pd.read_csv(processed_dir / "threshold_candidate_analysis.csv")
graph_stats = pd.read_csv(processed_dir / "graph_statistics_reduced_lightgcn.csv")

display(reduction_stats)
display(candidate_stats)
display(graph_stats)

print("Final LightGCN-ready files:")
for filename in [
    "ratings_reduced_lightgcn.csv",
    "books_reduced_lightgcn.csv",
    "users_reduced_lightgcn.csv",
    "merged_reduced_lightgcn.csv",
    "cleaned_ratings.csv",
    "user_mapping.csv",
    "book_mapping.csv",
    "edge_index.csv",
    "edge_weight.csv",
    "lightgcn_graph.pt",
    "artifact_manifest_reduced_lightgcn.csv",
]:
    path = processed_dir / filename
    print(f"- {filename}: {path.exists()} ({path.stat().st_size if path.exists() else 0:,} bytes)")
"""


def main() -> None:
    """Update the notebook idempotently."""

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells = notebook["cells"]
    cells = [
        cell
        for cell in cells
        if SECTION_MARKER not in "".join(cell.get("source", ""))
        and "reduce_book_crossing_for_lightgcn" not in "".join(cell.get("source", ""))
    ]
    cells.extend([new_markdown_cell(MARKDOWN), new_code_cell(CODE)])
    notebook["cells"] = cells
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH} with the LightGCN data-reduction section.")


if __name__ == "__main__":
    main()
