"""Build the LightGCN-ready graph from the cleaned Book-Crossing ratings."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.graph_construction import build_lightgcn_ready_graph, print_graph_summary  # noqa: E402


def main() -> None:
    """Run the graph-construction pipeline."""

    artifacts = build_lightgcn_ready_graph()
    print_graph_summary(artifacts)


if __name__ == "__main__":
    main()
