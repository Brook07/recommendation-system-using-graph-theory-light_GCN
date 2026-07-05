"""Inference wrapper used by the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List

from src.book_reco.recommender_pipeline import recommend


class InferenceEngine:
    """Serve recommendations from saved LightGCN embeddings or popularity fallback."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k recommendations for a user."""

        return recommend(user_id=user_id, top_k=top_k)


if __name__ == "__main__":
    engine = InferenceEngine()
    print(engine.get_recommendations(11676))
