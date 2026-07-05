"""Simple Matrix Factorization (FunkSVD) baseline for Book-Crossing.

This module provides reusable functions to train a matrix-factorization model
using SGD and to evaluate ranking/accuracy metrics for comparison with GNNs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

PROCESSED_DIR = Path("data/processed")
MODEL_TRAINING_DIR = PROCESSED_DIR / "model_training"
METRICS_DIR = MODEL_TRAINING_DIR / "metrics"
PLOTS_DIR = MODEL_TRAINING_DIR / "plots"
SPLITS_DIR = MODEL_TRAINING_DIR / "splits"
RECOMMENDATIONS_DIR = MODEL_TRAINING_DIR / "recommendations"


@dataclass
class BaselineArtifacts:
    user_map: Dict[int, int]
    item_map: Dict[str, int]
    inv_user_map: Dict[int, int]
    inv_item_map: Dict[int, str]
    user_factors: np.ndarray
    item_factors: np.ndarray


def load_cleaned_ratings(path: Path = PROCESSED_DIR / "cleaned_ratings.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_test_split_per_user(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    train_rows = []
    test_rows = []
    for user, g in df.groupby("User-ID"):
        n = len(g)
        if n < 2:
            train_rows.append(g.index.values)
            continue
        k = max(1, int(np.floor(n * test_frac)))
        idx = g.index.values
        test_idx = np.random.choice(idx, size=k, replace=False)
        train_idx = np.setdiff1d(idx, test_idx)
        train_rows.extend(train_idx.tolist())
        test_rows.extend(test_idx.tolist())
    train_df = df.loc[train_rows].reset_index(drop=True)
    test_df = df.loc[test_rows].reset_index(drop=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "svd_baseline_train.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "svd_baseline_test.csv", index=False)
    return train_df, test_df


def build_mappings(train_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[str, int], Dict[int, int], Dict[int, str]]:
    users = sorted(train_df["User-ID"].unique())
    items = sorted(train_df["ISBN"].unique())
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: i for i, it in enumerate(items)}
    inv_user_map = {i: u for u, i in user_map.items()}
    inv_item_map = {i: it for it, i in item_map.items()}
    return user_map, item_map, inv_user_map, inv_item_map


class MatrixFactorization:
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64, lr: float = 0.01, reg: float = 0.02, epochs: int = 20, seed: int = 42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        rng = np.random.RandomState(seed)
        self.user_factors = 0.1 * rng.randn(n_users, n_factors)
        self.item_factors = 0.1 * rng.randn(n_items, n_factors)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0

    def fit(self, interactions: List[Tuple[int, int, float]]):
        self.global_bias = np.mean([r for (_, _, r) in interactions])
        for epoch in trange(self.epochs, desc="Training MF"):
            np.random.shuffle(interactions)
            for u, i, r in interactions:
                pred = self.predict_single_index(u, i)
                err = r - pred
                # update biases
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                # update latent factors
                uf = self.user_factors[u]
                vf = self.item_factors[i]
                self.user_factors[u] += self.lr * (err * vf - self.reg * uf)
                self.item_factors[i] += self.lr * (err * uf - self.reg * vf)
            # optionally decay lr or compute train loss
        return self

    def predict_single_index(self, u: int, i: int) -> float:
        return (
            self.global_bias + self.user_bias[u] + self.item_bias[i] + float(self.user_factors[u].dot(self.item_factors[i]))
        )

    def predict_user(self, u: int) -> np.ndarray:
        return self.global_bias + self.user_bias[u] + self.item_bias + self.item_factors.dot(self.user_factors[u])


def prepare_interactions(train_df: pd.DataFrame, user_map: Dict[int, int], item_map: Dict[str, int]) -> List[Tuple[int, int, float]]:
    interactions = []
    for _, row in train_df.iterrows():
        u = user_map.get(row["User-ID"])
        i = item_map.get(row["ISBN"])
        if u is None or i is None:
            continue
        interactions.append((u, i, float(row["Book-Rating"])))
    return interactions


def recommend_top_n(model: MatrixFactorization, user_map: Dict[int, int], inv_item_map: Dict[int, str], train_items_by_user: Dict[int, set], n: int = 10) -> Dict[int, List[str]]:
    recs = {}
    for u_original, u in user_map.items():
        pred_scores = model.predict_user(u)
        # mask train items
        seen = train_items_by_user.get(u_original, set())
        # map seen to indices
        seen_idx = set()
        for it in seen:
            idx = None
            try:
                idx = list(inv_item_map.keys())[list(inv_item_map.values()).index(it)]
            except ValueError:
                idx = None
            if idx is not None:
                seen_idx.add(idx)
        # get top indices
        candidate_idx = np.argsort(-pred_scores)
        top = []
        for idx in candidate_idx:
            if idx in seen_idx:
                continue
            top.append(inv_item_map[idx])
            if len(top) >= n:
                break
        recs[u_original] = top
    return recs


def precision_recall_ndcg_at_k(recommended: List[str], ground_truth: set, k: int = 10) -> Tuple[float, float, float]:
    recommended_k = recommended[:k]
    hits = [1 if r in ground_truth else 0 for r in recommended_k]
    precision = sum(hits) / k
    recall = sum(hits) / max(1, len(ground_truth))
    # DCG
    dcg = sum([h / np.log2(idx + 2) for idx, h in enumerate(hits)])
    # IDCG
    ideal_hits = [1] * min(len(ground_truth), k)
    idcg = sum([h / np.log2(idx + 2) for idx, h in enumerate(ideal_hits)])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def evaluate(model: MatrixFactorization, train_df: pd.DataFrame, test_df: pd.DataFrame, user_map: Dict[int, int], item_map: Dict[str, int], inv_item_map: Dict[int, str], k: int = 10) -> Dict[str, float]:
    # build train items by user
    train_items_by_user: Dict[int, set] = {}
    for _, row in train_df.iterrows():
        u = row["User-ID"]
        train_items_by_user.setdefault(u, set()).add(row["ISBN"])
    # test items by user
    test_items_by_user: Dict[int, set] = {}
    for _, row in test_df.iterrows():
        u = row["User-ID"]
        test_items_by_user.setdefault(u, set()).add(row["ISBN"])

    users = list(train_items_by_user.keys())
    precisions = []
    recalls = []
    ndcgs = []
    # RMSE on test
    sq_errs = []
    for u_original in users:
        if u_original not in user_map:
            continue
        u = user_map[u_original]
        # predict scores
        pred_scores = model.predict_user(u)
        # mask seen
        seen = train_items_by_user.get(u_original, set())
        seen_idx = set()
        for it in seen:
            if it in item_map:
                seen_idx.add(item_map[it])
        candidate_idx = np.argsort(-pred_scores)
        recs = []
        for idx in candidate_idx:
            if idx in seen_idx:
                continue
            recs.append(inv_item_map[idx])
            if len(recs) >= k:
                break
        gt = test_items_by_user.get(u_original, set())
        prec, rec, ndcg = precision_recall_ndcg_at_k(recs, gt, k)
        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)
    # RMSE over test interactions
    for _, row in test_df.iterrows():
        uo = row["User-ID"]
        io = row["ISBN"]
        if uo not in user_map or io not in item_map:
            continue
        u = user_map[uo]
        i = item_map[io]
        pred = model.predict_single_index(u, i)
        sq_errs.append((row["Book-Rating"] - pred) ** 2)
    rmse = float(np.sqrt(np.mean(sq_errs))) if sq_errs else float('nan')
    return {
        "Precision@10": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "NDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "RMSE": rmse,
    }


def save_metrics(metrics: Dict[str, float], output_path: Path = METRICS_DIR / "baseline_metrics.csv") -> Path:
    df = pd.DataFrame([metrics])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def plot_sample_recommendations(recommendations: Dict[int, List[str]], sample_users: List[int], figures_dir: Path = PLOTS_DIR) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for u in sample_users:
        recs = recommendations.get(u, [])
        rows.extend({"User-ID": u, "rank": rank, "ISBN": isbn} for rank, isbn in enumerate(recs, start=1))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(recs)), [1] * len(recs))
        ax.set_xticks(range(len(recs)))
        ax.set_xticklabels(recs, rotation=45, ha='right')
        ax.set_title(f"Top-{len(recs)} Recommendations for User {u}")
        plt.tight_layout()
        out = figures_dir / f"svd_sample_recs_user_{u}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(RECOMMENDATIONS_DIR / "svd_sample_recommendations.csv", index=False)
    return figures_dir
