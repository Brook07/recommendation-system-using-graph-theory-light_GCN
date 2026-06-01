"""LightGCN implementation for Book-Crossing using PyTorch (no PyG dependency required).

This module builds a bipartite graph from mappings, constructs the normalized
adjacency as a sparse tensor, implements LightGCN propagation, trains with BPR
loss and negative sampling, and provides evaluation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
CLEANED_PATH = PROCESSED_DIR / "cleaned_ratings.csv"
USER_MAP_PATH = PROCESSED_DIR / "user_mapping.csv"
BOOK_MAP_PATH = PROCESSED_DIR / "book_mapping.csv"
EDGE_INDEX_PATH = PROCESSED_DIR / "edge_index.csv"
EDGE_WEIGHT_PATH = PROCESSED_DIR / "edge_weight.csv"


@dataclass
class LightGCNArtifacts:
    model_state: Dict
    metrics: Dict[str, float]
    training_losses: List[float]
    embedding_dim: int


def load_cleaned_interactions(path: Path = CLEANED_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_mappings(user_map_path: Path = USER_MAP_PATH, book_map_path: Path = BOOK_MAP_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_map = pd.read_csv(user_map_path)
    book_map = pd.read_csv(book_map_path)
    return user_map, book_map


def build_adjacency(num_nodes: int, edge_index_path: Path = EDGE_INDEX_PATH) -> torch.sparse.FloatTensor:
    df = pd.read_csv(edge_index_path)
    # ensure columns 'source','target'
    src = torch.LongTensor(df['source'].to_numpy())
    tgt = torch.LongTensor(df['target'].to_numpy())
    indices = torch.stack([src, tgt], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
    # make symmetric (should already be symmetric) and normalize by D^-0.5 * A * D^-0.5
    adj = adj.coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    # compute normalization: for each nonzero (i,j) value v, multiply by deg_inv_sqrt[i]*deg_inv_sqrt[j]
    rows = adj.indices()[0, :]
    cols = adj.indices()[1, :]
    norm_vals = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
    adj_norm = torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.size())
    return adj_norm.coalesce()


class LightGCNModel(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, n_layers: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        # perform propagation and return final embeddings (num_nodes x emb_dim)
        all_embeddings = [self.embedding.weight]
        emb = self.embedding.weight
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(adj, emb)
            all_embeddings.append(emb)
        all_embeddings = torch.stack(all_embeddings, dim=1)  # num_nodes x (n_layers+1) x emb_dim
        final_emb = torch.mean(all_embeddings, dim=1)
        return final_emb


def split_interactions(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    train_idx = []
    val_idx = []
    test_idx = []
    for _, g in df.groupby('User-ID'):
        idx = g.index.values
        n = len(idx)
        if n < 3:
            # put all in train
            train_idx.extend(idx.tolist())
            continue
        # shuffle
        perm = np.random.permutation(idx)
        n_test = max(1, int(math.floor(n * test_frac)))
        n_val = max(1, int(math.floor(n * val_frac)))
        test = perm[:n_test]
        val = perm[n_test:n_test + n_val]
        train = perm[n_test + n_val:]
        if len(train) == 0:
            # ensure at least one in train
            train = np.append(train, val[0])
            val = val[1:]
        train_idx.extend(train.tolist())
        val_idx.extend(val.tolist())
        test_idx.extend(test.tolist())
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def bpr_loss(u_emb: torch.Tensor, i_emb: torch.Tensor, j_emb: torch.Tensor) -> torch.Tensor:
    # u_emb, i_emb, j_emb: batch x emb_dim
    x_ui = torch.sum(u_emb * i_emb, dim=1)
    x_uj = torch.sum(u_emb * j_emb, dim=1)
    xuij = x_ui - x_uj
    loss = -torch.mean(torch.log(torch.sigmoid(xuij) + 1e-12))
    return loss


def train_lightgcn(
    emb_dim: int = 64,
    n_layers: int = 3,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 0.01,
    reg: float = 1e-4,
    device: str = 'cpu',
) -> LightGCNArtifacts:
    df = load_cleaned_interactions()
    user_map_df, book_map_df = load_mappings()
    user_map = dict(zip(user_map_df['User-ID'].to_list(), user_map_df['node_index'].to_list()))
    book_map = dict(zip(book_map_df['ISBN'].to_list(), book_map_df['node_index'].to_list()))
    num_nodes = int(user_map_df['node_index'].max() + 1 + (book_map_df['node_index'].max() - user_map_df['node_index'].max()))
    # build interactions in node indices (user_node, book_node)
    interactions = df[['User-ID', 'ISBN', 'Book-Rating']].copy()
    interactions['user_node'] = interactions['User-ID'].map(user_map)
    interactions['item_node'] = interactions['ISBN'].map(book_map)
    interactions = interactions.dropna(subset=['user_node', 'item_node']).astype({'user_node': int, 'item_node': int})

    train_df, val_df, test_df = split_interactions(interactions, val_frac=0.1, test_frac=0.1)

    # build adjacency from saved edge_index
    adj = build_adjacency(num_nodes)
    adj = adj.coalesce().to(device)

    model = LightGCNModel(num_nodes, emb_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    # prepare user->train items mapping for negative sampling
    train_user_items: Dict[int, set] = {}
    for _, row in train_df.iterrows():
        train_user_items.setdefault(int(row['User-ID']), set()).add(int(row['item_node']))

    all_items = np.array(sorted(book_map.values()))

    training_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        final_embeddings = model(adj)  # num_nodes x emb_dim
        # derive user and item embeddings
        user_embeddings = final_embeddings[[int(x) for x in user_map.values()], :]
        item_embeddings = final_embeddings[[int(x) for x in book_map.values()], :]

        # create positive pairs list
        pos_pairs = list(zip(train_df['User-ID'].astype(int).to_list(), train_df['item_node'].astype(int).to_list()))
        np.random.shuffle(pos_pairs)

        # batch training using BPR
        for start in range(0, len(pos_pairs), batch_size):
            batch = pos_pairs[start:start + batch_size]
            batch_u = []
            batch_i = []
            batch_j = []
            for (u_orig, i_node) in batch:
                u_node = user_map[u_orig]
                # negative sample from all_items excluding user's train items
                neg = None
                while True:
                    candidate = np.random.choice(all_items)
                    if candidate not in train_user_items.get(u_orig, set()):
                        neg = candidate
                        break
                batch_u.append(u_node)
                batch_i.append(i_node)
                batch_j.append(neg)
            batch_u = torch.LongTensor(batch_u).to(device)
            batch_i = torch.LongTensor(batch_i).to(device)
            batch_j = torch.LongTensor(batch_j).to(device)

            final_embeddings = model(adj)  # recompute so gradients flow
            u_emb = final_embeddings[batch_u]
            i_emb = final_embeddings[batch_i]
            j_emb = final_embeddings[batch_j]

            loss = bpr_loss(u_emb, i_emb, j_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        avg_loss = epoch_loss / max(1, len(pos_pairs))
        training_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")

    # evaluation on test
    model.eval()
    with torch.no_grad():
        final_embeddings = model(adj)
        # build inverted maps
        inv_item_map = {v: k for k, v in book_map.items()}
        inv_user_map = {v: k for k, v in user_map.items()}

        # user indices for evaluation: use users that appear in test
        eval_users = sorted(set(test_df['User-ID'].astype(int).to_list()))
        precision_list = []
        recall_list = []
        ndcg_list = []
        for u_orig in eval_users:
            u_node = user_map.get(u_orig, None)
            if u_node is None:
                continue
            u_emb = final_embeddings[u_node]
            scores = torch.matmul(final_embeddings, u_emb)
            # item scores: restrict to item indices
            item_indices = np.array(sorted(book_map.values()))
            item_scores = scores[item_indices]
            # mask train items
            train_seen = train_user_items.get(u_orig, set())
            mask = np.zeros(len(item_indices), dtype=bool)
            for idx_i, item_node in enumerate(item_indices):
                if item_node in train_seen:
                    mask[idx_i] = True
            item_scores_np = item_scores.cpu().numpy()
            item_scores_np[mask] = -np.inf
            topk_idx = np.argpartition(-item_scores_np, 10)[:10]
            topk = item_indices[topk_idx]
            topk_isbns = [inv_item_map[int(x)] for x in topk]
            # ground truth
            # ground truth as item_node indices (not raw ISBN strings)
            gt_nodes = set(test_df[test_df['User-ID'] == u_orig]['item_node'].astype(int).to_list())
            # compute precision/recall/ndcg using node ids
            hits = [1 if int(x) in gt_nodes else 0 for x in topk]
            prec = sum(hits) / 10.0
            rec = sum(hits) / max(1, len(gt_nodes))
            dcg = sum([h / math.log2(i + 2) for i, h in enumerate(hits)])
            idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(gt_nodes), 10))]) if len(gt_nodes) > 0 else 0.0
            ndcg = dcg / idcg if idcg > 0 else 0.0
            precision_list.append(prec)
            recall_list.append(rec)
            ndcg_list.append(ndcg)

        metrics = {
            'Precision@10': float(np.mean(precision_list)) if precision_list else 0.0,
            'Recall@10': float(np.mean(recall_list)) if recall_list else 0.0,
            'NDCG@10': float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        }

    # save model and metrics
    out_model = PROCESSED_DIR / f'lightgcn_model_dim{emb_dim}.pt'
    torch.save({'model_state_dict': model.state_dict(), 'emb_dim': emb_dim}, out_model)
    out_metrics = PROCESSED_DIR / f'lightgcn_metrics_dim{emb_dim}.csv'
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False)
    # save training loss plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('BPR Loss')
    plt.title(f'Training Loss (dim={emb_dim})')
    loss_path = FIGURES_DIR / f'training_loss_dim{emb_dim}.png'
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')

    return LightGCNArtifacts(model_state={'path': str(out_model)}, metrics=metrics, training_losses=training_losses, embedding_dim=emb_dim)


def recommend_topk_from_model(artifact: LightGCNArtifacts, topk: int = 10) -> Dict[int, List[str]]:
    # load model state
    model_file = Path(artifact.model_state['path'])
    checkpoint = torch.load(model_file)
    emb_dim = checkpoint.get('emb_dim')
    # rebuild model
    user_map_df, book_map_df = load_mappings()
    num_nodes = int(user_map_df['node_index'].max() + 1 + (book_map_df['node_index'].max() - user_map_df['node_index'].max()))
    model = LightGCNModel(num_nodes, emb_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    adj = build_adjacency(num_nodes)
    final_embeddings = model(adj)
    book_map = dict(zip(book_map_df['ISBN'].to_list(), book_map_df['node_index'].to_list()))
    inv_item_map = {v: k for k, v in book_map.items()}

    # sample 5 users
    sample_users = list(user_map_df['User-ID'].head(5).to_list())
    recs = {}
    for u_orig in sample_users:
        u_node = int(user_map_df[user_map_df['User-ID'] == u_orig]['node_index'].iloc[0])
        u_emb = final_embeddings[u_node]
        scores = torch.matmul(final_embeddings, u_emb)
        item_indices = np.array(sorted(book_map.values()))
        item_scores = scores[item_indices].cpu().detach().numpy()
        topk_idx = np.argsort(-item_scores)[:topk]
        topk_nodes = item_indices[topk_idx]
        recs[u_orig] = [inv_item_map[int(x)] for x in topk_nodes]
    return recs
