import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import logging
from collections import defaultdict
import math

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim, normalized_adj, n_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        
        self.adj = normalized_adj

    def forward(self):
        emb = self.embedding.weight
        embs = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.adj, emb)
            embs.append(emb)
            
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        return final_emb, self.embedding.weight

def bpr_loss(users, pos_items, neg_items, final_emb, initial_emb, reg_weight=1e-4):
    user_emb = final_emb[users]
    pos_emb = final_emb[pos_items]
    neg_emb = final_emb[neg_items]
    
    pos_scores = (user_emb * pos_emb).sum(dim=1)
    neg_scores = (user_emb * neg_emb).sum(dim=1)
    
    bpr = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    
    # Regularization
    user_emb0 = initial_emb[users]
    pos_emb0 = initial_emb[pos_items]
    neg_emb0 = initial_emb[neg_items]
    
    reg_loss = (1/2) * (user_emb0.norm(2).pow(2) + 
                        pos_emb0.norm(2).pow(2) + 
                        neg_emb0.norm(2).pow(2)) / float(len(users))
    
    return bpr + reg_weight * reg_loss

def split_interactions(edge_index, num_users):
    # edge_index has shape [2, E] and represents undirected edges (u->i and i->u)
    # Filter to only keep user->item edges (row 0 is user, row 1 is item)
    mask = edge_index[0] < num_users
    u_i_edges = edge_index[:, mask].cpu().numpy().T # shape [E/2, 2]
    
    np.random.seed(42)
    np.random.shuffle(u_i_edges)
    
    n_edges = len(u_i_edges)
    n_train = int(n_edges * 0.8)
    n_val = int(n_edges * 0.1)
    
    train_edges = u_i_edges[:n_train]
    val_edges = u_i_edges[n_train:n_train+n_val]
    test_edges = u_i_edges[n_train+n_val:]
    
    return train_edges, val_edges, test_edges

def generate_negative_samples(users, num_items, item_offset, train_interact_dict):
    neg_items = []
    for u in users:
        while True:
            neg_i = np.random.randint(0, num_items) + item_offset
            if neg_i not in train_interact_dict[u]:
                neg_items.append(neg_i)
                break
    return np.array(neg_items)

def compute_metrics(val_edges, train_dict, final_emb, item_offset, num_items, k=10, device='cpu'):
    # Build validation dict
    val_dict = defaultdict(list)
    for u, i in val_edges:
        val_dict[u].append(i)
        
    users = list(val_dict.keys())
    
    # We evaluate in batches to avoid OOM
    batch_size = 1024
    recalls = []
    precisions = []
    ndcgs = []
    
    for i in range(0, len(users), batch_size):
        batch_users = users[i:i+batch_size]
        batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        
        user_emb = final_emb[batch_users_tensor]
        item_emb = final_emb[item_offset:item_offset+num_items]
        
        # scores: [batch_size, num_items]
        scores = torch.matmul(user_emb, item_emb.T)
        
        # Mask out training items
        for idx, u in enumerate(batch_users):
            train_items = train_dict.get(u, [])
            if train_items:
                # adjust indices to 0-based for item_emb
                train_items_adj = [ti - item_offset for ti in train_items]
                scores[idx, train_items_adj] = -1e9
                
        _, top_k_indices = torch.topk(scores, k=k, dim=1)
        top_k_indices = top_k_indices.cpu().numpy() + item_offset # restore global indices
        
        for idx, u in enumerate(batch_users):
            ground_truth = set(val_dict[u])
            preds = top_k_indices[idx]
            
            hits = [1 if p in ground_truth else 0 for p in preds]
            hit_count = sum(hits)
            
            recalls.append(hit_count / len(ground_truth))
            precisions.append(hit_count / k)
            
            # NDCG
            dcg = sum(h / math.log2(rank + 2) for rank, h in enumerate(hits))
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(ground_truth))))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
    return np.mean(recalls), np.mean(precisions), np.mean(ndcgs)

def main():
    BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
    GRAPH_DIR = BASE_DIR / "data" / "processed-graph"
    OUTPUT_DIR = BASE_DIR / "models" / "lightgcn"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"{Fore.CYAN}Using device: {device}{Style.RESET_ALL}")
    
    # 1. Load Datasets
    logging.info(f"{Fore.CYAN}Loading graph datasets...{Style.RESET_ALL}")
    edge_index = torch.load(GRAPH_DIR / "edge_index.pt")
    
    # Load normalized adj
    adj_data = torch.load(GRAPH_DIR / "normalized_adj.pt")
    normalized_adj = torch.sparse_coo_tensor(
        adj_data['indices'], 
        adj_data['values'], 
        adj_data['shape']
    ).to(device)
    
    user_mapping = pd.read_csv(GRAPH_DIR / "user_mapping.csv")
    book_mapping = pd.read_csv(GRAPH_DIR / "book_mapping.csv")
    
    num_users = len(user_mapping)
    num_books = len(book_mapping)
    num_nodes = num_users + num_books
    
    # 2. Split interactions
    logging.info(f"{Fore.CYAN}Splitting interactions into train/val/test...{Style.RESET_ALL}")
    train_edges, val_edges, test_edges = split_interactions(edge_index, num_users)
    logging.info(f"Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
    
    train_dict = defaultdict(set)
    for u, i in train_edges:
        train_dict[u].add(i)
        
    # 3. Model setup
    embedding_dim = 64
    batch_size = 2048
    epochs = 100
    learning_rate = 1e-3
    patience = 10
    
    model = LightGCN(num_nodes, embedding_dim, normalized_adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = []
    best_recall = 0
    best_epoch = 0
    epochs_no_improve = 0
    
    logging.info(f"\n{Fore.GREEN}Starting Training...{Style.RESET_ALL}")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training
        model.train()
        np.random.shuffle(train_edges)
        
        total_loss = 0
        n_batches = len(train_edges) // batch_size + 1
        
        for i in range(0, len(train_edges), batch_size):
            batch_edges = train_edges[i:i+batch_size]
            users = batch_edges[:, 0]
            pos_items = batch_edges[:, 1]
            
            neg_items = generate_negative_samples(users, num_books, num_users, train_dict)
            
            users = torch.tensor(users, dtype=torch.long, device=device)
            pos_items = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_items = torch.tensor(neg_items, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            final_emb, initial_emb = model()
            loss = bpr_loss(users, pos_items, neg_items, final_emb, initial_emb)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            final_emb, _ = model()
            
            # Compute Val Loss (simplified randomly sampled negatives)
            val_users = val_edges[:, 0]
            val_pos = val_edges[:, 1]
            val_neg = generate_negative_samples(val_users, num_books, num_users, train_dict)
            
            val_users_t = torch.tensor(val_users, dtype=torch.long, device=device)
            val_pos_t = torch.tensor(val_pos, dtype=torch.long, device=device)
            val_neg_t = torch.tensor(val_neg, dtype=torch.long, device=device)
            
            val_loss = bpr_loss(val_users_t, val_pos_t, val_neg_t, final_emb, initial_emb)
            
            # Metrics
            recall, precision, ndcg = compute_metrics(val_edges, train_dict, final_emb, num_users, num_books, device=device)
            
        epoch_time = time.time() - start_time
        
        log_msg = f"Epoch [{epoch:03d}/{epochs}] | Time: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f} | R@10: {recall:.4f} | P@10: {precision:.4f} | NDCG@10: {ndcg:.4f}"
        logging.info(log_msg)
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss.item(),
            'recall': recall,
            'precision': precision,
            'ndcg': ndcg,
            'time': epoch_time
        })
        
        # Early stopping & best model
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), OUTPUT_DIR / "best_lightgcn_model.pt")
            torch.save(final_emb.cpu(), OUTPUT_DIR / "final_embeddings.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"{Fore.YELLOW}Early stopping triggered after {epoch} epochs.{Style.RESET_ALL}")
                break
                
    logging.info(f"{Fore.GREEN}Training finished! Best Validation Recall@10: {best_recall:.4f} at epoch {best_epoch}{Style.RESET_ALL}")
    
    # Save History and Plots
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    
    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('BPR Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "loss_curve.png")
    
    # Plot Metrics
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df['epoch'], hist_df['recall'], label='Recall@10')
    plt.plot(hist_df['epoch'], hist_df['precision'], label='Precision@10')
    plt.plot(hist_df['epoch'], hist_df['ndcg'], label='NDCG@10')
    plt.title('Metrics Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "metrics_curve.png")
    
    logging.info(f"{Fore.GREEN}Saved all model outputs to {OUTPUT_DIR}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
