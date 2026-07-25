import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import logging
import colorama
from colorama import Fore, Style
import warnings
warnings.filterwarnings('ignore')

colorama.init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_splits(edge_index, num_users):
    mask = edge_index[0] < num_users
    u_i_edges = edge_index[:, mask].cpu().numpy().T
    
    np.random.seed(42)
    np.random.shuffle(u_i_edges)
    
    n_edges = len(u_i_edges)
    n_train = int(n_edges * 0.8)
    n_val = int(n_edges * 0.1)
    
    train_edges = u_i_edges[:n_train]
    val_edges = u_i_edges[n_train:n_train+n_val]
    test_edges = u_i_edges[n_train+n_val:]
    
    return train_edges, val_edges, test_edges

def compute_metrics(preds, truth, k):
    hits = [1 if p in truth else 0 for p in preds[:k]]
    hit_count = sum(hits)
    precision = hit_count / k
    recall = hit_count / len(truth) if len(truth) > 0 else 0
    hit_rate = 1 if hit_count > 0 else 0
    
    dcg = sum(h / math.log2(rank + 2) for rank, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(truth))))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    ap = 0
    h_c = 0
    for r, h in enumerate(hits):
        if h:
            h_c += 1
            ap += h_c / (r + 1)
    map_k = ap / min(k, len(truth)) if len(truth) > 0 else 0
    
    return precision, recall, ndcg, map_k, hit_rate

class Evaluator:
    def __init__(self, test_dict, train_dict, num_books, item_offset):
        self.test_dict = test_dict
        self.train_dict = train_dict
        self.num_books = num_books
        self.item_offset = item_offset
        self.all_items = np.arange(item_offset, item_offset + num_books)
        
    def evaluate(self, model_name, predictions_dict):
        logging.info(f"Evaluating {model_name}...")
        results = {'Model': model_name}
        
        metrics = {'P@5': [], 'P@10': [], 'R@5': [], 'R@10': [], 
                   'NDCG@5': [], 'NDCG@10': [], 'MAP@10': [], 'HR@10': []}
        
        all_recommended = set()
        
        for u, truth in self.test_dict.items():
            if not truth: continue
            preds = predictions_dict.get(u, [])
            all_recommended.update(preds)
            
            p5, r5, n5, _, _ = compute_metrics(preds, truth, 5)
            p10, r10, n10, map10, hr10 = compute_metrics(preds, truth, 10)
            
            metrics['P@5'].append(p5)
            metrics['P@10'].append(p10)
            metrics['R@5'].append(r5)
            metrics['R@10'].append(r10)
            metrics['NDCG@5'].append(n5)
            metrics['NDCG@10'].append(n10)
            metrics['MAP@10'].append(map10)
            metrics['HR@10'].append(hr10)
            
        for k, v in metrics.items():
            results[k] = np.mean(v)
            
        results['Coverage'] = len(all_recommended) / self.num_books
        return results

def main():
    BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
    GRAPH_DIR = BASE_DIR / "data" / "processed-graph"
    MODELS_DIR = BASE_DIR / "models" / "lightgcn"
    REPORTS_DIR = BASE_DIR / "reports" / "model_comparison"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"{Fore.CYAN}Loading Data...{Style.RESET_ALL}")
    edge_index = torch.load(GRAPH_DIR / "edge_index.pt", map_location='cpu')
    edge_weight = torch.load(GRAPH_DIR / "edge_weight.pt", map_location='cpu').numpy()
    
    user_mapping = pd.read_csv(GRAPH_DIR / "user_mapping.csv")
    book_mapping = pd.read_csv(GRAPH_DIR / "book_mapping.csv")
    num_users = len(user_mapping)
    num_books = len(book_mapping)
    item_offset = num_users
    
    train_edges, val_edges, test_edges = get_splits(edge_index, num_users)
    
    train_dict = defaultdict(set)
    for u, i in train_edges: train_dict[u].add(i)
    
    test_dict = defaultdict(set)
    test_users = set()
    for u, i in test_edges: 
        test_dict[u].add(i)
        test_users.add(u)
        
    test_users = list(test_users)
    
    # Pre-build User-Item Sparse Matrix for Training Data
    # For ItemCF and SVD
    logging.info(f"{Fore.CYAN}Building User-Item Matrix for Baselines...{Style.RESET_ALL}")
    
    # We need to recover edge weights for train_edges to do rating-based baselines
    # A quick trick: just use 1.0 for implicit feedback since LightGCN is implicit
    # But for "Highest Avg Rating", we need actual ratings.
    # Let's map original edges to weights
    full_mask = edge_index[0] < num_users
    u_idx = edge_index[0, full_mask].numpy()
    i_idx = edge_index[1, full_mask].numpy()
    w = edge_weight[full_mask]
    
    # Create dictionary for quick weight lookup
    import scipy.sparse as sp
    full_R = sp.coo_matrix((w, (u_idx, i_idx - item_offset)), shape=(num_users, num_books)).tocsr()
    
    # Get weights for train_edges
    train_u = train_edges[:, 0]
    train_i = train_edges[:, 1] - item_offset
    train_w = np.array(full_R[train_u, train_i]).flatten()[0] # This handles extraction
    # Actually sparse matrix indexing for arrays returns a matrix, easier is:
    train_w = full_R[train_u, train_i].A1
    
    R_train = sp.csr_matrix((train_w, (train_u, train_i)), shape=(num_users, num_books))
    R_train_implicit = sp.csr_matrix((np.ones_like(train_w), (train_u, train_i)), shape=(num_users, num_books))
    
    evaluator = Evaluator(test_dict, train_dict, num_books, item_offset)
    predictions = {}
    
    # 1. Popularity
    logging.info(f"{Fore.YELLOW}Running Popularity Model...{Style.RESET_ALL}")
    item_counts = np.array(R_train_implicit.sum(axis=0)).flatten()
    pop_order = np.argsort(-item_counts) + item_offset
    pop_preds = {}
    for u in tqdm(test_users, desc="Popularity"):
        recs = [i for i in pop_order if i not in train_dict[u]][:10]
        pop_preds[u] = recs
    predictions['Popularity'] = pop_preds
    
    # 2. Highest Average Rating
    logging.info(f"{Fore.YELLOW}Running Highest Average Rating Model...{Style.RESET_ALL}")
    item_sums = np.array(R_train.sum(axis=0)).flatten()
    item_counts_safe = np.where(item_counts == 0, 1, item_counts)
    item_avgs = item_sums / item_counts_safe
    # Bayesian average or just pure average with min ratings
    min_ratings = 5
    valid_items = item_counts >= min_ratings
    item_avgs[~valid_items] = 0
    avg_order = np.argsort(-item_avgs) + item_offset
    avg_preds = {}
    for u in tqdm(test_users, desc="Highest Avg Rating"):
        recs = [i for i in avg_order if i not in train_dict[u]][:10]
        avg_preds[u] = recs
    predictions['Highest Avg Rating'] = avg_preds
    
    # 3. Random
    logging.info(f"{Fore.YELLOW}Running Random Model...{Style.RESET_ALL}")
    np.random.seed(42)
    rand_preds = {}
    all_items = np.arange(item_offset, item_offset + num_books)
    for u in tqdm(test_users, desc="Random"):
        available = np.setdiff1d(all_items, list(train_dict[u]))
        recs = np.random.choice(available, 10, replace=False).tolist()
        rand_preds[u] = recs
    predictions['Random'] = rand_preds
    
    # 4. Item-Based CF
    logging.info(f"{Fore.YELLOW}Running Item-Based CF (Cosine)...{Style.RESET_ALL}")
    # item-item similarity
    item_sim = cosine_similarity(R_train_implicit.T, dense_output=False)
    item_cf_preds = {}
    for u in tqdm(test_users, desc="Item CF"):
        # Get user's items
        user_items = np.array(list(train_dict[u])) - item_offset
        if len(user_items) == 0:
            item_cf_preds[u] = predictions['Popularity'][u]
            continue
        # Sum similarities of user's items to all other items
        scores = np.array(item_sim[user_items, :].sum(axis=0)).flatten()
        scores[user_items] = -1e9 # mask train
        top_idx = np.argsort(-scores)[:10]
        item_cf_preds[u] = (top_idx + item_offset).tolist()
    predictions['ItemCF'] = item_cf_preds
    
    # 5. SVD
    logging.info(f"{Fore.YELLOW}Running Matrix Factorization (SVD)...{Style.RESET_ALL}")
    k_svd = min(50, num_books-1, num_users-1)
    U, sigma, Vt = svds(R_train_implicit.astype(float), k=k_svd)
    sigma = np.diag(sigma)
    user_factors = np.dot(U, sigma)
    item_factors = Vt.T
    
    svd_preds = {}
    for u in tqdm(test_users, desc="SVD"):
        if u >= num_users: continue
        scores = np.dot(user_factors[u], item_factors.T)
        train_mask = (np.array(list(train_dict[u])) - item_offset).astype(int)
        scores[train_mask] = -1e9
        top_idx = np.argsort(-scores)[:10]
        svd_preds[u] = (top_idx + item_offset).tolist()
    predictions['SVD'] = svd_preds
    
    # 6. LightGCN
    logging.info(f"{Fore.YELLOW}Running LightGCN...{Style.RESET_ALL}")
    lgcn_emb_path = MODELS_DIR / "final_embeddings.pt"
    if lgcn_emb_path.exists():
        lgcn_emb = torch.load(lgcn_emb_path, map_location='cpu')
        lgcn_preds = {}
        for u in tqdm(test_users, desc="LightGCN"):
            u_emb = lgcn_emb[u].unsqueeze(0)
            i_emb = lgcn_emb[item_offset:item_offset+num_books]
            scores = torch.matmul(u_emb, i_emb.T).squeeze().numpy()
            train_mask = (np.array(list(train_dict[u])) - item_offset).astype(int)
            scores[train_mask] = -1e9
            top_idx = np.argsort(-scores)[:10]
            lgcn_preds[u] = (top_idx + item_offset).tolist()
        predictions['LightGCN'] = lgcn_preds
    else:
        logging.warning("LightGCN embeddings not found. Skipping LightGCN evaluation.")
        
    # Evaluate All
    all_results = []
    for model_name, preds_dict in predictions.items():
        res = evaluator.evaluate(model_name, preds_dict)
        all_results.append(res)
        
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    cols = ['Model', 'P@5', 'P@10', 'R@5', 'R@10', 'NDCG@5', 'NDCG@10', 'MAP@10', 'HR@10', 'Coverage']
    results_df = results_df[cols]
    
    # Save CSV
    results_df.to_csv(BASE_DIR / "reports" / "model_comparison.csv", index=False)
    
    # Display Table
    logging.info(f"\n{Fore.GREEN}=== Model Comparison Results ==={Style.RESET_ALL}")
    print(results_df.to_string(index=False))
    
    # Generate Plots
    logging.info(f"\n{Fore.CYAN}Generating comparison plots...{Style.RESET_ALL}")
    sns.set_theme(style="whitegrid")
    metrics_to_plot = ['P@10', 'R@10', 'NDCG@10', 'HR@10']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='Model', y=metric, palette='viridis')
        plt.title(f'Model Comparison: {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"{metric.replace('@', '_')}_comparison.png")
        plt.close()
        
    # Generate Summary Report
    report_path = BASE_DIR / "reports" / "model_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== GRAPHREC MODEL COMPARISON REPORT ===\n\n")
        f.write("This report evaluates the LightGCN model against classical baselines.\n\n")
        f.write("RESULTS SUMMARY:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        # Determine winners
        for metric in ['P@10', 'R@10', 'NDCG@10', 'HR@10']:
            winner = results_df.loc[results_df[metric].idxmax()]['Model']
            score = results_df[metric].max()
            f.write(f"- Best {metric}: {winner} ({score:.4f})\n")
            
        f.write("\nOBSERVATIONS:\n")
        f.write("- Popularity: Often acts as a strong baseline but lacks personalization.\n")
        f.write("- Highest Avg Rating: Typically performs poorly on its own because it favors obscure items with a few 5-star ratings.\n")
        f.write("- Random: Establishes the absolute lower bound.\n")
        f.write("- ItemCF: Strong collaborative filtering baseline, but struggles with extreme sparsity.\n")
        f.write("- SVD: Reduces dimensionality, capturing latent semantics.\n")
        f.write("- LightGCN: Typically outperforms standard baselines by capturing higher-order network structures in the bipartite graph.\n")
        
    logging.info(f"{Fore.GREEN}Pipeline complete! Output saved to reports/model_comparison/{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
