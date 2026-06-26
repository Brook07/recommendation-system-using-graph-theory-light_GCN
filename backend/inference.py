import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add src to sys.path to allow importing from src
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.lightgcn import LightGCNModel, build_adjacency

class InferenceEngine:
    def __init__(self, model_path: str = "models/best_lightgcn_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load mappings
        self.user_map_df = pd.read_csv("data/processed/filtered_user_mapping.csv")
        self.book_map_df = pd.read_csv("data/processed/filtered_book_mapping.csv")
        
        self.user_map = dict(zip(self.user_map_df['User-ID'].to_list(), self.user_map_df['node_index'].to_list()))
        self.book_map = dict(zip(self.book_map_df['ISBN'].to_list(), self.book_map_df['node_index'].to_list()))
        self.inv_item_map = {v: k for k, v in self.book_map.items()}
        
        num_nodes = int(self.user_map_df['node_index'].max() + 1 + (self.book_map_df['node_index'].max() - self.user_map_df['node_index'].max()))
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        emb_dim = checkpoint.get('embedding_size', checkpoint.get('emb_dim', 64))
        n_layers = checkpoint.get('num_layers', 3)
        
        self.model = LightGCNModel(num_nodes, emb_dim, n_layers=n_layers).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Build adjacency and precompute embeddings
        print("Precomputing LightGCN embeddings...")
        adj = build_adjacency(num_nodes, edge_index_path=Path("data/processed/filtered_edge_index.csv"))
        adj = adj.to(self.device)
        with torch.no_grad():
            self.final_embeddings = self.model(adj)
            
        self.item_indices = np.array(sorted(self.book_map.values()))
        
        # Load Books metadata for merging later
        print("Loading Books metadata...")
        # read_csv on Books.csv might have mixed types, specifying dtype or low_memory=False
        self.books_df = pd.read_csv("data/raw-dataset-books/Books.csv", low_memory=False)
        
        # Prepare fallback (popularity-based) recommendations
        print("Preparing fallback recommendations...")
        ratings = pd.read_csv("data/processed/cleaned_ratings.csv")
        # Count number of ratings and average rating per book
        book_stats = ratings.groupby('ISBN').agg({'Book-Rating': ['count', 'mean']})
        book_stats.columns = ['rating_count', 'rating_mean']
        
        # Filter for books with at least 50 ratings and sort by rating mean then count
        popular_books = book_stats[book_stats['rating_count'] >= 50].sort_values(
            by=['rating_mean', 'rating_count'], ascending=[False, False]
        )
        self.fallback_isbns = popular_books.index.tolist()[:10]
        self.fallback_scores = popular_books['rating_mean'].tolist()[:10]
        print("Inference Engine initialized successfully.")

    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        isbns = []
        scores = []
        is_fallback = False
        
        if user_id in self.user_map:
            # User is known, use LightGCN model
            u_node = self.user_map[user_id]
            u_emb = self.final_embeddings[u_node]
            
            # Compute scores for all items
            all_scores = torch.matmul(self.final_embeddings, u_emb)
            item_scores = all_scores[self.item_indices].cpu().detach().numpy()
            
            # Get top K
            topk_idx = np.argsort(-item_scores)[:top_k]
            topk_nodes = self.item_indices[topk_idx]
            
            isbns = [self.inv_item_map[int(x)] for x in topk_nodes]
            scores = [float(item_scores[idx]) for idx in topk_idx]
        else:
            # Unknown user, use fallback popularity
            isbns = self.fallback_isbns[:top_k]
            scores = self.fallback_scores[:top_k]
            is_fallback = True

        # Attach metadata
        recs = []
        for isbn, score in zip(isbns, scores):
            book_info = self.books_df[self.books_df['ISBN'] == isbn]
            if not book_info.empty:
                book_info = book_info.iloc[0]
                recs.append({
                    "ISBN": isbn,
                    "Book-Title": book_info.get("Book-Title", "Unknown Title"),
                    "Book-Author": book_info.get("Book-Author", "Unknown Author"),
                    "Year-Of-Publication": book_info.get("Year-Of-Publication", "Unknown"),
                    "Publisher": book_info.get("Publisher", "Unknown"),
                    "Image-URL-L": book_info.get("Image-URL-L", ""),
                    "Score": score,
                    "Is-Fallback": is_fallback
                })
            else:
                recs.append({
                    "ISBN": isbn,
                    "Book-Title": "Unknown Title",
                    "Book-Author": "Unknown Author",
                    "Year-Of-Publication": "Unknown",
                    "Publisher": "Unknown",
                    "Image-URL-L": "",
                    "Score": score,
                    "Is-Fallback": is_fallback
                })
        return recs

# For testing
if __name__ == "__main__":
    engine = InferenceEngine()
    print("Recommendations for User 276964:")
    print(engine.get_recommendations(276964))
    print("Recommendations for Unknown User 999999:")
    print(engine.get_recommendations(999999))
