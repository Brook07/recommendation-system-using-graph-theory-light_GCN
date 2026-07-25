"""Inference wrapper used by the FastAPI backend."""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import torch

class InferenceEngine:
    """Serve recommendations from saved LightGCN embeddings and metadata caches."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data" / "processed-dataset"
        self.graph_dir = self.base_dir / "data" / "processed-graph"
        self.models_dir = self.base_dir / "models" / "lightgcn"
        self.processed_dir = self.base_dir / "processed"
        
        self.books_df = None
        self.profiles_df = None
        self.user_mapping = None
        self.book_mapping = None
        self.embeddings = None
        self.seen_dict = {}
        self.book_popularity_by_genre = {}
        
        self._load_data()

    def _load_data(self):
        logging.info("Loading Inference Engine Caches...")
        t0 = time.time()
        
        # 1. Load Books
        books_path = self.data_dir / "Books_Final_Clean.csv"
        if books_path.exists():
            self.books_df = pd.read_csv(books_path, low_memory=False)
            
        if self.books_df is not None:
            self.books_df['ISBN'] = self.books_df['ISBN'].astype(str)
            self.books_df.set_index('ISBN', inplace=True)
            
            # Recompute global stats if missing
            if 'Dataset_Avg_Rating' not in self.books_df.columns:
                self.books_df['Dataset_Avg_Rating'] = 0.0
                self.books_df['Dataset_Rating_Count'] = 0.0
                
            # Pop by genre
            for genre, group in self.books_df.groupby('Main Genre'):
                top = group.sort_values(by='Dataset_Rating_Count', ascending=False).head(20).index.tolist()
                self.book_popularity_by_genre[genre] = set(top)
        else:
            self.books_df = pd.DataFrame()
            
        # 2. Load Profiles
        profiles_path = self.processed_dir / "user_profiles.csv"
        if profiles_path.exists():
            self.profiles_df = pd.read_csv(profiles_path)
            self.profiles_df.set_index('User-ID', inplace=True)
        else:
            self.profiles_df = pd.DataFrame()
            
        # 3. Load Graph Mappings
        u_map_path = self.graph_dir / "user_mapping.csv"
        b_map_path = self.graph_dir / "book_mapping.csv"
        
        if u_map_path.exists() and b_map_path.exists():
            self.user_mapping = pd.read_csv(u_map_path)
            self.u_lookup = self.user_mapping.set_index('User-ID')['Node_Index'].to_dict()
            self.num_users = len(self.user_mapping)
            
            self.book_mapping = pd.read_csv(b_map_path)
            self.book_mapping['ISBN'] = self.book_mapping['ISBN'].astype(str)
            self.i_lookup = self.book_mapping.set_index('ISBN')['Node_Index'].to_dict()
            self.inv_i_lookup = {v: k for k, v in self.i_lookup.items()}
            self.num_books = len(self.book_mapping)
        else:
            self.u_lookup = {}
            self.i_lookup = {}
            self.num_users = 0
            self.num_books = 0
            
        # 4. Load Seen Dict (for masking)
        ratings_path = self.data_dir / "Ratings_Final.csv"
        if ratings_path.exists():
            ratings = pd.read_csv(ratings_path)
            ratings['ISBN'] = ratings['ISBN'].astype(str)
            self.seen_dict = ratings.groupby('User-ID')['ISBN'].apply(set).to_dict()
            self.total_ratings = len(ratings)
        else:
            self.total_ratings = 0
            
        # 5. Load Embeddings
        emb_path = self.models_dir / "final_embeddings.pt"
        if emb_path.exists():
            self.embeddings = torch.load(emb_path, map_location='cpu')
            
        logging.info(f"Loaded engine in {time.time() - t0:.2f}s")
        
    def _generate_explanations(self, user_id: int, top_isbns: List[str], scores: np.ndarray) -> List[Dict]:
        profile = {}
        if user_id in self.profiles_df.index:
            row = self.profiles_df.loc[user_id]
            # Convert NaN to empty strings/sets
            profile = {
                'main_genres': set(str(row.get('Favorite_Main_Genres', '')).split('|')),
                'sub_genres': set(str(row.get('Favorite_Sub_Genres', '')).split('|')),
                'authors': set(str(row.get('Favorite_Authors', '')).split('|')),
            }
            
        max_score = np.max(scores) if len(scores) > 0 else 1.0
        explanations_list = []
        
        for isbn, score in zip(top_isbns, scores):
            tags = []
            if isbn not in self.books_df.index:
                continue
            book = self.books_df.loc[isbn]
            
            b_main_genre = str(book.get('Main Genre', ''))
            b_sub_genre = str(book.get('Sub Genre', ''))
            b_author = str(book.get('Author', ''))
            
            confidence = 50.0
            norm_score = max(0, min(1, score / max(max_score, 1e-9)))
            confidence += norm_score * 20
            
            if b_main_genre in profile.get('main_genres', set()) and b_main_genre:
                tags.append(f"Because you frequently read {b_main_genre} books.")
                confidence += 10
            if b_sub_genre in profile.get('sub_genres', set()) and b_sub_genre:
                tags.append("Matches one of your favorite genres.")
                confidence += 5
            if b_author in profile.get('authors', set()) and b_author:
                tags.append("Written by an author similar to books you've rated highly.")
                confidence += 15
                
            amz_rating = pd.to_numeric(book.get('Rating', 0), errors='coerce')
            amz_count = pd.to_numeric(book.get('Amazon_Rating_Count', book.get('No. of People rated', 0)), errors='coerce')
            
            if not pd.isna(amz_rating) and not pd.isna(amz_count) and amz_rating >= 4.5 and amz_count >= 1000:
                count_str = f"{int(amz_count/1000)}k" if amz_count >= 1000 else str(int(amz_count))
                tags.append(f"Highly rated on Amazon ({amz_rating}★ from {count_str}+ readers).")
                confidence += 5
                
            if b_main_genre in self.book_popularity_by_genre and isbn in self.book_popularity_by_genre[b_main_genre]:
                tags.append("Popular among users with similar reading preferences.")
                confidence += 5
                
            if not tags:
                if norm_score > 0.8:
                    tags.append("Similar readers also enjoyed this book.")
                else:
                    tags.append("Recommended based on your learned reading behavior.")
                    
            confidence = min(99.9, confidence)
            
            explanations_list.append({
                "ISBN": str(isbn),
                "Book_Title": str(book.get('Title', 'Unknown')),
                "Author": str(book.get('Author', 'Unknown')),
                "Main_Genre": str(book.get('Main Genre', 'Unknown')),
                "Sub_Genre": str(book.get('Sub Genre', 'Unknown')),
                "Book_Type": str(book.get('Type', 'Unknown')),
                "Price": str(book.get('Price', 'Unknown')),
                "Amazon_Rating": float(amz_rating) if not pd.isna(amz_rating) else 0.0,
                "Number_of_People_Rated": int(amz_count) if not pd.isna(amz_count) else 0,
                "Dataset_Average_Rating": float(book.get('Dataset_Avg_Rating', 0.0)),
                "Recommendation_Score": round(float(score), 4),
                "Confidence_Score": f"{round(confidence, 1)}%",
                "Explanation_Tags": tags[:3],
                "Book_Cover_URL": str(book.get('Google_Thumbnail', '')) if pd.notna(book.get('Google_Thumbnail')) and str(book.get('Google_Thumbnail', '')).strip() else str(book.get('Image-URL-L', '')),
                "Amazon_URL": str(book.get('URL', ''))
            })
            
        return explanations_list

    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return enriched top-k recommendations."""
        if self.embeddings is None or user_id not in self.u_lookup:
            return [] # Empty recs for unknown user or missing model
            
        u_node = self.u_lookup[user_id]
        item_offset = self.num_users
        
        u_emb = self.embeddings[u_node].unsqueeze(0)
        i_emb = self.embeddings[item_offset:item_offset+self.num_books]
        
        scores = torch.matmul(u_emb, i_emb.T).squeeze().numpy()
        
        seen_isbns = self.seen_dict.get(user_id, set())
        seen_nodes = [self.i_lookup[isbn] - item_offset for isbn in seen_isbns if isbn in self.i_lookup]
        scores[seen_nodes] = -1e9
        
        top_k_idx = np.argsort(-scores)[:top_k]
        top_scores = scores[top_k_idx]
        top_isbns = [self.inv_i_lookup[idx + item_offset] for idx in top_k_idx]
        
        return self._generate_explanations(user_id, top_isbns, top_scores)

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        if user_id in self.profiles_df.index:
            row = self.profiles_df.loc[user_id]
            return {
                "Favorite_Genres": [g for g in str(row.get('Favorite_Main_Genres', '')).split('|') if g],
                "Favorite_Authors": [a for a in str(row.get('Favorite_Authors', '')).split('|') if a],
                "Books_Rated": int(row.get('Number_of_Books_Rated', 0)),
                "Average_Rating": float(row.get('Average_Rating_Given', 0.0)),
                "Reading_Diversity": int(row.get('Reading_Diversity', 0)),
                "Total_Recommendations_Generated": 0 # Tracked elsewhere if needed
            }
        return None

    def get_book_metadata(self, isbn: str) -> Optional[Dict]:
        if isbn in self.books_df.index:
            book = self.books_df.loc[isbn]
            return book.to_dict()
        return None

    def get_analytics(self) -> Dict:
        return {
            "Total_Users": int(self.num_users),
            "Total_Books": int(self.num_books),
            "Total_Ratings": int(self.total_ratings),
            "Most_Popular_Genres": list(self.book_popularity_by_genre.keys())[:5],
            "Most_Active_Users": [],
            "Most_Recommended_Books": []
        }
