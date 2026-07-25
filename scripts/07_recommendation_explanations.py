import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from tqdm import tqdm
import logging
import colorama
from colorama import Fore, Style
import json
from collections import defaultdict, Counter

colorama.init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ExplainableRecommender:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data" / "processed-dataset"
        self.graph_dir = self.base_dir / "data" / "processed-graph"
        self.models_dir = self.base_dir / "models" / "lightgcn"
        self.reports_dir = self.base_dir / "reports" / "explanations"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.books = None
        self.ratings = None
        self.user_mapping = None
        self.book_mapping = None
        self.embeddings = None
        self.user_profiles = {}
        self.book_popularity_by_genre = {}
        
    def load_data(self):
        logging.info(f"{Fore.CYAN}[1/5] Loading datasets and models...{Style.RESET_ALL}")
        
        self.books = pd.read_csv(self.data_dir / "Books_Final_Clean.csv")
        self.books['ISBN'] = self.books['ISBN'].astype(str)
        self.books.set_index('ISBN', inplace=True)
        
        self.ratings = pd.read_csv(self.data_dir / "Ratings_Final.csv")
        self.ratings['ISBN'] = self.ratings['ISBN'].astype(str)
        
        self.user_mapping = pd.read_csv(self.graph_dir / "user_mapping.csv")
        self.book_mapping = pd.read_csv(self.graph_dir / "book_mapping.csv")
        self.book_mapping['ISBN'] = self.book_mapping['ISBN'].astype(str)
        
        self.num_users = len(self.user_mapping)
        self.num_books = len(self.book_mapping)
        
        emb_path = self.models_dir / "final_embeddings.pt"
        if emb_path.exists():
            self.embeddings = torch.load(emb_path, map_location='cpu')
        else:
            logging.warning(f"{Fore.RED}No LightGCN embeddings found. Will use popularity fallback.{Style.RESET_ALL}")
            
    def build_user_profiles(self):
        logging.info(f"{Fore.CYAN}[2/5] Building user preference profiles...{Style.RESET_ALL}")
        
        merged = self.ratings.merge(self.books.reset_index(), on='ISBN', how='inner')
        positive_interactions = merged[merged['Book-Rating'] >= 3]
        
        profiles = []
        grouped = positive_interactions.groupby('User-ID')
        
        for user_id, group in tqdm(grouped, desc="User Profiles", total=len(grouped)):
            avg_rating = group['Book-Rating'].mean()
            num_rated = len(group)
            
            main_genres = group['Main Genre'].value_counts()
            top_main_genres = main_genres.head(3).index.tolist()
            
            diversity = len(main_genres)
            
            # Highest Rated Genres
            genre_avg = group.groupby('Main Genre')['Book-Rating'].mean()
            highest_rated_genres = genre_avg.sort_values(ascending=False).head(3).index.tolist()
            
            sub_genres = group['Sub Genre'].value_counts()
            top_sub_genres = sub_genres.head(3).index.tolist()
            
            authors = group['Author'].value_counts()
            top_authors = authors.head(3).index.tolist()
            
            profile = {
                'User-ID': user_id,
                'Favorite_Main_Genres': "|".join(top_main_genres),
                'Favorite_Sub_Genres': "|".join(top_sub_genres),
                'Favorite_Authors': "|".join(top_authors),
                'Average_Rating_Given': avg_rating,
                'Number_of_Books_Rated': num_rated,
                'Highest_Rated_Genres': "|".join(highest_rated_genres),
                'Reading_Diversity': diversity,
                'Most_Frequently_Read_Genres': "|".join(top_main_genres)
            }
            profiles.append(profile)
            
            self.user_profiles[user_id] = {
                'main_genres': set(top_main_genres),
                'sub_genres': set(top_sub_genres),
                'authors': set(top_authors),
                'highest_rated_genres': set(highest_rated_genres),
                'num_rated': num_rated
            }
            
        profiles_df = pd.DataFrame(profiles)
        
        # Ensure target directory exists before saving
        processed_dir = self.base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        profiles_df.to_csv(processed_dir / "user_profiles.csv", index=False)
        logging.info(f"Saved {len(profiles_df)} user profiles to processed/user_profiles.csv.")
        
    def analyze_book_metadata(self):
        logging.info(f"{Fore.CYAN}[3/5] Analyzing book popularity by genre...{Style.RESET_ALL}")
        
        book_stats = self.ratings.groupby('ISBN')['Book-Rating'].agg(['mean', 'count']).reset_index()
        book_stats.rename(columns={'mean': 'Dataset_Avg_Rating', 'count': 'Dataset_Rating_Count'}, inplace=True)
        
        self.books = self.books.merge(book_stats, on='ISBN', how='left')
        self.books.set_index('ISBN', inplace=True)
        
        self.books['Dataset_Avg_Rating'] = self.books['Dataset_Avg_Rating'].fillna(0)
        self.books['Dataset_Rating_Count'] = self.books['Dataset_Rating_Count'].fillna(0)
        
        for genre, group in self.books.groupby('Main Genre'):
            top_books = group.sort_values(by='Dataset_Rating_Count', ascending=False).head(20).index.tolist()
            self.book_popularity_by_genre[genre] = set(top_books)
            
    def generate_explanations(self, user_id, recommended_isbns, scores, max_score):
        profile = self.user_profiles.get(user_id, {})
        explanations_list = []
        
        for isbn, score in zip(recommended_isbns, scores):
            tags = []
            book = self.books.loc[isbn] if isbn in self.books.index else None
            if book is None:
                continue
                
            b_main_genre = book.get('Main Genre', '')
            b_sub_genre = book.get('Sub Genre', '')
            b_author = book.get('Author', '')
            
            confidence_points = 50.0 # Base confidence
            
            # Normalize CF score contribution (0 to 20 points)
            norm_score = max(0, min(1, score / max(max_score, 1e-9)))
            confidence_points += norm_score * 20
            
            # Tags & Confidence boosts
            if 'main_genres' in profile and b_main_genre in profile['main_genres']:
                tags.append(f"Because you frequently read {b_main_genre} books.")
                confidence_points += 10
                
            if 'sub_genres' in profile and b_sub_genre in profile['sub_genres']:
                tags.append("Matches one of your favorite genres.")
                confidence_points += 5
                
            if 'authors' in profile and b_author in profile['authors']:
                tags.append("Written by an author similar to books you've rated highly.")
                confidence_points += 15
                
            amz_rating = pd.to_numeric(book.get('Rating', 0), errors='coerce')
            amz_count = pd.to_numeric(book.get('Amazon_Rating_Count', book.get('No. of People rated', 0)), errors='coerce')
            
            if not pd.isna(amz_rating) and not pd.isna(amz_count):
                if amz_rating >= 4.5 and amz_count >= 1000:
                    # Format as 25k+ if large
                    count_str = f"{int(amz_count/1000)}k" if amz_count >= 1000 else str(int(amz_count))
                    tags.append(f"Highly rated on Amazon ({amz_rating}★ from {count_str}+ readers).")
                    confidence_points += 5
                    
            if b_main_genre in self.book_popularity_by_genre and isbn in self.book_popularity_by_genre[b_main_genre]:
                tags.append("Popular among users with similar reading preferences.")
                confidence_points += 5
                
            # Fallbacks
            if len(tags) == 0:
                if norm_score > 0.8:
                    tags.append("Similar readers also enjoyed this book.")
                else:
                    tags.append("Recommended based on your learned reading behavior.")
                    
            confidence_points = min(99.9, confidence_points)
            tags = tags[:3] # Max 3 tags
            
            explanations_list.append({
                'ISBN': isbn,
                'Title': book.get('Title', 'Unknown'),
                'Author': b_author,
                'Main_Genre': b_main_genre,
                'Sub_Genre': b_sub_genre,
                'Price': str(book.get('Price', '')),
                'Amazon_Rating': amz_rating,
                'Dataset_Rating': round(book.get('Dataset_Avg_Rating', 0), 2),
                'Similarity_Score': round(float(score), 4),
                'Confidence': f"{round(confidence_points, 1)}%",
                'Explanation': tags
            })
            
        return explanations_list

    def run_inference_sample(self, sample_size=1000):
        logging.info(f"{Fore.CYAN}[4/5] Generating recommendations and explanations...{Style.RESET_ALL}")
        
        if self.embeddings is None:
            return
            
        u_lookup = self.user_mapping.set_index('User-ID')['Node_Index'].to_dict()
        i_lookup = self.book_mapping.set_index('ISBN')['Node_Index'].to_dict()
        inv_i_lookup = {v: k for k, v in i_lookup.items()}
        item_offset = self.num_users
        
        seen_dict = self.ratings.groupby('User-ID')['ISBN'].apply(set).to_dict()
        active_users = self.ratings['User-ID'].value_counts().head(sample_size).index.tolist()
        
        api_responses = []
        all_tags_flat = []
        
        for user_id in tqdm(active_users, desc="Generating Recs"):
            if user_id not in u_lookup: continue
            
            u_node = u_lookup[user_id]
            u_emb = self.embeddings[u_node].unsqueeze(0)
            i_emb = self.embeddings[item_offset:item_offset+self.num_books]
            
            scores = torch.matmul(u_emb, i_emb.T).squeeze().numpy()
            
            seen_isbns = seen_dict.get(user_id, set())
            seen_nodes = [i_lookup[isbn] - item_offset for isbn in seen_isbns if isbn in i_lookup]
            scores[seen_nodes] = -1e9
            
            top_k_idx = np.argsort(-scores)[:10]
            top_scores = scores[top_k_idx]
            max_score = np.max(scores)
            top_isbns = [inv_i_lookup[idx + item_offset] for idx in top_k_idx]
            
            explained_recs = self.generate_explanations(user_id, top_isbns, top_scores, max_score)
            
            user_response = {
                "user_id": int(user_id),
                "recommendations": explained_recs
            }
            api_responses.append(user_response)
            
            for rec in explained_recs:
                all_tags_flat.extend(rec['Explanation'])
                
        if not api_responses:
            logging.warning(f"{Fore.RED}No recommendations generated! Check mappings.{Style.RESET_ALL}")
            return
            
        # Save JSON API Response
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        api_output_path = self.reports_dir / "api_response_sample.json"
        with open(api_output_path, 'w', encoding='utf-8') as f:
            json.dump(api_responses, f, indent=4)
            
        # Save Explanation Stats
        logging.info(f"{Fore.CYAN}[5/5] Saving statistics...{Style.RESET_ALL}")
        tag_counts = Counter(all_tags_flat)
        stats_df = pd.DataFrame(tag_counts.items(), columns=['Explanation Tag', 'Count']).sort_values('Count', ascending=False)
        stats_df.to_csv(self.reports_dir / "explanation_statistics.csv", index=False)
        
        logging.info(f"\n{Fore.GREEN}Explanation Distribution:{Style.RESET_ALL}")
        print(stats_df.to_string(index=False))
        
        logging.info(f"{Fore.GREEN}\nPipeline complete! API response saved to {api_output_path}{Style.RESET_ALL}")

if __name__ == "__main__":
    BASE_DIR = r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory"
    explainer = ExplainableRecommender(BASE_DIR)
    explainer.load_data()
    explainer.build_user_profiles()
    explainer.analyze_book_metadata()
    explainer.run_inference_sample(sample_size=100) # Fast demonstration
