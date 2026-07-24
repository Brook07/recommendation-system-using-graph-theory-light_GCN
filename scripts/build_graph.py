import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import torch
import scipy.sparse as sp
import json

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class GraphBuilder:
    def __init__(self, books_path, ratings_path, output_dir):
        self.books_path = Path(books_path)
        self.ratings_path = Path(ratings_path)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.books = None
        self.ratings = None
        
        self.num_users = 0
        self.num_books = 0
        self.num_nodes = 0
        
        self.user_mapping = {}
        self.book_mapping = {}

    def load_and_validate(self):
        logging.info(f"{Fore.CYAN}[1/5] Loading and Validating Data...{Style.RESET_ALL}")
        
        self.books = pd.read_csv(self.books_path)
        
        # Check for v2 fallback
        v2_path = str(self.ratings_path).replace('.csv', '_v2.csv')
        if os.path.exists(v2_path):
            logging.info(f"{Fore.YELLOW}Found {Path(v2_path).name}, using it for graph construction.{Style.RESET_ALL}")
            self.ratings = pd.read_csv(v2_path)
        else:
            self.ratings = pd.read_csv(self.ratings_path)
            
        # Validate ISBNs
        books_isbns = set(self.books['ISBN'].astype(str))
        ratings_isbns = set(self.ratings['ISBN'].astype(str))
        
        missing = ratings_isbns - books_isbns
        if missing:
            logging.warning(f"{Fore.RED}Found {len(missing)} ISBNs in ratings not present in books! Removing them.{Style.RESET_ALL}")
            self.ratings = self.ratings[self.ratings['ISBN'].astype(str).isin(books_isbns)]
        else:
            logging.info(f"{Fore.GREEN}Validation passed: All ratings ISBNs exist in books.{Style.RESET_ALL}")
            
        # Remove duplicates
        initial_len = len(self.ratings)
        self.ratings = self.ratings.drop_duplicates(subset=['User-ID', 'ISBN'])
        if len(self.ratings) < initial_len:
            logging.info(f"Removed {initial_len - len(self.ratings)} duplicate interactions.")

    def create_mappings(self):
        logging.info(f"\n{Fore.CYAN}[2/5] Creating Node Mappings...{Style.RESET_ALL}")
        
        unique_users = self.ratings['User-ID'].unique()
        # We only map books that are in the ratings to save space, or map all books?
        # Usually we map all books in the catalog.
        unique_books = self.books['ISBN'].astype(str).unique()
        
        self.num_users = len(unique_users)
        self.num_books = len(unique_books)
        self.num_nodes = self.num_users + self.num_books
        
        # Users get IDs 0 to U-1
        self.user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        
        # Books get IDs U to U+B-1
        self.book_mapping = {isbn: idx + self.num_users for idx, isbn in enumerate(unique_books)}
        
        # Save mappings
        pd.DataFrame({
            'User-ID': list(self.user_mapping.keys()),
            'Node_Index': list(self.user_mapping.values())
        }).to_csv(self.output_dir / "user_mapping.csv", index=False)
        
        pd.DataFrame({
            'ISBN': list(self.book_mapping.keys()),
            'Node_Index': list(self.book_mapping.values())
        }).to_csv(self.output_dir / "book_mapping.csv", index=False)
        
        logging.info(f"Mapped {self.num_users:,} users and {self.num_books:,} books.")
        logging.info(f"Total Nodes: {self.num_nodes:,}")

    def build_graph(self):
        logging.info(f"\n{Fore.CYAN}[3/5] Building Bipartite Graph...{Style.RESET_ALL}")
        
        # Map IDs in dataframe
        user_nodes = self.ratings['User-ID'].map(self.user_mapping).values
        book_nodes = self.ratings['ISBN'].astype(str).map(self.book_mapping).values
        ratings = self.ratings['Book-Rating'].values
        
        # Bipartite graph is undirected, so we add edges in both directions
        row = np.concatenate([user_nodes, book_nodes])
        col = np.concatenate([book_nodes, user_nodes])
        data = np.concatenate([ratings, ratings])
        
        # Create sparse adjacency matrix
        logging.info("Constructing sparse adjacency matrix...")
        self.adj_matrix = sp.coo_matrix(
            (data, (row, col)), 
            shape=(self.num_nodes, self.num_nodes),
            dtype=np.float32
        )
        
        # Edge Index (2 x E)
        self.edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
        self.edge_weight = torch.tensor(data, dtype=torch.float32)

    def normalize_graph(self):
        logging.info(f"\n{Fore.CYAN}[4/5] Normalizing Adjacency Matrix...{Style.RESET_ALL}")
        
        # D^-0.5 A D^-0.5
        # Calculate degree matrix (sum of edge weights)
        rowsum = np.array(self.adj_matrix.sum(1)).flatten()
        
        # D^-0.5
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        logging.info("Applying symmetric normalization...")
        normalized_adj = self.adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        
        # Convert to PyTorch sparse tensor format (values and indices)
        indices = torch.tensor(np.vstack((normalized_adj.row, normalized_adj.col)), dtype=torch.long)
        values = torch.tensor(normalized_adj.data, dtype=torch.float32)
        
        self.normalized_adj_tensor = {'indices': indices, 'values': values, 'shape': normalized_adj.shape}

    def compute_statistics_and_save(self):
        logging.info(f"\n{Fore.CYAN}[5/5] Computing Statistics and Saving...{Style.RESET_ALL}")
        
        num_edges = len(self.ratings)
        density = num_edges / (self.num_users * self.num_books)
        avg_degree = num_edges / self.num_nodes
        avg_rating = self.ratings['Book-Rating'].mean()
        
        logging.info("Computing connected components...")
        n_components, labels = sp.csgraph.connected_components(
            csgraph=self.adj_matrix, directed=False, return_labels=True
        )
        
        stats = {
            "num_users": int(self.num_users),
            "num_books": int(self.num_books),
            "num_nodes": int(self.num_nodes),
            "num_edges": int(num_edges),
            "graph_density": float(density),
            "average_degree": float(avg_degree),
            "connected_components": int(n_components),
            "average_rating": float(avg_rating)
        }
        
        # Save Tensors
        logging.info("Saving PyTorch tensors...")
        torch.save(self.edge_index, self.output_dir / "edge_index.pt")
        torch.save(self.edge_weight, self.output_dir / "edge_weight.pt")
        torch.save(self.normalized_adj_tensor, self.output_dir / "normalized_adj.pt")
        
        # Save JSON Stats
        with open(self.output_dir / "graph_statistics.json", 'w') as f:
            json.dump(stats, f, indent=4)
            
        logging.info(f"\n{Fore.YELLOW}--- Graph Statistics ---{Style.RESET_ALL}")
        logging.info(f"Number of users: {self.num_users:,}")
        logging.info(f"Number of books: {self.num_books:,}")
        logging.info(f"Number of edges (interactions): {num_edges:,}")
        logging.info(f"Graph density: {density:.6e}")
        logging.info(f"Average degree: {avg_degree:.2f}")
        logging.info(f"Connected components: {n_components:,}")
        logging.info(f"Average rating: {avg_rating:.2f}")
        logging.info(f"{Fore.YELLOW}------------------------{Style.RESET_ALL}")
        
        logging.info(f"{Fore.GREEN}Graph construction complete! Outputs saved to {self.output_dir}{Style.RESET_ALL}")

    def run(self):
        try:
            self.load_and_validate()
            self.create_mappings()
            self.build_graph()
            self.normalize_graph()
            self.compute_statistics_and_save()
        except Exception as e:
            logging.error(f"{Fore.RED}Graph construction failed: {e}{Style.RESET_ALL}")
            raise

if __name__ == "__main__":
    BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
    PROCESSED_DIR = BASE_DIR / "data" / "processed-dataset"
    GRAPH_DIR = BASE_DIR / "data" / "processed-graph"
    
    BOOKS_PATH = PROCESSED_DIR / "Enhanced_Books.csv"
    RATINGS_PATH = PROCESSED_DIR / "Enhanced_Ratings.csv"
    
    builder = GraphBuilder(
        books_path=BOOKS_PATH,
        ratings_path=RATINGS_PATH,
        output_dir=GRAPH_DIR
    )
    
    builder.run()
