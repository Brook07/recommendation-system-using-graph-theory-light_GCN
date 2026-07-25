import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import logging

# Initialize colorama
colorama.init(autoreset=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class DatasetBuilder:
    def __init__(self, books_path, genres_path, ratings_path, output_dir):
        self.books_path = books_path
        self.genres_path = genres_path
        self.ratings_path = ratings_path
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.books_df = None
        self.genres_df = None
        self.ratings_df = None

    def load_data(self):
        logging.info(f"{Fore.CYAN}[1/6] Loading datasets...{Style.RESET_ALL}")
        try:
            with tqdm(total=3, desc="Loading CSVs") as pbar:
                self.books_df = pd.read_csv(self.books_path)
                # Rename the column as requested
                self.books_df = self.books_df.rename(columns={'No. of People rated': 'Amazon_Rating_Count'})
                pbar.update(1)
                
                self.genres_df = pd.read_csv(self.genres_path)
                pbar.update(1)
                
                # Ratings might be large, read_csv is fast but we just update progress
                self.ratings_df = pd.read_csv(self.ratings_path)
                pbar.update(1)
                
            # Clean ISBNs to ensure they are exactly 10 digits and purely numeric
            logging.info("Cleaning ISBNs (enforcing 10-digit numeric)...")
            self.ratings_df['ISBN'] = self.ratings_df['ISBN'].astype(str).str.upper()
            
            # Replace 'X' with '0' to keep it numeric without changing length
            self.ratings_df['ISBN'] = self.ratings_df['ISBN'].str.replace('X', '0', regex=False)
            
            # Remove any non-digit characters
            self.ratings_df['ISBN'] = self.ratings_df['ISBN'].str.replace(r'\D', '', regex=True)
            
            # Pad or truncate to ensure exactly 10 digits
            self.ratings_df['ISBN'] = self.ratings_df['ISBN'].str.zfill(10).str.slice(0, 10)
                
            self._print_stats("Before Processing")
        except Exception as e:
            logging.error(f"{Fore.RED}Error loading data: {e}{Style.RESET_ALL}")
            raise

    def assign_isbns(self):
        logging.info(f"{Fore.CYAN}[2/6] Assigning ISBNs to books...{Style.RESET_ALL}")
        try:
            # Extract unique ISBNs from ratings
            unique_isbns = self.ratings_df['ISBN'].unique()
            
            if len(unique_isbns) < len(self.books_df):
                raise ValueError("Not enough unique ISBNs in ratings to assign to all books.")
            
            # Assign sequentially and uniquely
            assigned_isbns = unique_isbns[:len(self.books_df)]
            self.books_df['ISBN'] = assigned_isbns
            
        except Exception as e:
            logging.error(f"{Fore.RED}Error assigning ISBNs: {e}{Style.RESET_ALL}")
            raise

    def merge_genres(self):
        logging.info(f"{Fore.CYAN}[3/6] Merging genre information...{Style.RESET_ALL}")
        try:
            # In Genre_df, the 'Title' column actually contains the genre name
            # which matches 'Main Genre' in Books_df
            genres_renamed = self.genres_df.rename(columns={'Title': 'Main Genre'})
            
            self.books_df = pd.merge(
                self.books_df, 
                genres_renamed, 
                on='Main Genre', 
                how='left'
            )
            # rename URL to Genre_URL or drop it depending on need, let's keep it as Genre_URL
            if 'URL' in self.books_df.columns:
                self.books_df = self.books_df.rename(columns={'URL': 'Genre_URL'})
                
        except Exception as e:
            logging.error(f"{Fore.RED}Error merging genres: {e}{Style.RESET_ALL}")
            raise

    def process_ratings(self):
        logging.info(f"{Fore.CYAN}[4/6] Filtering and normalizing ratings...{Style.RESET_ALL}")
        try:
            # Filter ratings to only include valid ISBNs present in the new Books_df
            valid_isbns = set(self.books_df['ISBN'])
            
            # Apply filter
            logging.info("Filtering ratings...")
            self.ratings_df = self.ratings_df[self.ratings_df['ISBN'].isin(valid_isbns)].copy()
            
            # Normalize Book-Rating from 1-10 to 1-5, keeping 0 as 0
            logging.info("Normalizing ratings...")
            self.ratings_df['Book-Rating'] = np.where(
                self.ratings_df['Book-Rating'] == 0, 
                0, 
                self.ratings_df['Book-Rating'] / 2.0
            )
            
        except Exception as e:
            logging.error(f"{Fore.RED}Error processing ratings: {e}{Style.RESET_ALL}")
            raise

    def validate(self):
        logging.info(f"{Fore.CYAN}[5/6] Validating processed data...{Style.RESET_ALL}")
        try:
            # 1. No duplicate ISBNs in books
            if self.books_df['ISBN'].duplicated().any():
                raise ValueError("Duplicate ISBNs found in Enhanced_Books.csv")
            logging.info(f"{Fore.GREEN}✓ No duplicate ISBNs in books.{Style.RESET_ALL}")
            
            # 2. All ISBNs in ratings exist in books
            books_isbns = set(self.books_df['ISBN'])
            ratings_isbns = set(self.ratings_df['ISBN'])
            if not ratings_isbns.issubset(books_isbns):
                raise ValueError("Found ISBNs in ratings that do not exist in books.")
            logging.info(f"{Fore.GREEN}✓ All ratings map to valid books.{Style.RESET_ALL}")
            
            self._print_stats("After Processing")
            
        except Exception as e:
            logging.error(f"{Fore.RED}Validation failed: {e}{Style.RESET_ALL}")
            raise

    def save_data(self):
        logging.info(f"{Fore.CYAN}[6/6] Saving datasets...{Style.RESET_ALL}")
        try:
            books_out = os.path.join(self.output_dir, 'Enhanced_Books.csv')
            ratings_out = os.path.join(self.output_dir, 'Enhanced_Ratings.csv')
            
            # Reorder columns to make 'ISBN' the first column
            books_cols = ['ISBN'] + [c for c in self.books_df.columns if c != 'ISBN']
            self.books_df = self.books_df[books_cols]
            
            ratings_cols = ['ISBN'] + [c for c in self.ratings_df.columns if c != 'ISBN']
            self.ratings_df = self.ratings_df[ratings_cols]
            
            with tqdm(total=2, desc="Saving CSVs") as pbar:
                self.books_df.to_csv(books_out, index=False)
                pbar.update(1)
                
                self.ratings_df.to_csv(ratings_out, index=False)
                pbar.update(1)
                
            logging.info(f"{Fore.GREEN}Successfully saved processed data to {self.output_dir}{Style.RESET_ALL}")
            
        except Exception as e:
            logging.error(f"{Fore.RED}Error saving data: {e}{Style.RESET_ALL}")
            raise

    def _print_stats(self, stage):
        logging.info(f"\n{Fore.YELLOW}--- Dataset Statistics ({stage}) ---{Style.RESET_ALL}")
        logging.info(f"Books: {len(self.books_df):,} rows")
        logging.info(f"Ratings: {len(self.ratings_df):,} rows")
        if self.genres_df is not None and stage == "Before Processing":
            logging.info(f"Genres: {len(self.genres_df):,} rows")
        logging.info(f"{Fore.YELLOW}--------------------------------------{Style.RESET_ALL}\n")

    def run(self):
        try:
            self.load_data()
            self.assign_isbns()
            self.merge_genres()
            self.process_ratings()
            self.validate()
            self.save_data()
            logging.info(f"{Fore.GREEN}Dataset building pipeline completed successfully!{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"{Fore.RED}Pipeline failed: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
    RAW_DIR = BASE_DIR / "data" / "raw-dataset-books"
    PROCESSED_DIR = BASE_DIR / "data" / "processed-dataset"
    
    BOOKS_PATH = RAW_DIR / "Books_df.csv"
    GENRES_PATH = RAW_DIR / "Genre_df.csv"
    RATINGS_PATH = RAW_DIR / "Ratings.csv"
    
    builder = DatasetBuilder(
        books_path=BOOKS_PATH,
        genres_path=GENRES_PATH,
        ratings_path=RATINGS_PATH,
        output_dir=PROCESSED_DIR
    )
    
    builder.run()
