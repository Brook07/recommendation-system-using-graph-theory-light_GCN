import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_synthetic_ratings():
    BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
    PROCESSED_DIR = BASE_DIR / "data" / "processed-dataset"
    
    BOOKS_PATH = PROCESSED_DIR / "Enhanced_Books.csv"
    RATINGS_PATH = PROCESSED_DIR / "Enhanced_Ratings.csv"
    OUT_PATH = PROCESSED_DIR / "Enhanced_Ratings_v2.csv"
    
    logging.info(f"{Fore.CYAN}[1/4] Loading processed datasets...{Style.RESET_ALL}")
    books = pd.read_csv(BOOKS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    
    total_ratings = len(ratings)
    initial_zeros = (ratings['Book-Rating'] == 0).sum()
    initial_non_zeros = total_ratings - initial_zeros
    
    logging.info(f"Total ratings: {total_ratings:,}")
    logging.info(f"Initial zero ratings (implicit): {initial_zeros:,}")
    logging.info(f"Initial non-zero ratings: {initial_non_zeros:,}")
    
    logging.info(f"\n{Fore.CYAN}[2/4] Merging metadata for synthetic generation...{Style.RESET_ALL}")
    # Prepare metadata
    books['Rating'] = pd.to_numeric(books['Rating'], errors='coerce').fillna(3.5)
    books['Amazon_Rating_Count'] = pd.to_numeric(books['Amazon_Rating_Count'], errors='coerce').fillna(0)
    
    # Merge book metadata into ratings to guide generation
    ratings = ratings.merge(books[['ISBN', 'Rating', 'Amazon_Rating_Count']], on='ISBN', how='left')
    
    logging.info(f"\n{Fore.CYAN}[3/4] Generating realistic synthetic ratings...{Style.RESET_ALL}")
    
    zero_mask = ratings['Book-Rating'] == 0
    
    # 1. Keep around 15% of the original 0 ratings to simulate implicit interactions
    keep_zero_ratio = 0.15
    keep_zero_idx = np.random.choice(ratings[zero_mask].index, size=int(keep_zero_ratio * zero_mask.sum()), replace=False)
    
    keep_zero_mask = pd.Series(False, index=ratings.index)
    keep_zero_mask.loc[keep_zero_idx] = True
    
    synth_mask = zero_mask & ~keep_zero_mask
    num_to_generate = synth_mask.sum()
    logging.info(f"Will generate {num_to_generate:,} synthetic ratings.")
    logging.info(f"Will keep {len(keep_zero_idx):,} original zero ratings.")
    
    # 2. Categorize books by popularity to assign base synthetic ratings
    np.random.seed(42) # For reproducibility
    
    high_pop = synth_mask & (ratings['Rating'] >= 4.3) & (ratings['Amazon_Rating_Count'] > 5000)
    avg_pop = synth_mask & ~high_pop & (ratings['Rating'] >= 3.8)
    low_pop = synth_mask & ~high_pop & ~avg_pop
    
    # Assign base ratings with varying probabilities based on popularity
    ratings.loc[high_pop, 'Book-Rating'] = np.random.choice([4.0, 4.5, 5.0], size=high_pop.sum(), p=[0.2, 0.4, 0.4])
    ratings.loc[avg_pop, 'Book-Rating'] = np.random.choice([3.0, 3.5, 4.0, 4.5], size=avg_pop.sum(), p=[0.1, 0.3, 0.4, 0.2])
    ratings.loc[low_pop, 'Book-Rating'] = np.random.choice([2.0, 2.5, 3.0, 3.5], size=low_pop.sum(), p=[0.2, 0.3, 0.3, 0.2])
    
    # Add random noise to make it look natural
    noise = np.random.choice([-0.5, 0.0, 0.5], size=num_to_generate, p=[0.25, 0.5, 0.25])
    ratings.loc[synth_mask, 'Book-Rating'] += noise
    
    # Clip ratings to be strictly between 1.0 and 5.0
    ratings.loc[synth_mask, 'Book-Rating'] = ratings.loc[synth_mask, 'Book-Rating'].clip(1.0, 5.0)
    
    # Ensure ratings are clean (e.g. rounded to 1 decimal place like 4.0, 4.5)
    ratings.loc[synth_mask, 'Book-Rating'] = ratings.loc[synth_mask, 'Book-Rating'].round(1)
    
    # Clean up temporary metadata columns
    ratings = ratings.drop(columns=['Rating', 'Amazon_Rating_Count'])
    
    logging.info(f"\n{Fore.CYAN}[4/4] Validation and Saving...{Style.RESET_ALL}")
    
    final_zeros = (ratings['Book-Rating'] == 0).sum()
    final_non_zeros = total_ratings - final_zeros
    
    logging.info(f"\n{Fore.YELLOW}--- Final Dataset Statistics ---{Style.RESET_ALL}")
    logging.info(f"Synthetic ratings generated: {num_to_generate:,}")
    logging.info(f"Remaining zero ratings: {final_zeros:,}")
    logging.info(f"Total non-zero ratings: {final_non_zeros:,}")
    logging.info(f"\n{Fore.GREEN}Rating Distribution:{Style.RESET_ALL}")
    
    # Print value counts for non-zero ratings
    dist = ratings[ratings['Book-Rating'] > 0]['Book-Rating'].value_counts().sort_index()
    for rating_val, count in dist.items():
        logging.info(f"Rating {rating_val}: {count:,} ({count/final_non_zeros*100:.1f}%)")
        
    logging.info(f"{Fore.YELLOW}--------------------------------{Style.RESET_ALL}\n")
    
    # Save the modified ratings to a new v2 file
    ratings.to_csv(OUT_PATH, index=False)
    logging.info(f"{Fore.GREEN}Successfully saved synthetic dataset to {OUT_PATH}{Style.RESET_ALL}")

if __name__ == "__main__":
    generate_synthetic_ratings()
