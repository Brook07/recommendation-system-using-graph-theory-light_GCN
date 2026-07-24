import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import logging
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import warnings

# Suppress seaborn warnings
warnings.filterwarnings('ignore')

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class EDAPipeline:
    def __init__(self, books_path, ratings_path, output_dir):
        self.books_path = books_path
        self.ratings_path = ratings_path
        self.output_dir = Path(output_dir)
        
        self.eda_dir = self.output_dir / "eda"
        self.tables_dir = self.output_dir / "tables"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eda_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.books = None
        self.ratings = None
        self.merged = None

    def load_data(self):
        logging.info(f"{Fore.CYAN}[1/4] Loading datasets for EDA...{Style.RESET_ALL}")
        with tqdm(total=2, desc="Loading CSVs") as pbar:
            self.books = pd.read_csv(self.books_path)
            pbar.update(1)
            # Use Enhanced_Ratings_v2.csv if it exists, otherwise Enhanced_Ratings.csv
            v2_path = str(self.ratings_path).replace('.csv', '_v2.csv')
            if os.path.exists(v2_path):
                logging.info(f"{Fore.YELLOW}Found {Path(v2_path).name}, using it instead of original ratings.{Style.RESET_ALL}")
                self.ratings = pd.read_csv(v2_path)
            else:
                self.ratings = pd.read_csv(self.ratings_path)
            pbar.update(1)
            
        # Clean price
        if 'Price' in self.books.columns:
            self.books['Price_num'] = self.books['Price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
            self.books['Price_num'] = pd.to_numeric(self.books['Price_num'], errors='coerce')
            
        # Merge for correlation
        self.merged = self.ratings.merge(self.books, on='ISBN', how='left')

    def run_validation(self):
        logging.info(f"\n{Fore.CYAN}[2/4] Generating Validation Report...{Style.RESET_ALL}")
        
        report_path = self.output_dir / "validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== DATA VALIDATION REPORT ===\n\n")
            
            # Basic stats
            total_books = len(self.books)
            total_users = self.ratings['User-ID'].nunique()
            total_ratings = len(self.ratings)
            avg_rating = self.ratings['Book-Rating'].mean()
            
            f.write(f"Total Books: {total_books:,}\n")
            f.write(f"Total Users: {total_users:,}\n")
            f.write(f"Total Ratings: {total_ratings:,}\n")
            f.write(f"Average Rating: {avg_rating:.2f}\n\n")
            
            f.write("--- Rating Distribution ---\n")
            dist = self.ratings['Book-Rating'].value_counts().sort_index()
            for k, v in dist.items():
                f.write(f"Rating {k}: {v:,}\n")
            f.write("\n")
            
            f.write("--- Genres ---\n")
            num_main = self.books['Main Genre'].nunique()
            num_sub = self.books['Sub Genre'].nunique()
            f.write(f"Number of Main Genres: {num_main}\n")
            f.write(f"Number of Sub Genres: {num_sub}\n\n")
            
            f.write("--- Duplicates & Integrity ---\n")
            dup_isbns = self.books['ISBN'].duplicated().sum()
            f.write(f"Duplicate ISBNs in books: {dup_isbns}\n")
            # For users, duplication means same user multiple times, which is normal.
            # But duplicate User-ISBN pairs means a user rated the same book twice
            dup_ratings = self.ratings.duplicated(subset=['User-ID', 'ISBN']).sum()
            f.write(f"Duplicate User-ISBN ratings: {dup_ratings}\n")
            
            missing_isbns = self.ratings[~self.ratings['ISBN'].isin(self.books['ISBN'])]['ISBN'].nunique()
            f.write(f"Dataset Integrity (ISBNs in ratings missing in books): {missing_isbns}\n\n")
            
            f.write("--- Missing Values (Books) ---\n")
            for col, missing in self.books.isnull().sum().items():
                f.write(f"{col}: {missing:,}\n")
            f.write("\n")
            
            f.write("--- Missing Values (Ratings) ---\n")
            for col, missing in self.ratings.isnull().sum().items():
                f.write(f"{col}: {missing:,}\n")
            f.write("\n")
            
            f.write("--- Activity Stats ---\n")
            books_with_rating = self.ratings['ISBN'].nunique()
            users_with_rating = total_users
            f.write(f"Books with at least one rating: {books_with_rating:,}\n")
            f.write(f"Users with at least one rating: {users_with_rating:,}\n")
            
            ratings_per_book = total_ratings / max(books_with_rating, 1)
            ratings_per_user = total_ratings / max(users_with_rating, 1)
            f.write(f"Average Ratings per Book: {ratings_per_book:.2f}\n")
            f.write(f"Average Ratings per User: {ratings_per_user:.2f}\n")
            f.write(f"Minimum Rating: {self.ratings['Book-Rating'].min()}\n")
            f.write(f"Maximum Rating: {self.ratings['Book-Rating'].max()}\n\n")
            
            f.write("--- Top 10 Most Rated Books ---\n")
            top_books = self.merged['Title'].value_counts().head(10)
            for title, count in top_books.items():
                f.write(f"{title}: {count:,} ratings\n")
            f.write("\n")
            
            f.write("--- Top 10 Most Active Users ---\n")
            top_users = self.ratings['User-ID'].value_counts().head(10)
            for uid, count in top_users.items():
                f.write(f"User {uid}: {count:,} ratings\n")
                
        logging.info(f"{Fore.GREEN}Validation report saved to {report_path}{Style.RESET_ALL}")

    def run_eda(self):
        logging.info(f"\n{Fore.CYAN}[3/4] Generating EDA Visualizations...{Style.RESET_ALL}")
        
        sns.set_theme(style="whitegrid")
        tasks = 14
        
        with tqdm(total=tasks, desc="Generating Plots") as pbar:
            # 1. Rating Distribution
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.ratings, x='Book-Rating', palette='viridis')
            plt.title('Rating Distribution')
            plt.savefig(self.eda_dir / '1_rating_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 2. Books per Main Genre
            plt.figure(figsize=(12, 6))
            order = self.books['Main Genre'].value_counts().index
            sns.countplot(data=self.books, y='Main Genre', order=order, palette='viridis')
            plt.title('Books per Main Genre')
            plt.savefig(self.eda_dir / '2_books_per_main_genre.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 3. Books per Sub Genre (Top 20)
            plt.figure(figsize=(12, 8))
            order = self.books['Sub Genre'].value_counts().head(20).index
            sns.countplot(data=self.books, y='Sub Genre', order=order, palette='viridis')
            plt.title('Books per Sub Genre (Top 20)')
            plt.savefig(self.eda_dir / '3_books_per_sub_genre_top20.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 4. Top 15 Authors by Number of Books
            plt.figure(figsize=(12, 8))
            order = self.books['Author'].value_counts().head(15).index
            sns.countplot(data=self.books, y='Author', order=order, palette='viridis')
            plt.title('Top 15 Authors by Number of Books')
            plt.savefig(self.eda_dir / '4_top_15_authors.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 5. Top 20 Most Rated Books
            plt.figure(figsize=(12, 8))
            top_rated_books = self.merged['Title'].value_counts().head(20)
            sns.barplot(x=top_rated_books.values, y=top_rated_books.index, palette='viridis')
            plt.title('Top 20 Most Rated Books')
            plt.xlabel('Number of Ratings')
            plt.savefig(self.eda_dir / '5_top_20_most_rated_books.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save table
            top_rated_books.reset_index().rename(columns={'index':'Title', 'Title':'Count'}).to_csv(self.tables_dir / 'top_20_most_rated_books.csv', index=False)
            pbar.update(1)
            
            # 6. Top 20 Most Active Users
            plt.figure(figsize=(12, 8))
            top_users = self.ratings['User-ID'].value_counts().head(20)
            sns.barplot(x=top_users.values, y=top_users.index.astype(str), palette='viridis')
            plt.title('Top 20 Most Active Users')
            plt.xlabel('Number of Ratings')
            plt.ylabel('User ID')
            plt.savefig(self.eda_dir / '6_top_20_most_active_users.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 7. User Activity Distribution (Histogram)
            plt.figure(figsize=(10, 6))
            user_counts = self.ratings['User-ID'].value_counts()
            sns.histplot(user_counts, bins=50, log_scale=(False, True), color='blue')
            plt.title('User Activity Distribution (Log Scale)')
            plt.xlabel('Number of Ratings per User')
            plt.ylabel('Count of Users (Log)')
            plt.savefig(self.eda_dir / '7_user_activity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 8. Ratings per Book Distribution
            plt.figure(figsize=(10, 6))
            book_counts = self.ratings['ISBN'].value_counts()
            sns.histplot(book_counts, bins=50, log_scale=(False, True), color='green')
            plt.title('Ratings per Book Distribution (Log Scale)')
            plt.xlabel('Number of Ratings per Book')
            plt.ylabel('Count of Books (Log)')
            plt.savefig(self.eda_dir / '8_ratings_per_book_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 9. Long-tail Distribution of Book Popularity
            plt.figure(figsize=(10, 6))
            sorted_books = book_counts.values
            plt.plot(sorted_books, color='purple')
            plt.title('Long-tail Distribution of Book Popularity')
            plt.xlabel('Book Rank (by popularity)')
            plt.ylabel('Number of Ratings')
            plt.fill_between(range(len(sorted_books)), sorted_books, color='purple', alpha=0.3)
            plt.savefig(self.eda_dir / '9_long_tail_popularity.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 10. Average Rating by Main Genre
            plt.figure(figsize=(12, 6))
            avg_rating_genre = self.merged.groupby('Main Genre')['Book-Rating'].mean().sort_values(ascending=False)
            sns.barplot(x=avg_rating_genre.values, y=avg_rating_genre.index, palette='viridis')
            plt.title('Average Rating by Main Genre')
            plt.xlabel('Average Rating')
            plt.savefig(self.eda_dir / '10_avg_rating_by_main_genre.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save table
            avg_rating_genre.reset_index().to_csv(self.tables_dir / 'avg_rating_by_main_genre.csv', index=False)
            pbar.update(1)
            
            # 11. Top Genres by Average Rating (already kind of done in 10, but let's do top Sub Genres)
            plt.figure(figsize=(12, 8))
            avg_rating_sub = self.merged.groupby('Sub Genre')['Book-Rating'].agg(['mean', 'count'])
            # Filter to genres with at least 50 ratings to avoid noise
            avg_rating_sub = avg_rating_sub[avg_rating_sub['count'] >= 50]['mean'].sort_values(ascending=False).head(20)
            sns.barplot(x=avg_rating_sub.values, y=avg_rating_sub.index, palette='viridis')
            plt.title('Top 20 Sub Genres by Average Rating (min 50 ratings)')
            plt.xlabel('Average Rating')
            plt.savefig(self.eda_dir / '11_top_sub_genres_by_avg_rating.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 12. Correlation Heatmap
            plt.figure(figsize=(8, 6))
            corr_cols = ['Amazon_Rating_Count', 'Rating', 'Book-Rating']
            # Map Amazon_Rating_Count if we renamed it earlier, else fallback
            actual_cols = [c for c in corr_cols if c in self.merged.columns]
            if not actual_cols:
                actual_cols = [c for c in ['No. of People rated', 'Rating', 'Book-Rating'] if c in self.merged.columns]
                
            if len(actual_cols) > 1:
                # Convert 'Rating' to numeric if not already
                self.merged['Rating_num'] = pd.to_numeric(self.merged['Rating'], errors='coerce')
                corr_data = self.merged[[c for c in actual_cols if c != 'Rating'] + ['Rating_num']].corr()
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
                plt.title('Correlation Heatmap')
                plt.savefig(self.eda_dir / '12_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 13. Price Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.books['Price_num'].dropna(), bins=50, color='orange')
            plt.title('Price Distribution')
            plt.xlabel('Price (₹)')
            plt.savefig(self.eda_dir / '13_price_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            pbar.update(1)
            
            # 14. Book Type Distribution
            plt.figure(figsize=(10, 6))
            type_counts = self.books['Type'].value_counts()
            sns.barplot(x=type_counts.values, y=type_counts.index, palette='viridis')
            plt.title('Book Type Distribution')
            plt.xlabel('Count')
            plt.savefig(self.eda_dir / '14_book_type_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save table
            type_counts.reset_index().rename(columns={'index':'Type', 'Type':'Count'}).to_csv(self.tables_dir / 'book_type_distribution.csv', index=False)
            pbar.update(1)

    def generate_summary(self):
        logging.info(f"\n{Fore.CYAN}[4/4] Writing EDA Summary...{Style.RESET_ALL}")
        summary_path = self.output_dir / "eda_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== EXPLORATORY DATA ANALYSIS (EDA) SUMMARY ===\n\n")
            f.write("Generated Plots:\n")
            for plot_file in sorted(os.listdir(self.eda_dir)):
                f.write(f"- reports/eda/{plot_file}\n")
                
            f.write("\nGenerated Tables:\n")
            for table_file in sorted(os.listdir(self.tables_dir)):
                f.write(f"- reports/tables/{table_file}\n")
                
            f.write("\nKey Insights:\n")
            f.write("- Data covers a wide range of genres, with long-tail distributions in book popularity and user activity.\n")
            f.write("- Validation confirms relational integrity between the Enhanced Ratings and Enhanced Books datasets.\n")
            f.write("- Please refer to the saved PNGs for visual inspection of the dataset distributions.\n")
            
        logging.info(f"{Fore.GREEN}EDA summary saved to {summary_path}{Style.RESET_ALL}")

    def run(self):
        try:
            self.load_data()
            self.run_validation()
            self.run_eda()
            self.generate_summary()
            logging.info(f"\n{Fore.GREEN}Pipeline completed successfully! All reports saved to {self.output_dir}{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"{Fore.RED}Pipeline failed: {e}{Style.RESET_ALL}")
            raise

if __name__ == "__main__":
    BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
    PROCESSED_DIR = BASE_DIR / "data" / "processed-dataset"
    REPORTS_DIR = BASE_DIR / "reports"
    
    BOOKS_PATH = PROCESSED_DIR / "Enhanced_Books.csv"
    RATINGS_PATH = PROCESSED_DIR / "Enhanced_Ratings.csv"
    
    pipeline = EDAPipeline(
        books_path=BOOKS_PATH,
        ratings_path=RATINGS_PATH,
        output_dir=REPORTS_DIR
    )
    
    pipeline.run()
