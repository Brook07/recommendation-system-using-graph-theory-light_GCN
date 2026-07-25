"""
End-to-End Validation and Final Report Generator.
Run this after training to validate the full pipeline and generate reports.

Usage:
    python scripts/validate_end_to_end.py
"""
import sys
import os
import time
import datetime
from pathlib import Path

BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import torch
import colorama
from colorama import Fore, Style
from collections import defaultdict

colorama.init(autoreset=True)

PROCESSED_DIR = BASE_DIR / "data" / "processed-dataset"
GRAPH_DIR = BASE_DIR / "data" / "processed-graph"
MODELS_DIR = BASE_DIR / "models" / "lightgcn"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BOOKS_FILE = PROCESSED_DIR / "Books_Final_Clean.csv"
RATINGS_FILE = PROCESSED_DIR / "Ratings_Final.csv"
MODEL_FILE = MODELS_DIR / "best_lightgcn_model.pt"

print(f"\n{Fore.CYAN}{'='*60}")
print("  GraphRec End-to-End Validation")
print(f"{'='*60}{Style.RESET_ALL}\n")

# ─────────────────────────────────────────
# 1. Dataset Validation
# ─────────────────────────────────────────
print(f"{Fore.YELLOW}[1/6] Validating datasets...{Style.RESET_ALL}")
df_books = pd.read_csv(BOOKS_FILE, dtype=str)
df_ratings = pd.read_csv(RATINGS_FILE, dtype=str)
df_ratings['Book-Rating'] = pd.to_numeric(df_ratings['Book-Rating'], errors='coerce')

book_isbns = set(df_books['ISBN'].astype(str))
rating_isbns = set(df_ratings['ISBN'].astype(str))
missing_isbns = rating_isbns - book_isbns

assert len(missing_isbns) == 0, f"❌ {len(missing_isbns)} ISBNs in Ratings_Final not found in Books_Final_Clean!"
print(f"{Fore.GREEN}  ✅ All {len(rating_isbns):,} rated ISBNs exist in Books_Final_Clean.{Style.RESET_ALL}")

missing_thumbs = df_books['Google_Thumbnail'].isna() | (df_books['Google_Thumbnail'].str.strip() == "")
print(f"{Fore.GREEN}  ✅ Missing thumbnails in books: {missing_thumbs.sum()} (should be 0){Style.RESET_ALL}")

# ─────────────────────────────────────────
# 2. Graph Validation
# ─────────────────────────────────────────
print(f"\n{Fore.YELLOW}[2/6] Validating graph artifacts...{Style.RESET_ALL}")
assert (GRAPH_DIR / "user_mapping.csv").exists(), "❌ user_mapping.csv missing!"
assert (GRAPH_DIR / "book_mapping.csv").exists(), "❌ book_mapping.csv missing!"
assert (GRAPH_DIR / "edge_index.pt").exists(), "❌ edge_index.pt missing!"
assert (GRAPH_DIR / "normalized_adj.pt").exists(), "❌ normalized_adj.pt missing!"

user_mapping = pd.read_csv(GRAPH_DIR / "user_mapping.csv")
book_mapping = pd.read_csv(GRAPH_DIR / "book_mapping.csv")
edge_index = torch.load(GRAPH_DIR / "edge_index.pt", map_location='cpu')

num_users = len(user_mapping)
num_books = len(book_mapping)
num_edges = edge_index.shape[1] // 2  # undirected

print(f"{Fore.GREEN}  ✅ Graph artifacts found.{Style.RESET_ALL}")
print(f"     Users: {num_users:,}, Books: {num_books:,}, Edges: {num_edges:,}")

# Verify no invalid user mappings
invalid_users = set(user_mapping['User-ID'].astype(str)) - set(df_ratings['User-ID'].astype(str))
print(f"     Invalid user mappings: {len(invalid_users)} (should be 0)")

# ─────────────────────────────────────────
# 3. Model Validation
# ─────────────────────────────────────────
print(f"\n{Fore.YELLOW}[3/6] Validating model...{Style.RESET_ALL}")
if not MODEL_FILE.exists():
    print(f"{Fore.RED}  ❌ best_lightgcn_model.pt not found at {MODEL_FILE}!")
    print(f"  ⚠️  Please run: python scripts/05_train_lightgcn.py")
    # Continue anyway to generate the report with what we have
    model_loaded = False
else:
    try:
        from backend.inference import InferenceEngine
        engine = InferenceEngine(base_dir=str(BASE_DIR))
        if engine.embeddings is not None:
            print(f"{Fore.GREEN}  PASS: InferenceEngine loaded with embeddings.{Style.RESET_ALL}")
            model_loaded = True
        else:
            print(f"{Fore.YELLOW}  WARN: Engine loaded but embeddings are None (run training first).{Style.RESET_ALL}")
            model_loaded = False
    except Exception as e:
        print(f"{Fore.RED}  FAIL: Error loading InferenceEngine: {e}{Style.RESET_ALL}")
        model_loaded = False

# ─────────────────────────────────────────
# 4. End-to-End Recommendation Test
# ─────────────────────────────────────────
print(f"\n{Fore.YELLOW}[4/6] Running end-to-end recommendation tests for 5 personas...{Style.RESET_ALL}")

TEST_USERS = [
    (10005, "Sci-Fi Fan"),
    (10022, "Romance Reader"),
    (10078, "Mystery/Thriller Reader"),
    (10200, "Business/Finance Reader"),
    (10300, "Computer Science Student"),
]

rec_results = {}
for user_id, persona_label in TEST_USERS:
    print(f"\n  {Fore.CYAN}--- User {user_id} ({persona_label}) ---{Style.RESET_ALL}")
    try:
        result = engine.get_recommendations(user_id=user_id, top_k=10)
        if result:
            recs = result
            print(f"  Top {len(recs)} recommendations:")
            genres_found = []
            for i, rec in enumerate(recs[:10], 1):
                title = rec.get('Book_Title', 'N/A')[:50]
                genre = rec.get('Main_Genre', 'N/A')
                score = rec.get('Recommendation_Score', 0)
                has_thumb = bool(rec.get('Book_Cover_URL', ''))
                genres_found.append(genre)
                thumb_icon = "[IMG]" if has_thumb else "[NO IMG]"
                print(f"    {i:2}. {thumb_icon} [{genre:20s}] {title} (score: {score:.4f})")
            rec_results[user_id] = {"persona": persona_label, "genres": genres_found, "count": len(recs)}
        else:
            print(f"  {Fore.YELLOW}  No recommendations (user may not be in the trained model).{Style.RESET_ALL}")
            rec_results[user_id] = {"persona": persona_label, "genres": [], "count": 0}
    except Exception as e:
        print(f"  {Fore.RED}  ❌ Error: {e}{Style.RESET_ALL}")
        rec_results[user_id] = {"persona": persona_label, "genres": [], "count": 0, "error": str(e)}

# ─────────────────────────────────────────
# 5. Genre distribution check
# ─────────────────────────────────────────
print(f"\n{Fore.YELLOW}[5/6] Dataset statistics...{Style.RESET_ALL}")

df_books['Rating'] = pd.to_numeric(df_books['Rating'], errors='coerce').fillna(0)
genre_counts = df_books['Main Genre'].value_counts().head(15)
print(f"\n  Genre Distribution (Top 15):")
for genre, count in genre_counts.items():
    bar = '█' * (count // 2)
    print(f"    {genre:30s}: {count:4d} {bar}")

rating_dist = df_ratings['Book-Rating'].value_counts(normalize=True).sort_index(ascending=False) * 100
print(f"\n  Rating Distribution:")
for r, pct in rating_dist.items():
    bar = '█' * int(pct / 2)
    print(f"    {int(r)} Stars: {pct:5.1f}% {bar}")

# ─────────────────────────────────────────
# 6. Generate Final Report
# ─────────────────────────────────────────
print(f"\n{Fore.YELLOW}[6/6] Generating final pipeline report...{Style.RESET_ALL}")

# Get training metrics if history exists
training_metrics = {}
hist_path = MODELS_DIR / "training_history.csv"
if hist_path.exists():
    hist_df = pd.read_csv(hist_path)
    best_row = hist_df.loc[hist_df['recall'].idxmax()]
    training_metrics = {
        'best_epoch': int(best_row['epoch']),
        'best_recall': float(best_row['recall']),
        'best_precision': float(best_row['precision']),
        'best_ndcg': float(best_row['ndcg']),
        'total_epochs': len(hist_df),
        'total_time': hist_df['time'].sum() if 'time' in hist_df else 'N/A',
        'final_train_loss': float(hist_df['train_loss'].iloc[-1]),
    }
else:
    training_metrics = {"note": "Training history not found. Run 05_train_lightgcn.py first."}

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_path = REPORTS_DIR / "final_pipeline_report.txt"

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  GRAPHREC - FINAL PIPELINE REPORT\n")
    f.write(f"  Generated: {now}\n")
    f.write("=" * 65 + "\n\n")

    f.write("─── DATASET SUMMARY ─────────────────────────────────────────\n")
    f.write(f"  Books (Books_Final_Clean.csv):  {len(df_books):>10,}\n")
    f.write(f"  Users (Ratings_Final.csv):      {df_ratings['User-ID'].nunique():>10,}\n")
    f.write(f"  Ratings (Ratings_Final.csv):    {len(df_ratings):>10,}\n")
    f.write(f"  Avg Ratings / User:             {len(df_ratings)/df_ratings['User-ID'].nunique():>10.1f}\n")
    f.write(f"  Avg Ratings / Book:             {len(df_ratings)/len(df_books):>10.1f}\n")
    f.write(f"  Unique Genres:                  {df_books['Main Genre'].nunique():>10,}\n\n")

    f.write("─── DATASET PATHS ───────────────────────────────────────────\n")
    f.write(f"  Books:   {BOOKS_FILE}\n")
    f.write(f"  Ratings: {RATINGS_FILE}\n")
    f.write(f"  Graph:   {GRAPH_DIR}\n")
    f.write(f"  Model:   {MODEL_FILE}\n\n")

    f.write("─── GRAPH STATISTICS ────────────────────────────────────────\n")
    f.write(f"  Users:    {num_users:,}\n")
    f.write(f"  Books:    {num_books:,}\n")
    f.write(f"  Edges:    {num_edges:,}\n")
    f.write(f"  Density:  {num_edges / (num_users * num_books):.6f}\n\n")

    f.write("─── TRAINING METRICS ────────────────────────────────────────\n")
    if isinstance(training_metrics, dict) and 'best_recall' in training_metrics:
        f.write(f"  Best Epoch:     {training_metrics['best_epoch']}\n")
        f.write(f"  Total Epochs:   {training_metrics['total_epochs']}\n")
        f.write(f"  Recall@10:      {training_metrics['best_recall']:.4f}\n")
        f.write(f"  Precision@10:   {training_metrics['best_precision']:.4f}\n")
        f.write(f"  NDCG@10:        {training_metrics['best_ndcg']:.4f}\n")
        f.write(f"  Final Train Loss: {training_metrics['final_train_loss']:.4f}\n")
        f.write(f"  Total Train Time: {training_metrics['total_time']:.1f}s\n\n")
    else:
        f.write(f"  {training_metrics.get('note', 'N/A')}\n\n")

    f.write("─── GENRE DISTRIBUTION (Top 15) ─────────────────────────────\n")
    for genre, count in genre_counts.items():
        f.write(f"  {genre:35s}: {count:4d}\n")
    f.write("\n")

    f.write("─── RATING DISTRIBUTION ─────────────────────────────────────\n")
    for r, pct in rating_dist.items():
        f.write(f"  {int(r)} Stars: {pct:5.1f}%\n")
    f.write("\n")

    f.write("─── VALIDATION RESULTS ──────────────────────────────────────\n")
    f.write(f"  ISBNs cross-validated: {'✅ PASS' if len(missing_isbns) == 0 else '❌ FAIL'}\n")
    f.write(f"  Missing thumbnails:    {'✅ NONE' if missing_thumbs.sum() == 0 else f'❌ {missing_thumbs.sum()} missing'}\n")
    f.write(f"  Model loaded:          {'✅ YES' if model_loaded else '⚠️  NOT YET - Run training first'}\n\n")

    f.write("─── E2E RECOMMENDATION TESTS ────────────────────────────────\n")
    for uid, result in rec_results.items():
        f.write(f"  User {uid} ({result['persona']}): {result['count']} recs returned\n")
        if result['genres']:
            genre_summary = ", ".join(set(result['genres'][:5]))
            f.write(f"    Genres: {genre_summary}\n")
        if 'error' in result:
            f.write(f"    ERROR: {result['error']}\n")
    f.write("\n")
    f.write("=" * 65 + "\n")

print(f"\n{Fore.GREEN}✅ Final pipeline report saved to: {report_path}{Style.RESET_ALL}")
print(f"\n{Fore.CYAN}{'='*60}")
print("  Validation complete!")
print(f"{'='*60}{Style.RESET_ALL}\n")
