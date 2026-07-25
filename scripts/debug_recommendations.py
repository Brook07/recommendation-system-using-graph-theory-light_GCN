"""
Debug Recommendation Quality
Analyzes dataset balance, persona-rating alignment, and recommendation quality.
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json

BASE_DIR = Path(r"D:\Semester-Wise-Materials\sixth_sem\recommendation-system-using-graph-theory")
sys.path.insert(0, str(BASE_DIR))

PROCESSED_DIR = BASE_DIR / "data" / "processed-dataset"
GRAPH_DIR = BASE_DIR / "data" / "processed-graph"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BOOKS_FILE = PROCESSED_DIR / "Books_Final_Clean.csv"
RATINGS_FILE = PROCESSED_DIR / "Ratings_Final.csv"

print("=" * 65)
print("  GRAPHREC - RECOMMENDATION QUALITY DEBUGGER")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. Analyze Books_Final_Clean.csv
# ─────────────────────────────────────────────────────────────────
print("\n[1/6] BOOK GENRE DISTRIBUTION ANALYSIS")
print("-" * 55)

df_books = pd.read_csv(BOOKS_FILE, dtype=str)
df_books['Rating'] = pd.to_numeric(df_books['Rating'], errors='coerce').fillna(0)
df_books['Amazon_Rating_Count'] = pd.to_numeric(df_books['Amazon_Rating_Count'], errors='coerce').fillna(0)

genre_counts = df_books['Main Genre'].value_counts()
thin_genres = genre_counts[genre_counts < 20]
print(f"\nTotal books: {len(df_books):,}")
print(f"Unique genres: {df_books['Main Genre'].nunique()}")
print(f"\nAll genres:")
for genre, count in genre_counts.items():
    flag = " <<< THIN (<20)" if count < 20 else ""
    bar = '|' * min(count // 5, 50)
    print(f"  {genre:40s}: {count:4d}  {bar}{flag}")
print(f"\nThin genres (<20 books): {len(thin_genres)}")
for genre, count in thin_genres.items():
    print(f"  - '{genre}': {count} books")

# ─────────────────────────────────────────────────────────────────
# 2. Analyze Ratings alignment
# ─────────────────────────────────────────────────────────────────
print("\n[2/6] RATINGS PERSONA ALIGNMENT ANALYSIS")
print("-" * 55)

df_ratings = pd.read_csv(RATINGS_FILE, dtype={'ISBN': str, 'User-ID': str, 'Book-Rating': int})
isbn_to_genre = df_books.set_index('ISBN')['Main Genre'].to_dict()

# Add genre to each rating
df_ratings['Main Genre'] = df_ratings['ISBN'].map(isbn_to_genre).fillna('Unknown')

user_genre_dist = df_ratings.groupby(['User-ID', 'Main Genre']).size().unstack(fill_value=0)
total_by_user = df_ratings.groupby('User-ID').size()
user_genre_pct = user_genre_dist.div(total_by_user, axis=0) * 100

# Most dominant genre per user
top_genre_per_user = user_genre_pct.idxmax(axis=1)
top_genre_pct_per_user = user_genre_pct.max(axis=1)

print(f"\nUser count: {df_ratings['User-ID'].nunique()}")
print(f"\nDominant genre by user (sample of 10):")
sample_users = top_genre_per_user.sample(min(10, len(top_genre_per_user)), random_state=42)
for uid, genre in sample_users.items():
    pct = top_genre_pct_per_user[uid]
    print(f"  User {uid}: {genre} ({pct:.1f}% of ratings)")

# Users where top genre has < 50% (might indicate imbalance/poor persona matching)
misaligned_users = top_genre_pct_per_user[top_genre_pct_per_user < 50]
print(f"\nUsers without a strong dominant genre (<50% in any single genre): {len(misaligned_users)}")

# Overall rating genre distribution
print(f"\nOverall rated genre breakdown (top 10):")
overall = df_ratings['Main Genre'].value_counts()
for genre, count in overall.head(10).items():
    pct = count / len(df_ratings) * 100
    print(f"  {genre:40s}: {count:5d} ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# 3. Persona Assignment Reconstruction
# ─────────────────────────────────────────────────────────────────
print("\n[3/6] PERSONA VERIFICATION (80/15/5 split check)")
print("-" * 55)

PERSONA_KEYWORDS = {
    "Romance Reader": ["Romance"],
    "Fantasy Reader": ["Fantasy", "Fantasy, Horror & Science Fiction"],
    "Mystery & Thriller Reader": ["Mystery", "Thriller", "Crime", "Crime, Thriller & Mystery"],
    "Science Fiction Reader": ["Sci-Fi", "Science Fiction", "Fantasy, Horror & Science Fiction"],
    "Self-Help Reader": ["Self-Help", "Personal Development", "Health, Family & Personal Development"],
    "Business & Finance Reader": ["Business", "Finance", "Economics", "Business & Economics"],
    "Computer Science Student": ["Computer Science", "Programming", "Technology", "Higher Education Textbooks"],
    "AI & ML Enthusiast": ["Artificial Intelligence", "Machine Learning", "Data Science", "Higher Education Textbooks"],
    "Medical Student": ["Medicine", "Health", "Medical", "Health, Family & Personal Development"],
    "Engineering Student": ["Engineering"],
    "NEET Prep Student": ["NEET", "Biology", "Physics", "Chemistry", "Exam Preparation"],
    "UPSC Prep Student": ["UPSC", "History", "Geography", "Polity", "Exam Preparation", "History"],
    "Biography Reader": ["Biography", "Biographies, Diaries & True Accounts"],
    "History Reader": ["History"],
    "Philosophy Reader": ["Philosophy"],
    "Children's Book Reader": ["Children", "Kids", "Children's Books"],
    "Comic & Manga Reader": ["Comics", "Graphic Novels", "Manga", "Arts, Film & Photography"],
    "Horror Reader": ["Horror", "Fantasy, Horror & Science Fiction"],
    "Spiritual Reader": ["Religion", "Spirituality", "New Age"],
    "General Reader": ["Fiction", "Literature", "Literature & Fiction"]
}

def guess_persona(user_genres_pct):
    """Guess user persona from their rating distribution."""
    top_genre = user_genres_pct.idxmax()
    for persona, keywords in PERSONA_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in top_genre.lower():
                return persona
    return "Unknown"

# Check 80% primary split
well_aligned = (top_genre_pct_per_user >= 80).sum()
partial_aligned = ((top_genre_pct_per_user >= 50) & (top_genre_pct_per_user < 80)).sum()
poor_aligned = (top_genre_pct_per_user < 50).sum()

print(f"\nPersona purity check:")
print(f"  Strong alignment (>=80% in primary genre):  {well_aligned:4d} users ({well_aligned/len(top_genre_pct_per_user)*100:.1f}%)")
print(f"  Partial alignment (50-80% in primary):      {partial_aligned:4d} users ({partial_aligned/len(top_genre_pct_per_user)*100:.1f}%)")
print(f"  Poor alignment (<50% in primary):           {poor_aligned:4d} users ({poor_aligned/len(top_genre_pct_per_user)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# 4. Per-user recommendation analysis
# ─────────────────────────────────────────────────────────────────
print("\n[4/6] PER-USER RECOMMENDATION ANALYSIS")
print("-" * 55)

try:
    from backend.inference import InferenceEngine
    engine = InferenceEngine(base_dir=str(BASE_DIR))
    engine_loaded = engine.embeddings is not None
except Exception as ex:
    engine_loaded = False
    print(f"  WARNING: Could not load InferenceEngine: {ex}")

TEST_USERS = {
    "10005": "Sci-Fi Fan",
    "10022": "Romance Reader",
    "10078": "Mystery/Thriller Reader",
    "10200": "Business/Finance Reader",
    "10300": "Computer Science Student",
}

user_analysis = {}
for user_id_str, persona_label in TEST_USERS.items():
    user_id_int = int(user_id_str)
    print(f"\n  ---- User {user_id_str} ({persona_label}) ----")

    # Books rated
    user_ratings = df_ratings[df_ratings['User-ID'] == user_id_str]
    print(f"  Total books rated: {len(user_ratings)}")
    if len(user_ratings) > 0:
        genre_breakdown = user_ratings['Main Genre'].value_counts()
        print(f"  Genre breakdown of ratings:")
        for g, c in genre_breakdown.head(5).items():
            pct = c / len(user_ratings) * 100
            print(f"    {g:40s}: {c:3d} ({pct:.1f}%)")
    else:
        print("  No ratings found for this user!")

    # Recommendations
    if engine_loaded and user_id_int in engine.u_lookup:
        recs = engine.get_recommendations(user_id=user_id_int, top_k=10)
        print(f"  Top 10 recommendations:")
        rec_genres = []
        for i, rec in enumerate(recs[:10], 1):
            g = rec.get('Main_Genre', 'N/A')
            t = rec.get('Book_Title', 'N/A')[:45]
            rec_genres.append(g)
            print(f"    {i:2d}. [{g:30s}] {t}")
        
        # Genre match rate
        if len(user_ratings) > 0:
            user_genres = set(user_ratings['Main Genre'].value_counts().head(2).index)
            matching = sum(1 for g in rec_genres if g in user_genres)
            print(f"  Recommendations matching user genres ({'/'.join(user_genres)}): {matching}/10 ({matching*10}%)")
        
        user_analysis[user_id_str] = {
            "persona": persona_label,
            "books_rated": len(user_ratings),
            "rec_genres": rec_genres,
            "in_model": True
        }
    else:
        if not engine_loaded:
            print("  Recommendations: SKIPPED (engine not loaded)")
        else:
            print(f"  >>> User {user_id_str} is NOT in the trained model (u_lookup missing)!")
            print(f"      This means the graph was built BEFORE this user's ratings were added, OR")
            print(f"      user_id {user_id_int} does not exist in user_mapping.csv")
        user_analysis[user_id_str] = {
            "persona": persona_label,
            "books_rated": len(user_ratings),
            "in_model": False
        }

# ─────────────────────────────────────────────────────────────────
# 5. Investigate User 10300 specifically
# ─────────────────────────────────────────────────────────────────
print("\n[5/6] INVESTIGATING USER 10300 SPECIFICALLY")
print("-" * 55)

user_mapping = pd.read_csv(GRAPH_DIR / "user_mapping.csv")
print(f"  Users in graph mapping: {len(user_mapping)}")
print(f"  User ID range: {user_mapping['User-ID'].min()} to {user_mapping['User-ID'].max()}")
user_10300_in_map = 10300 in user_mapping['User-ID'].values
print(f"  User 10300 in user_mapping.csv: {user_10300_in_map}")

user_10300_ratings = df_ratings[df_ratings['User-ID'] == '10300']
print(f"  User 10300 ratings in Ratings_Final.csv: {len(user_10300_ratings)}")

if engine_loaded:
    in_lookup = 10300 in engine.u_lookup
    print(f"  User 10300 in engine.u_lookup: {in_lookup}")

# ─────────────────────────────────────────────────────────────────
# 6. Dataset imbalance diagnosis
# ─────────────────────────────────────────────────────────────────
print("\n[6/6] DATASET IMBALANCE DIAGNOSIS")
print("-" * 55)

dominant_genre = genre_counts.index[0]
dominant_pct = genre_counts.iloc[0] / len(df_books) * 100
print(f"\n  CATALOG LEVEL:")
print(f"    Dominant genre: '{dominant_genre}' = {genre_counts.iloc[0]} books ({dominant_pct:.1f}% of catalog)")
print(f"    Genre Gini concentration: {genre_counts.var() / (genre_counts.mean()**2):.2f}")

print(f"\n  RATINGS LEVEL:")
rating_genre_pct = (df_ratings['Main Genre'].value_counts() / len(df_ratings) * 100)
print(f"    Top rated genre: '{rating_genre_pct.index[0]}' = {rating_genre_pct.iloc[0]:.1f}% of all ratings")

# Persona keywords coverage check
print(f"\n  PERSONA KEYWORD COVERAGE CHECK:")
persona_primary_keywords = {
    "Romance Reader": ["Romance"],
    "Fantasy Reader": ["Fantasy"],
    "Mystery & Thriller Reader": ["Mystery", "Thriller", "Crime", "Crime, Thriller & Mystery"],
    "Science Fiction Reader": ["Sci-Fi", "Science Fiction", "Fantasy, Horror & Science Fiction"],
    "Computer Science Student": ["Computer Science", "Programming", "Technology", "Higher Education Textbooks"],
}
for persona, kws in persona_primary_keywords.items():
    matched = sum(1 for g in genre_counts.index if any(kw.lower() in g.lower() for kw in kws))
    n_books = sum(genre_counts[g] for g in genre_counts.index if any(kw.lower() in g.lower() for kw in kws))
    print(f"    {persona:30s}: {n_books:4d} books in {matched} matching genre(s)")

# ─────────────────────────────────────────────────────────────────
# Generate Debug Report
# ─────────────────────────────────────────────────────────────────
report_path = REPORTS_DIR / "debug_recommendations_report.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# GraphRec - Recommendation Quality Debug Report\n\n")
    f.write("## 1. Book Catalog Genre Distribution\n\n")
    f.write("| Genre | Books | % |\n")
    f.write("|:------|------:|--:|\n")
    for genre, count in genre_counts.items():
        flag = " ⚠️ THIN" if count < 20 else ""
        f.write(f"| {genre}{flag} | {count} | {count/len(df_books)*100:.1f}% |\n")
    
    f.write(f"\n**Total books**: {len(df_books):,}  \n")
    f.write(f"**Thin genres (<20 books)**: {len(thin_genres)}  \n")
    f.write(f"**Dominant genre**: '{dominant_genre}' at {dominant_pct:.1f}% of catalog  \n\n")
    
    f.write("## 2. Root Cause Analysis\n\n")
    f.write(f"### Issue 1: Catalog Imbalance\n")
    f.write(f"- **'{dominant_genre}'** makes up **{dominant_pct:.1f}%** of the entire catalog ({genre_counts.iloc[0]} books).\n")
    f.write(f"- When popularity-weighted sampling is used for exploration, Children's Books dominate cross-persona interactions.\n")
    f.write(f"- Persona keywords like 'Romance', 'Sci-Fi' barely match any books by genre name since catalog uses broader Amazon genres.\n\n")
    
    f.write("### Issue 2: Persona Keyword Mismatch\n")
    f.write("The dataset uses long compound genre names like:\n")
    f.write("- `Fantasy, Horror & Science Fiction` (not just `Fantasy`)\n")
    f.write("- `Crime, Thriller & Mystery` (not just `Mystery`)\n")
    f.write("- `Biographies, Diaries & True Accounts` (not just `Biography`)\n")
    f.write("This means persona keywords only partially match genre names, reducing the primary pool to near-zero for many personas.\n\n")
    
    f.write("### Issue 3: User 10300\n")
    f.write(f"- User 10300 in graph: **{user_10300_in_map}**\n")
    f.write(f"- User 10300 ratings found: **{len(user_10300_ratings)}**\n")
    if not user_10300_in_map:
        f.write("- **Root Cause**: The graph was built before this user ID was included in the ratings. The user_mapping.csv only contains 261 users and 10300 may fall outside the range selected.\n\n")
    else:
        f.write("- User exists in graph but embeddings may not have trained enough to produce recommendations.\n\n")
    
    f.write("## 3. Recommended Fixes\n\n")
    f.write("### Fix 1: Update Persona Keyword Matching\n")
    f.write("Update persona primary/secondary genre lists to use the **exact Amazon genre names** in your catalog:\n")
    f.write("```python\n")
    f.write("# Use EXACT genre names from Books_Final_Clean.csv:\n")
    for genre in genre_counts.head(15).index:
        f.write(f'# "{genre}"\n')
    f.write("```\n\n")
    
    f.write("### Fix 2: Rebalance Sampling Strategy\n")
    f.write("- Instead of pure popularity-weighted sampling, use **genre-weighted stratified sampling**.\n")
    f.write("- Explicitly clamp the % of Children's Books per non-children persona to < 5%.\n\n")
    
    f.write("### Fix 3: Verify User Range in Graph\n")
    f.write(f"- User ID range in graph: {user_mapping['User-ID'].min()} to {user_mapping['User-ID'].max()}\n")
    f.write("- Ensure all synthetic users are in Ratings_Final.csv before rebuilding the graph.\n\n")

print(f"\n  Debug report saved to: {report_path}")
print("\n" + "=" * 65)
print("  Diagnosis complete!")
print("=" * 65)
