"""
FIXED Synthetic Ratings Generator v2

Fixes applied:
1. Personas now use EXACT genre names from Books_Final_Clean.csv
2. Balanced sampling - Children's Books are capped for non-children personas
3. User range starts at 10000 and now generates enough users (no out-of-range)
4. Each persona has its own capped exploration pool to prevent catalog bias
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import math

np.random.seed(42)
random.seed(42)

DATA_DIR = Path("data/processed-dataset")
INPUT_FILE = DATA_DIR / "Books_Final_Clean.csv"
OUTPUT_FILE = DATA_DIR / "Ratings_Final.csv"

print("Loading books...")
df_books = pd.read_csv(INPUT_FILE, dtype=str)
df_books['Rating'] = pd.to_numeric(df_books['Rating'], errors='coerce').fillna(0)
df_books['Amazon_Rating_Count'] = pd.to_numeric(df_books['Amazon_Rating_Count'], errors='coerce').fillna(0)
df_books['Main Genre'] = df_books['Main Genre'].fillna("Unknown")
df_books['Sub Genre'] = df_books['Sub Genre'].fillna("Unknown")
df_books = df_books.reset_index(drop=True)

# Print genre distribution for verification
print("\nGenre distribution in catalog:")
for genre, cnt in df_books['Main Genre'].value_counts().items():
    print(f"  {genre:45s}: {cnt}")

# Popularity-weighted pick probability per book
weighted_score = (df_books['Rating'] + 1) * np.log1p(df_books['Amazon_Rating_Count'] + 1)
df_books['Pick_Prob'] = weighted_score / weighted_score.sum()

# ─────────────────────────────────────────
# EXACT GENRE PERSONAS (matching Books_Final_Clean.csv genre names)
# ─────────────────────────────────────────
PERSONAS = {
    "Children's Book Reader": {
        "primary":   ["Children's Books"],
        "secondary": ["Language, Linguistics & Writing", "Literature & Fiction"],
    },
    "Fantasy / Horror / Sci-Fi Reader": {
        "primary":   ["Fantasy, Horror & Science Fiction"],
        "secondary": ["Literature & Fiction", "Children's Books"],
    },
    "Business & Economics Reader": {
        "primary":   ["Business & Economics"],
        "secondary": ["Health, Family & Personal Development", "Higher Education Textbooks"],
    },
    "Arts & Photography Enthusiast": {
        "primary":   ["Arts, Film & Photography"],
        "secondary": ["Crafts, Home & Lifestyle", "Language, Linguistics & Writing"],
    },
    "Crafts & Lifestyle Reader": {
        "primary":   ["Crafts, Home & Lifestyle"],
        "secondary": ["Health, Family & Personal Development", "Business & Economics"],
    },
    "Higher Education Student": {
        "primary":   ["Higher Education Textbooks"],
        "secondary": ["Engineering", "Exam Preparation"],
    },
    "Biography & History Reader": {
        "primary":   ["Biographies, Diaries & True Accounts"],
        "secondary": ["History", "Language, Linguistics & Writing"],
    },
    "Exam Preparation Student": {
        "primary":   ["Exam Preparation"],
        "secondary": ["Higher Education Textbooks", "Engineering"],
    },
    "Engineering Student": {
        "primary":   ["Engineering"],
        "secondary": ["Higher Education Textbooks", "Exam Preparation"],
    },
    "Health & Self-Help Reader": {
        "primary":   ["Health, Family & Personal Development"],
        "secondary": ["Business & Economics", "Crafts, Home & Lifestyle"],
    },
    "Crime & Mystery Reader": {
        "primary":   ["Crime, Thriller & Mystery"],
        "secondary": ["Fantasy, Horror & Science Fiction", "Literature & Fiction"],
    },
    "History Reader": {
        "primary":   ["History"],
        "secondary": ["Biographies, Diaries & True Accounts", "Language, Linguistics & Writing"],
    },
    "Language & Writing Enthusiast": {
        "primary":   ["Language, Linguistics & Writing"],
        "secondary": ["Literature & Fiction", "Arts, Film & Photography"],
    },
    "Law & Governance Reader": {
        "primary":   ["Law"],
        "secondary": ["Exam Preparation", "Higher Education Textbooks"],
    },
    "Literature & Fiction Reader": {
        "primary":   ["Literature & Fiction"],
        "secondary": ["Fantasy, Horror & Science Fiction", "Biographies, Diaries & True Accounts"],
    },
}

# Get all genre names
ALL_GENRES = df_books['Main Genre'].unique().tolist()

def build_persona_pools(persona_data):
    """Pre-index books for each persona, returning primary/secondary/unrelated pools."""
    primary_genres = set(persona_data['primary'])
    secondary_genres = set(persona_data['secondary'])

    pm = df_books['Main Genre'].isin(primary_genres)
    sm = df_books['Main Genre'].isin(secondary_genres) & ~pm
    um = ~pm & ~sm

    return {
        'primary': df_books[pm].index.tolist(),
        'secondary': df_books[sm].index.tolist(),
        'unrelated': df_books[um].index.tolist(),
    }

print("\nBuilding persona book pools...")
persona_pools = {name: build_persona_pools(data) for name, data in PERSONAS.items()}

for name, pools in persona_pools.items():
    print(f"  {name:45s}: primary={len(pools['primary']):4d}  secondary={len(pools['secondary']):4d}  unrelated={len(pools['unrelated']):4d}")

# ─────────────────────────────────────────
# Rating distributions per tier
# ─────────────────────────────────────────
def get_rating(tier):
    if tier == 'primary':
        return np.random.choice([5, 4, 3], p=[0.55, 0.30, 0.15])
    elif tier == 'secondary':
        return np.random.choice([5, 4, 3, 2], p=[0.25, 0.40, 0.25, 0.10])
    else:
        return np.random.choice([3, 2, 1], p=[0.40, 0.35, 0.25])

def sample_books_weighted(indices, n, exclude_dominant_genre=None, dominant_cap=3):
    """Sample books by popularity weight; optionally cap a dominant genre."""
    if not indices:
        return []
    n = min(n, len(indices))
    probs = df_books.loc[indices, 'Pick_Prob'].values.copy()
    s = probs.sum()
    probs = probs / s if s > 0 else np.ones(len(indices)) / len(indices)
    chosen = np.random.choice(indices, size=min(n * 3, len(indices)), replace=False, p=probs)
    # Apply cap on dominant genre for unrelated sampling
    if exclude_dominant_genre:
        cap_count = 0
        filtered = []
        for idx in chosen:
            if df_books.at[idx, 'Main Genre'] == exclude_dominant_genre:
                if cap_count < dominant_cap:
                    filtered.append(idx)
                    cap_count += 1
            else:
                filtered.append(idx)
        chosen = filtered[:n]
    else:
        chosen = list(chosen[:n])
    return chosen

# ─────────────────────────────────────────
# Generate ratings
# ─────────────────────────────────────────
# Aim for ~20 ratings per book
target_ratings = len(df_books) * 20
avg_ratings_per_user = 60  # midpoint of 20-100 range
USER_COUNT = math.ceil(target_ratings / avg_ratings_per_user)
USER_COUNT = max(USER_COUNT, 500)  # enforce at least 500 users for diversity

print(f"\nGenerating ratings for {USER_COUNT} users...")
ratings = []
persona_assignments = []   # track persona per user for personas.csv
persona_names = list(PERSONAS.keys())

for u_offset in tqdm(range(USER_COUNT), desc="Users"):
    u_id = 10000 + u_offset
    persona = random.choice(persona_names)
    p_data  = PERSONAS[persona]
    pools   = persona_pools[persona]
    num_ratings = random.randint(20, 100)

    n_primary   = int(num_ratings * 0.65)
    n_secondary = int(num_ratings * 0.25)
    n_unrelated = num_ratings - n_primary - n_secondary

    dom_cap_genre = "Children's Books" if persona != "Children's Book Reader" else None

    sampled_primary   = sample_books_weighted(pools['primary'],   n_primary)
    sampled_secondary = sample_books_weighted(pools['secondary'],  n_secondary)
    sampled_unrelated = sample_books_weighted(pools['unrelated'],  n_unrelated,
                                              exclude_dominant_genre=dom_cap_genre, dominant_cap=2)

    for b_idx in sampled_primary:
        ratings.append({"User-ID": u_id, "ISBN": df_books.at[b_idx, 'ISBN'], "Book-Rating": get_rating('primary')})
    for b_idx in sampled_secondary:
        ratings.append({"User-ID": u_id, "ISBN": df_books.at[b_idx, 'ISBN'], "Book-Rating": get_rating('secondary')})
    for b_idx in sampled_unrelated:
        ratings.append({"User-ID": u_id, "ISBN": df_books.at[b_idx, 'ISBN'], "Book-Rating": get_rating('unrelated')})

    persona_assignments.append({
        "User-ID":         u_id,
        "Persona":         persona,
        "Primary_Genre":   ", ".join(p_data["primary"]),
        "Secondary_Genre": ", ".join(p_data["secondary"]),
    })

df_ratings = pd.DataFrame(ratings)

print("\n--- VALIDATION ---")
print(f"Total Users:           {df_ratings['User-ID'].nunique():,}")
print(f"Total Books:           {len(df_books):,}")
print(f"Books Interacted With: {df_ratings['ISBN'].nunique():,}")
print(f"Total Ratings:         {len(df_ratings):,}")
print(f"Avg Ratings / User:    {len(df_ratings)/df_ratings['User-ID'].nunique():.1f}")
print(f"Avg Ratings / Book:    {len(df_ratings)/len(df_books):.1f}")

print("\nRating Distribution:")
dist = df_ratings['Book-Rating'].value_counts(normalize=True).sort_index(ascending=False) * 100
for r, pct in dist.items():
    print(f"  {int(r)} Stars: {pct:5.1f}%  {'*' * int(pct/2)}")

df_ratings.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")

# Save personas.csv
PERSONAS_FILE = DATA_DIR / "personas.csv"
df_personas = pd.DataFrame(persona_assignments)
df_personas.to_csv(PERSONAS_FILE, index=False)
print(f"Saved persona assignments to {PERSONAS_FILE}")
print(f"\nPersona distribution:")
for p, cnt in df_personas['Persona'].value_counts().items():
    print(f"  {p:45s}: {cnt} users")
