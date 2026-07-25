import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# Set fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
DATA_DIR = Path("data/processed-dataset")
INPUT_FILE = DATA_DIR / "Enhanced_Books_Final.csv"
OUTPUT_FILE = DATA_DIR / "Enhanced_Ratings_v3.csv"
USER_COUNT = 2500  # Will yield approx 2500 * (20+150)/2 = ~212,500 ratings

print(f"Loading books from {INPUT_FILE}...")
df_books = pd.read_csv(INPUT_FILE, dtype={'ISBN': str})
df_books['Rating'] = pd.to_numeric(df_books['Rating'], errors='coerce').fillna(0)
df_books['Amazon_Rating_Count'] = pd.to_numeric(df_books['Amazon_Rating_Count'], errors='coerce').fillna(0)
df_books['Main Genre'] = df_books['Main Genre'].fillna("Unknown")
df_books['Sub Genre'] = df_books['Sub Genre'].fillna("Unknown")

# Precompute book probabilities based on Amazon Rating and Count to simulate popularity
print("Calculating popularity weights...")
# Add a small epsilon to count so unrated books still have a tiny chance of being picked
weighted_score = (df_books['Rating'] + 1) * np.log1p(df_books['Amazon_Rating_Count'] + 1)
df_books['Pick_Prob'] = weighted_score / weighted_score.sum()

# Define Personas
PERSONAS = {
    "Romance Reader": {"primary": ["Romance"], "secondary": ["Young Adult", "Fiction"]},
    "Fantasy Reader": {"primary": ["Fantasy"], "secondary": ["Sci-Fi", "Young Adult"]},
    "Mystery & Thriller Reader": {"primary": ["Mystery", "Thriller", "Crime"], "secondary": ["Fiction"]},
    "Science Fiction Reader": {"primary": ["Sci-Fi", "Science Fiction"], "secondary": ["Fantasy", "Science"]},
    "Self-Help Reader": {"primary": ["Self-Help", "Personal Development"], "secondary": ["Business", "Health"]},
    "Business & Finance Reader": {"primary": ["Business", "Finance", "Economics"], "secondary": ["Self-Help"]},
    "Computer Science Student": {"primary": ["Computer Science", "Programming", "Technology"], "secondary": ["Mathematics"]},
    "AI & Machine Learning Enthusiast": {"primary": ["Artificial Intelligence", "Machine Learning", "Data Science"], "secondary": ["Computer Science", "Mathematics"]},
    "Medical Student": {"primary": ["Medicine", "Health", "Medical"], "secondary": ["Biology", "Science"]},
    "Engineering Student": {"primary": ["Engineering"], "secondary": ["Physics", "Mathematics", "Computer Science"]},
    "NEET Preparation Student": {"primary": ["NEET", "Biology", "Physics", "Chemistry"], "secondary": ["Exam Preparation"]},
    "UPSC Preparation Student": {"primary": ["UPSC", "History", "Geography", "Polity"], "secondary": ["Exam Preparation", "Current Affairs"]},
    "Biography Reader": {"primary": ["Biography", "Autobiography", "Memoir"], "secondary": ["History", "Politics"]},
    "History Reader": {"primary": ["History"], "secondary": ["Biography", "Politics", "Society"]},
    "Philosophy Reader": {"primary": ["Philosophy"], "secondary": ["Psychology", "Religion"]},
    "Children's Book Reader": {"primary": ["Children", "Kids", "Picture Books"], "secondary": ["Fantasy"]},
    "Comic & Manga Reader": {"primary": ["Comics", "Graphic Novels", "Manga"], "secondary": ["Fantasy", "Humor"]},
    "Horror Reader": {"primary": ["Horror"], "secondary": ["Thriller", "Mystery"]},
    "Spiritual Reader": {"primary": ["Religion", "Spirituality", "New Age"], "secondary": ["Philosophy"]},
    "General Reader": {"primary": ["Fiction", "Literature"], "secondary": ["Non-Fiction"]}
}

def matches_genre(book_main, book_sub, genre_list):
    for g in genre_list:
        g = g.lower()
        if g in str(book_main).lower() or g in str(book_sub).lower():
            return True
    return False

# Group books by persona matching to speed up generation
print("Mapping books to personas...")
persona_books = {}
for p_name, p_data in PERSONAS.items():
    primary_mask = df_books.apply(lambda x: matches_genre(x['Main Genre'], x['Sub Genre'], p_data['primary']), axis=1)
    secondary_mask = df_books.apply(lambda x: matches_genre(x['Main Genre'], x['Sub Genre'], p_data['secondary']), axis=1)
    
    # Store indices of matching books
    primary_idx = df_books[primary_mask].index.tolist()
    secondary_idx = df_books[secondary_mask & ~primary_mask].index.tolist()
    unrelated_idx = df_books[~primary_mask & ~secondary_mask].index.tolist()
    
    persona_books[p_name] = {
        'primary': primary_idx,
        'secondary': secondary_idx,
        'unrelated': unrelated_idx
    }

def get_rating(tier):
    if tier == 'primary':
        return np.random.choice([5, 4, 3, 2], p=[0.50, 0.30, 0.15, 0.05])
    elif tier == 'secondary':
        return np.random.choice([5, 4, 3, 2, 1], p=[0.20, 0.35, 0.30, 0.10, 0.05])
    else:
        # Unrelated: mostly unrated, but if they do rate it:
        return np.random.choice([4, 3, 2, 1], p=[0.05, 0.35, 0.40, 0.20])

print(f"Generating synthetic ratings for {USER_COUNT} users...")
ratings = []
user_ids = range(10000, 10000 + USER_COUNT)
persona_names = list(PERSONAS.keys())

for u_id in tqdm(user_ids, desc="Generating Users"):
    # 1. Assign Persona
    persona = random.choice(persona_names)
    num_ratings = random.randint(20, 150)
    
    p_books = persona_books[persona]
    
    # 2. Decide how many books from each tier
    # Typically 60% primary, 30% secondary, 10% unrelated
    n_primary = int(num_ratings * 0.6)
    n_secondary = int(num_ratings * 0.3)
    n_unrelated = num_ratings - n_primary - n_secondary
    
    # Extract prob arrays for sampling
    def sample_books(indices, n):
        if not indices: return []
        n = min(n, len(indices))
        probs = df_books.loc[indices, 'Pick_Prob'].values
        sum_probs = probs.sum()
        if sum_probs > 0:
            probs = probs / sum_probs
        else:
            probs = np.ones(len(indices)) / len(indices)
        return np.random.choice(indices, size=n, replace=False, p=probs)
        
    sampled_primary = sample_books(p_books['primary'], n_primary)
    sampled_secondary = sample_books(p_books['secondary'], n_secondary)
    sampled_unrelated = sample_books(p_books['unrelated'], n_unrelated)
    
    for b_idx in sampled_primary:
        ratings.append({"User-ID": u_id, "ISBN": df_books.at[b_idx, 'ISBN'], "Book-Rating": get_rating('primary')})
    for b_idx in sampled_secondary:
        ratings.append({"User-ID": u_id, "ISBN": df_books.at[b_idx, 'ISBN'], "Book-Rating": get_rating('secondary')})
    for b_idx in sampled_unrelated:
        ratings.append({"User-ID": u_id, "ISBN": df_books.at[b_idx, 'ISBN'], "Book-Rating": get_rating('unrelated')})

df_ratings = pd.DataFrame(ratings)

print("\n\033[92m--- GENERATION COMPLETE ---\033[0m")
print(f"Total Users Simulated: {df_ratings['User-ID'].nunique():,}")
print(f"Total Books Interacted: {df_ratings['ISBN'].nunique():,}")
print(f"Total Ratings Generated: {len(df_ratings):,}")

user_counts = df_ratings.groupby('User-ID').size()
book_counts = df_ratings.groupby('ISBN').size()

print(f"\nAverage Ratings per User: {user_counts.mean():.1f}")
print(f"Average Ratings per Book: {book_counts.mean():.1f}")

print("\n\033[94mRating Distribution:\033[0m")
dist = df_ratings['Book-Rating'].value_counts(normalize=True).sort_index(ascending=False) * 100
for r, pct in dist.items():
    print(f"{r} Stars: {pct:5.1f}% | {'*' * int(pct/2)}")

df_ratings.to_csv(OUTPUT_FILE, index=False)
print(f"\n\033[92mSaved successfully to {OUTPUT_FILE}\033[0m")
