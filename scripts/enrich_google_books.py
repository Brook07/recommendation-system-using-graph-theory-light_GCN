import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import urllib.parse

# Load environment variables from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

DATA_DIR = os.path.join(project_root, "data", "processed-dataset")
INPUT_FILE = os.path.join(DATA_DIR, "Enhanced_Books.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "Enhanced_Books_Final.csv")
CACHE_FILE = os.path.join(DATA_DIR, "api_cache.json")

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Cache file is corrupted. Starting fresh.")
    return {}

def save_cache(cache_data):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4)

def fetch_book_metadata(title, author, api_key, max_retries=3):
    """Fetch best matching book metadata from Google Books API."""
    query_parts = []
    if pd.notna(title) and str(title).strip():
        query_parts.append(f"intitle:{str(title).strip()}")
    if pd.notna(author) and str(author).strip():
        query_parts.append(f"inauthor:{str(author).strip()}")
        
    if not query_parts:
        return None
        
    query = " ".join(query_parts)
    url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(query)}"
    if api_key:
        url += f"&key={api_key}"
        
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("totalItems", 0) > 0 and "items" in data:
                    return extract_info(data["items"][0])
                return {"Real_ISBN_10": None, "Real_ISBN_13": None, "Google_Thumbnail": None}
            elif response.status_code == 429:
                time.sleep((2 ** attempt) + 1)
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {"Real_ISBN_10": None, "Real_ISBN_13": None, "Google_Thumbnail": None}
            time.sleep((2 ** attempt) + 1)
            
    return {"Real_ISBN_10": None, "Real_ISBN_13": None, "Google_Thumbnail": None}

def extract_info(item):
    """Extract Real_ISBN_10, Real_ISBN_13, and Google_Thumbnail."""
    volume_info = item.get("volumeInfo", {})
    
    isbn_10 = None
    isbn_13 = None
    
    for identifier in volume_info.get("industryIdentifiers", []):
        if identifier.get("type") == "ISBN_10":
            isbn_10 = identifier.get("identifier")
        elif identifier.get("type") == "ISBN_13":
            isbn_13 = identifier.get("identifier")
            
    image_links = volume_info.get("imageLinks", {})
    thumbnail = image_links.get("extraLarge") or image_links.get("large") or \
                image_links.get("medium") or image_links.get("small") or \
                image_links.get("thumbnail") or image_links.get("smallThumbnail")
                
    return {
        "Real_ISBN_10": isbn_10,
        "Real_ISBN_13": isbn_13,
        "Google_Thumbnail": thumbnail
    }

def process_dataset(max_books=10000):
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    try:
        df = pd.read_csv(INPUT_FILE, dtype=str)
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        return
        
    df = df.head(max_books)
    
    id_col = next((col for col in df.columns if col.upper() in ["ISBN", "ID"]), "Book-Title")
    title_col = next((col for col in df.columns if col.upper() in ["BOOK_TITLE", "BOOK-TITLE", "TITLE"]), None)
    author_col = next((col for col in df.columns if col.upper() in ["BOOK_AUTHOR", "BOOK-AUTHOR", "AUTHOR"]), None)
    
    if not title_col:
        print("Error: Could not identify title column in the dataset.")
        return

    for col in ["Real_ISBN_10", "Real_ISBN_13", "Google_Thumbnail"]:
        if col not in df.columns:
            df[col] = None
            
    cache = load_cache()
    
    print(f"Starting enrichment of {len(df)} books. Press Ctrl+C to interrupt and save progress safely.")
    
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enriching Books"):
            book_id = str(row.get(id_col, idx))
            title = row.get(title_col)
            author = row.get(author_col)
            
            cache_key = f"{book_id}_{title}"
            
            if cache_key in cache:
                cached_data = cache[cache_key]
                df.at[idx, "Real_ISBN_10"] = cached_data.get("Real_ISBN_10")
                df.at[idx, "Real_ISBN_13"] = cached_data.get("Real_ISBN_13")
                df.at[idx, "Google_Thumbnail"] = cached_data.get("Google_Thumbnail")
                continue
                
            tqdm.write(f"Processing: {title} by {author}")
            
            result = fetch_book_metadata(title, author, API_KEY)
            if result:
                df.at[idx, "Real_ISBN_10"] = result.get("Real_ISBN_10")
                df.at[idx, "Real_ISBN_13"] = result.get("Real_ISBN_13")
                df.at[idx, "Google_Thumbnail"] = result.get("Google_Thumbnail")
                
                cache[cache_key] = result
                
                if idx > 0 and idx % 20 == 0:
                    save_cache(cache)
                    
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving progress...")
    finally:
        save_cache(cache)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    if not API_KEY:
        print("Warning: GOOGLE_BOOKS_API_KEY is not set in .env. API limits will be heavily restricted.")
    
    process_dataset(10000)
