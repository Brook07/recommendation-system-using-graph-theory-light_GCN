import requests
import streamlit as st
from typing import Dict, List, Optional
import time

API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=300)
def fetch_analytics() -> Optional[Dict]:
    """Fetch global analytics from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_user_profile(user_id: int) -> Optional[Dict]:
    """Fetch profile data for a specific user."""
    try:
        response = requests.get(f"{API_BASE_URL}/user/{user_id}/profile", timeout=5)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_recommendations(user_id: int, top_k: int = 10) -> Optional[Dict]:
    """Fetch explainable recommendations for a user."""
    try:
        response = requests.get(f"{API_BASE_URL}/recommend/{user_id}?top_k={top_k}", timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_book_metadata(isbn: str) -> Optional[Dict]:
    """Fetch detailed metadata for a single book."""
    try:
        response = requests.get(f"{API_BASE_URL}/books/{isbn}", timeout=5)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

import os
import urllib.parse

@st.cache_data(ttl=86400)
def get_google_books_cover(title: str, author: str) -> str:
    """Fetch cover from Google Books API based on title and author."""
    try:
        query = urllib.parse.quote(f"intitle:{title} inauthor:{author}")
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=1"
        api_key = os.environ.get("GOOGLE_BOOKS_API_KEY")
        if api_key:
            url += f"&key={api_key}"
            
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                volume_info = data["items"][0].get("volumeInfo", {})
                image_links = volume_info.get("imageLinks", {})
                # Use https explicitly
                thumb = image_links.get("thumbnail", "")
                if thumb.startswith("http:"):
                    thumb = thumb.replace("http:", "https:")
                return thumb
    except Exception:
        pass
    return ""
