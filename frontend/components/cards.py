import streamlit as st
import urllib.parse
from typing import Dict, Any
from frontend.utils.state import track_book_view
from frontend.utils.api import get_google_books_cover

def render_book_card(book: Dict[str, Any], col_obj):
    """Renders a modern book card within a Streamlit column."""
    with col_obj:
        title = str(book.get("Book_Title", "Unknown"))
        author = str(book.get("Author", "Unknown"))
        
        cover_url = str(book.get("Book_Cover_URL", "")).strip()
        
        # Prefer the pre-fetched Google_Thumbnail if available
        if not cover_url or cover_url.lower() == 'nan':
            cover_url = str(book.get("Google_Thumbnail", "")).strip()
            
        # If still missing, fetch dynamically from Google Books API
        if not cover_url or cover_url.lower() == 'nan':
            cover_url = get_google_books_cover(title, author)
            if not cover_url:
                safe_title = urllib.parse.quote(title[:20])
                cover_url = f"https://placehold.co/150x220/101827/b3c5ff?text={safe_title}"
        score = book.get("Recommendation_Score", 0)
        confidence = str(book.get("Confidence_Score", "0%"))
        tags = book.get("Explanation_Tags", [])
        amazon_rating = book.get("Amazon_Rating", 0.0)
        amz_url = book.get("Amazon_URL", "#")
        price = book.get("Price", "N/A")
        
        # We use a native Streamlit container to hold both the HTML and the interactive buttons
        with st.container(border=True):
            # Style the card using raw HTML but without external borders/shadows so it blends in
            html_card = f"""
            <div style="
                display: flex;
                flex-direction: column;
                height: 380px; /* Fixed height for the upper content */
                position: relative;
            ">
                <div style="position: absolute; top: 10px; right: 10px; background: rgba(0,241,254,0.15); color: #00f1fe; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 0.8rem; border: 1px solid rgba(0,241,254,0.3); z-index: 10;">
                    {confidence} Match
                </div>
                <div style="height: 200px; min-height: 200px; overflow: hidden; display: flex; align-items: center; justify-content: center; border-radius: 8px;">
                    <img src="{cover_url}" alt="Cover" style="height: 100%; object-fit: cover; width: 100%;" onerror="this.src='https://placehold.co/150x200?text=No+Cover'">
                </div>
                <div style="padding-top: 15px; display: flex; flex-direction: column; flex-grow: 1;">
                    <h4 style="margin: 0 0 5px 0; font-size: 1.1rem; color: #eef3ff; line-height: 1.2; height: 42px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">{title}</h4>
                    <p style="margin: 0 0 10px 0; color: #9aa6c4; font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; height: 18px;">{author}</p>
                    <div style="margin-bottom: 10px; font-size: 0.85rem; color: #ffd700; height: 18px;">
                        {'★' * int(amazon_rating)}{'☆' * (5-int(amazon_rating))} {amazon_rating if amazon_rating else ''}
                    </div>
                    <div style="overflow: hidden; flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-start; gap: 4px;">
                        {"".join([f'<div style="background: rgba(179,197,255,0.1); color: #b3c5ff; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">✓ {tag}</div>' for tag in tags[:3]])}
                    </div>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)
        
            # Native Streamlit buttons below the HTML layout for interactivity
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔍 Details", key=f"details_{book.get('ISBN')}", use_container_width=True):
                    st.session_state.viewing_book = book
                    track_book_view(book)
                    st.rerun()
            with btn_col2:
                st.markdown(f'<a href="{amz_url}" target="_blank" style="display: block; text-align: center; background: #ff9900; color: #111; padding: 6px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 0.9rem;">🛒 Buy {price}</a>', unsafe_allow_html=True)
