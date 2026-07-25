import streamlit as st
import urllib.parse
from typing import Dict, Any
from frontend.utils.api import get_google_books_cover

@st.dialog("Book Details")
def show_book_details(book: Dict[str, Any]):
    """Renders a detailed modal for a specific book."""
    
    col1, col2 = st.columns([1, 2])
    
    title = str(book.get("Book_Title", "Unknown Title"))
    author = str(book.get("Author", "Unknown Author"))
    cover_url = str(book.get("Book_Cover_URL", "")).strip()
    
    if not cover_url or cover_url.lower() == 'nan':
        cover_url = get_google_books_cover(title, author)
        if not cover_url:
            safe_title = urllib.parse.quote(title[:20])
            cover_url = f"https://placehold.co/300x400/101827/b3c5ff?text={safe_title}"
        
    with col1:
        st.image(cover_url, use_container_width=True)
        amz_url = book.get("Amazon_URL", "#")
        price = book.get("Price", "N/A")
        st.markdown(f'<a href="{amz_url}" target="_blank" style="display: block; text-align: center; background: #ff9900; color: #111; padding: 10px; border-radius: 4px; text-decoration: none; font-weight: bold; margin-top: 15px;">🛒 Buy on Amazon ({price})</a>', unsafe_allow_html=True)

    with col2:
        st.header(book.get("Book_Title", "Unknown Title"))
        st.subheader(f"by {book.get('Author', 'Unknown Author')}")
        
        st.markdown("---")
        
        st.markdown(f"**Main Genre:** {book.get('Main_Genre', 'N/A')}")
        st.markdown(f"**Sub Genre:** {book.get('Sub_Genre', 'N/A')}")
        st.markdown(f"**Format:** {book.get('Book_Type', 'N/A')}")
        
        amz_rating = book.get("Amazon_Rating", 0.0)
        st.markdown(f"**Amazon Rating:** {'★' * int(amz_rating)}{'☆' * (5-int(amz_rating))} {amz_rating} ({book.get('Number_of_People_Rated', 0)} ratings)")
        st.markdown(f"**Community Rating:** {book.get('Dataset_Average_Rating', 0.0)} / 10.0")
        
        st.markdown("---")
        st.markdown("### Why this book?")
        st.info(f"**Confidence Score:** {book.get('Confidence_Score', '0%')}")
        
        for tag in book.get("Explanation_Tags", []):
            st.success(f"✓ {tag}")
            
    if st.button("Close"):
        del st.session_state.viewing_book
        st.rerun()
