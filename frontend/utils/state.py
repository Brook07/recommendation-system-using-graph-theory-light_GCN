import streamlit as st
from typing import Dict, List, Any

def init_session_state():
    """Initialize all session state variables."""
    if "history_users" not in st.session_state:
        st.session_state.history_users = []
    
    if "history_books" not in st.session_state:
        st.session_state.history_books = []
        
    if "selected_user" not in st.session_state:
        st.session_state.selected_user = 11676  # Default demo user
        
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Home"

def track_user_view(user_id: int):
    """Add user to history."""
    if not st.session_state.history_users or st.session_state.history_users[-1] != user_id:
        st.session_state.history_users.append(user_id)
        # Keep last 20
        if len(st.session_state.history_users) > 20:
            st.session_state.history_users.pop(0)

def track_book_view(book_metadata: Dict[str, Any]):
    """Add book to viewed history."""
    isbn = book_metadata.get('ISBN')
    # Remove if exists to push to front
    st.session_state.history_books = [b for b in st.session_state.history_books if b.get('ISBN') != isbn]
    st.session_state.history_books.append(book_metadata)
    # Keep last 30
    if len(st.session_state.history_books) > 30:
        st.session_state.history_books.pop(0)
