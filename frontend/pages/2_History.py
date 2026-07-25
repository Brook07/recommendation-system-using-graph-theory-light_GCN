import streamlit as st
import sys
from pathlib import Path

# Fix import paths
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.components.cards import render_book_card
from frontend.components.dialogs import show_book_details

st.set_page_config(
    page_title="Session History | GraphRec",
    page_icon="🕰️",
    layout="wide",
)

st.title("🕰️ Browsing History")
st.markdown("Recently viewed books and generated recommendations from this session.")

# Check state
if "history_books" not in st.session_state or not st.session_state.history_books:
    st.info("No history yet! Go back to the Home page and view some book details.")
else:
    if "viewing_book" in st.session_state:
        show_book_details(st.session_state.viewing_book)

    st.markdown("### Recently Viewed Books")
    
    # Reverse to show newest first
    books = list(reversed(st.session_state.history_books))
    
    cols = st.columns(4)
    for i, book in enumerate(books):
        render_book_card(book, cols[i % 4])

    if st.button("Clear History"):
        st.session_state.history_books = []
        st.rerun()

st.markdown("---")
st.markdown("### Recent Users")
if "history_users" in st.session_state and st.session_state.history_users:
    users = list(reversed(st.session_state.history_users))
    st.write(f"You recently viewed recommendations for users: {', '.join(map(str, users))}")
else:
    st.write("No user history yet.")
