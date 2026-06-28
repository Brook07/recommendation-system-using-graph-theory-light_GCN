"""GraphRec — Multi-User Demo Frontend.

Allows judges to instantly switch between three predefined user personas
and see completely different personalized book recommendations, similar to
switching accounts on an e-commerce platform like Daraz.
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8000"

DEMO_USERS = {
    "📖 The Avid Reader": {
        "id": 11676,
        "ratings": 2412,
        "description": "A voracious reader with 2,400+ ratings spanning Stephen King, "
                       "James Patterson, and Danielle Steel.",
        "avatar": "📖",
    },
    "💕 Romance Enthusiast": {
        "id": 153662,
        "ratings": 471,
        "description": "Deeply into romance novels — loves Nora Roberts, Jude Deveraux, "
                       "and Johanna Lindsey.",
        "avatar": "💕",
    },
    "🔍 Mystery & Thriller Fan": {
        "id": 95359,
        "ratings": 361,
        "description": "Drawn to suspense and mystery — follows James Patterson, "
                       "Stephen King, and Jonathan Kellerman.",
        "avatar": "🔍",
    },
}

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GraphRec — Book Recommendations",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar persona card ---- */
    .persona-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .persona-card .persona-name {
        font-size: 1.05em;
        font-weight: 600;
        color: #e0e0ff;
        margin-bottom: 4px;
    }
    .persona-card .persona-id {
        font-size: 0.82em;
        color: #8888cc;
        margin-bottom: 6px;
    }
    .persona-card .persona-desc {
        font-size: 0.8em;
        color: #aaaacc;
        line-height: 1.45;
    }

    /* ---- Book card ---- */
    .book-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 18px 14px;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .book-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(80,80,200,0.18);
    }
    .book-title {
        font-size: 0.95em;
        font-weight: 600;
        margin-top: 10px;
        color: #7eb8ff;
        line-height: 1.35;
        min-height: 2.7em;
    }
    .book-author {
        font-size: 0.82em;
        color: #aab4c8;
        margin-top: 4px;
    }
    .book-meta {
        font-size: 0.72em;
        color: #6b7899;
        margin-top: 3px;
    }
    .book-score {
        margin-top: 10px;
        font-size: 0.82em;
        font-weight: 600;
        color: #f0a500;
    }

    /* ---- Fallback alert ---- */
    .fallback-alert {
        padding: 14px 18px;
        background: linear-gradient(135deg, #3d2e00 0%, #4a3600 100%);
        border: 1px solid #f0a500;
        color: #ffd866;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 0.9em;
    }

    /* ---- Section header ---- */
    .section-header {
        font-size: 0.78em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #6b7899;
        margin-bottom: 6px;
        margin-top: 16px;
    }

    /* ---- Active user badge (main area) ---- */
    .active-user-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(126,184,255,0.2);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 18px;
    }
    .active-user-badge .badge-avatar {
        font-size: 1.6em;
    }
    .active-user-badge .badge-info {
        line-height: 1.4;
    }
    .active-user-badge .badge-name {
        font-weight: 600;
        color: #7eb8ff;
        font-size: 0.95em;
    }
    .active-user-badge .badge-detail {
        font-size: 0.78em;
        color: #8888cc;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API Helper (cached per user)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=600)
def fetch_recommendations(user_id: int, top_k: int = 10) -> dict | None:
    """Call the FastAPI backend and return the JSON payload. Cached per user."""
    try:
        resp = requests.get(
            f"{API_URL}/recommend/{user_id}",
            params={"top_k": top_k},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.exceptions.ConnectionError:
        return None


# ---------------------------------------------------------------------------
# Rendering Helpers
# ---------------------------------------------------------------------------

def render_persona_card(label: str, info: dict) -> None:
    """Render a compact persona summary card in the sidebar."""
    st.markdown(f"""
    <div class="persona-card">
        <div class="persona-name">{info['avatar']} {label.split(' ', 1)[1]}</div>
        <div class="persona-id">User ID: {info['id']}  ·  {info['ratings']} ratings</div>
        <div class="persona-desc">{info['description']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_book_card(rec: dict) -> None:
    """Render a single book recommendation card."""
    img_url = rec.get("Image_URL_L", "")
    if img_url and str(img_url).startswith("http"):
        st.image(img_url, use_container_width=True)
    else:
        st.image(
            "https://via.placeholder.com/150x200?text=No+Cover",
            use_container_width=True,
        )

    title = rec.get("Book_Title", "Unknown Title")
    author = rec.get("Book_Author", "Unknown Author")
    publisher = rec.get("Publisher", "")
    year = rec.get("Year_Of_Publication", "")
    score = rec.get("Score", 0.0)

    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">{title}</div>
        <div class="book-author">by {author}</div>
        <div class="book-meta">{publisher} · {year}</div>
        <div class="book-score">⭐ Score: {score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendations(recs: list) -> None:
    """Render the full grid of recommendation cards (5 columns × 2 rows)."""
    if not recs:
        st.warning("No recommendations returned.")
        return

    # Fallback alert
    if recs[0].get("Is_Fallback"):
        st.markdown("""
        <div class="fallback-alert">
            <strong>🆕 New or Unknown User!</strong> We don't have enough history
            for this user yet, so we're showing our all-time most popular and
            highly-rated books.
        </div>
        """, unsafe_allow_html=True)

    # Grid — 5 columns
    cols = st.columns(5)
    for idx, rec in enumerate(recs):
        with cols[idx % 5]:
            render_book_card(rec)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 GraphRec")
    st.caption("Powered by LightGCN · Graph Neural Networks")

    st.markdown('<div class="section-header">👤 Switch User</div>', unsafe_allow_html=True)

    selected_label = st.radio(
        "Choose a demo persona",
        options=list(DEMO_USERS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    # Show the selected persona's info card
    render_persona_card(selected_label, DEMO_USERS[selected_label])

    st.markdown("---")

    # Manual override (collapsed by default)
    with st.expander("⚙️ Advanced — Custom User ID"):
        custom_id = st.number_input(
            "Enter any User ID",
            min_value=1,
            value=999999,
            step=1,
            help="Use 999999 to test the cold-start fallback.",
        )
        use_custom = st.button("Fetch for Custom ID", type="secondary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="section-header">How it works</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Each persona represents a **real user** from the Book-Crossing dataset. "
        "When you switch personas, the app queries the LightGCN model in real time — "
        "the recommendations you see are **fully personalized** based on that user's "
        "unique graph embedding.",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------

st.title("📚 GraphRec: Book Recommendation Engine")

# Determine which user to show
if use_custom if "use_custom" in dir() else False:
    active_user_id = custom_id
    active_label = "Custom User"
    active_avatar = "⚙️"
    active_detail = f"User ID: {custom_id}"
else:
    info = DEMO_USERS[selected_label]
    active_user_id = info["id"]
    active_label = selected_label.split(" ", 1)[1]
    active_avatar = info["avatar"]
    active_detail = f"User ID: {info['id']}  ·  {info['ratings']} ratings"

# Active user badge
st.markdown(f"""
<div class="active-user-badge">
    <span class="badge-avatar">{active_avatar}</span>
    <div class="badge-info">
        <div class="badge-name">{active_label}</div>
        <div class="badge-detail">{active_detail}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch and display recommendations
with st.spinner(f"Fetching personalized recommendations for User {active_user_id}…"):
    data = fetch_recommendations(active_user_id)

if data is None:
    st.error(
        "❌ Failed to connect to the backend. "
        "Is the FastAPI server running on `localhost:8000`?"
    )
else:
    recs = data.get("recommendations", [])
    if recs and not recs[0].get("Is_Fallback"):
        st.success(f"Top 10 Personalized Recommendations for User {active_user_id}")
    render_recommendations(recs)
