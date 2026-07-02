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

    /* ---- Global Styles ---- */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        background: #0A1128;
        color: #E8F1FF;
    }

    /* ---- Page Background ---- */
    [data-testid="stAppViewContainer"] {
        background: #0A1128;
    }

    /* ---- Sidebar Styling ---- */
    [data-testid="stSidebar"] {
        background: #1B2A4A;
        border-right: 1px solid rgba(79, 195, 247, 0.2);
    }

    /* ---- Sidebar persona card ---- */
    .persona-card {
        background: #2E5EAA;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 2px solid rgba(79, 195, 247, 0.3);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .persona-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #4FC3F7 0%, #2196F3 100%);
        opacity: 1;
    }
    
    .persona-card:hover {
        background: #3A6FBF;
        border-color: rgba(79, 195, 247, 0.5);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.2);
    }
    }
    .persona-card .persona-name {
        font-size: 1.05em;
        font-weight: 600;
        color: #4FC3F7;
        margin-bottom: 4px;
    }
    .persona-card .persona-id {
        font-size: 0.82em;
        color: #90CAF9;
        margin-bottom: 6px;
    }
    .persona-card .persona-desc {
        font-size: 0.8em;
        color: #E8F1FF;
        line-height: 1.45;
    }

    /* ---- Book Card (Combined Image + Details) ---- */
    .book-card-container {
        background: #1B3A70;
        border-radius: 12px;
        margin: 8px 0;
        border: 1px solid rgba(79, 195, 247, 0.2);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: all 0.2s ease;
        overflow: hidden;
        cursor: pointer;
    }
    
    .book-card-container:hover {
        background: #2E5EAA;
        border-color: rgba(79, 195, 247, 0.4);
        box-shadow: 0 6px 16px rgba(79, 195, 247, 0.2);
        transform: translateY(-4px);
    }
    
    .book-card-container:hover img {
        filter: brightness(1.1) contrast(1.05);
        transform: scale(1.02);
    }
    
    .book-card-container img {
        width: 100%;
        display: block;
        transition: all 0.2s ease;
    }
    
    .book-card-content {
        padding: 20px 16px;
    }
    
    .book-card-container .book-title {
        font-size: 1.02em;
        font-weight: 700;
        color: #4FC3F7;
        line-height: 1.4;
        min-height: 3em;
        margin: 0;
    }
    
    .book-card-container .book-author {
        font-size: 0.88em;
        color: #BBDEFB;
        margin-top: 8px;
        margin-bottom: 0;
        font-weight: 500;
    }
    
    .book-card-container .book-meta {
        font-size: 0.78em;
        color: #90CAF9;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .book-card-container .book-score {
        margin-top: 14px;
        font-size: 0.88em;
        font-weight: 700;
        background: #4FC3F7;
        color: #0A1128;
        padding: 8px 16px;
        border-radius: 18px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(79, 195, 247, 0.2);
        margin-bottom: 0;
    }

    /* ---- Fallback Alert ---- */
    .fallback-alert {
        padding: 14px 18px;
        background: #2E5EAA;
        border: 1px solid rgba(79, 195, 247, 0.3);
        color: #E8F1FF;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 0.9em;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* ---- Section Header ---- */
    .section-header {
        font-size: 0.75em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #4FC3F7;
        margin-bottom: 8px;
        margin-top: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-header::before {
        content: '●';
        font-size: 1.2em;
        color: #4FC3F7;
    }

    /* ---- Active User Badge ---- */
    .active-user-badge {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        background: #2E5EAA;
        border: 1.5px solid rgba(79, 195, 247, 0.4);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .active-user-badge:hover {
        background: #3A6FBF;
        border-color: rgba(79, 195, 247, 0.6);
    }
    .active-user-badge .badge-avatar {
        font-size: 1.8em;
    }
    .active-user-badge .badge-info {
        line-height: 1.4;
    }
    .active-user-badge .badge-name {
        font-weight: 700;
        color: #4FC3F7;
        font-size: 0.98em;
    }
    .active-user-badge .badge-detail {
        font-size: 0.78em;
        color: #90CAF9;
    }

    /* ---- Streamlit Button ---- */
    button {
        transition: all 0.2s ease;
    }
    button:hover {
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.2) !important;
    }

    /* ---- Success/Error Messages ---- */
    .stSuccess, [data-testid="stAlert"][type="success"] {
        background: #2E5EAA !important;
        border: 1px solid rgba(79, 195, 247, 0.3) !important;
        color: #E8F1FF !important;
        border-radius: 10px !important;
    }

    .stError, [data-testid="stAlert"][type="error"] {
        background: #5D3A3A !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        color: #FF8A80 !important;
        border-radius: 10px !important;
    }

    .stWarning, [data-testid="stAlert"][type="warning"] {
        background: #5D4D3A !important;
        border: 1px solid rgba(255, 152, 0, 0.3) !important;
        color: #FFB74D !important;
        border-radius: 10px !important;
    }

    /* ---- Skeleton Loader ---- */
    .skeleton-loader {
        background: linear-gradient(90deg, #2E5EAA 0%, #4FC3F7 50%, #2E5EAA 100%);
        background-size: 200% 100%;
        animation: load 2s infinite;
        border-radius: 10px;
        height: 200px;
    }

    @keyframes load {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ---- Headings ---- */
    h1, h2, h3, h4, h5, h6 {
        color: #E8F1FF;
    }

    /* ---- Radio Button ---- */
    [data-testid="stRadio"] label {
        color: #E8F1FF;
        font-weight: 500;
    }
    [data-testid="stRadio"] {
        gap: 8px;
    }

    /* ---- Radio Button Enhanced Styling ---- */
    [data-testid="stRadio"] > label {
        display: flex !important;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 10px;
        background: transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    [data-testid="stRadio"] > label:hover {
        background: rgba(79, 195, 247, 0.15);
        border-radius: 10px;
    }

    [data-testid="stRadio"] input[type="radio"] {
        width: 18px !important;
        height: 18px !important;
        cursor: pointer;
        accent-color: #4FC3F7 !important;
    }

    [data-testid="stRadio"] input[type="radio"]:checked {
        accent-color: #2196F3 !important;
    }

    [data-testid="stRadio"] > label span:first-child {
        min-width: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ---- Expander ---- */
    [data-testid="stExpander"] {
        background: #2E5EAA;
        border: 1px solid rgba(79, 195, 247, 0.2);
        border-radius: 10px;
    }
    [data-testid="stExpander"] summary {
        color: #4FC3F7;
        font-weight: 600;
    }

    /* ---- Number Input ---- */
    [data-testid="stNumberInput"] input {
        background: #1B3A70 !important;
        color: #E8F1FF !important;
        border: 1px solid rgba(79, 195, 247, 0.2) !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: #4FC3F7 !important;
        box-shadow: 0 0 0 1px rgba(79, 195, 247, 0.3) !important;
    }

    /* ---- Mobile Responsive ---- */
    @media (max-width: 768px) {
        .active-user-badge {
            flex-direction: column;
            align-items: flex-start;
        }
        .book-card {
            padding: 12px 10px;
        }
    }

    /* ---- Hide unwanted elements ---- */
    [class*="stKeyboard"] {
        display: none !important;
    }
    
    /* ---- Button styling improvements ---- */
    [data-testid="baseButton-primary"] {
        background: #4FC3F7 !important;
        color: #0A1128 !important;
        font-weight: 600;
    }
    [data-testid="baseButton-primary"]:hover {
        background: #2196F3 !important;
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
    """Render a single book recommendation card with image and details combined."""
    img_url = rec.get("Image_URL_L", "")
    if img_url and str(img_url).startswith("http"):
        image_html = f'<img src="{img_url}" style="width: 100%; height: 320px; object-fit: cover; border-radius: 12px 12px 0 0;">'
    else:
        image_html = '<img src="https://via.placeholder.com/150x200?text=No+Cover" style="width: 100%; height: 320px; object-fit: cover; border-radius: 12px 12px 0 0;">'

    title = rec.get("Book_Title", "Unknown Title")
    author = rec.get("Book_Author", "Unknown Author")
    publisher = rec.get("Publisher", "")
    year = rec.get("Year_Of_Publication", "")
    score = rec.get("Score", 0.0)

    st.markdown(f"""
    <div class="book-card-container">
        {image_html}
        <div class="book-card-content">
            <div class="book-title">{title}</div>
            <div class="book-author">by {author}</div>
            <div class="book-meta">{publisher} · {year}</div>
            <div class="book-score">⭐ Score: {score:.4f}</div>
        </div>
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
        "",
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
        col1, col2 = st.columns(2)
        with col1:
            use_custom = st.button("🔍 Fetch Custom", use_container_width=True, type="primary")
        with col2:
            st.write("")

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
with st.spinner("Loading recommendations..."):
    data = fetch_recommendations(active_user_id)

if data is None:
    st.error(
        "❌ Failed to connect to the backend. "
        "Is the FastAPI server running on `localhost:8000`?"
    )
else:
    recs = data.get("recommendations", [])
    if recs and not recs[0].get("Is_Fallback"):
        st.success(f"✨ Top 10 Personalized Recommendations for User {active_user_id}")
    render_recommendations(recs)
