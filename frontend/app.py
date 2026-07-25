"""
GraphRec Streamlit Frontend — Pure Python UI
All layout uses native Streamlit components (st.columns, st.container, etc.)
CSS is injected only as a <style> block to theme the native components.
"""
import streamlit as st
import sys
import pandas as pd
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.utils.api import fetch_recommendations, fetch_analytics
from frontend.utils.state import init_session_state, track_user_view
from frontend.components.dialogs import show_book_details

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRec | Intelligent Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# CSS Theming (styles native Streamlit elements)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0e1a !important;
    color: #dfe2f3 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"]        { display: none !important; }
[data-testid="stSidebar"]       { display: none !important; }
[data-testid="collapsedControl"]{ display: none !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 4rem !important; max-width: 1380px !important; }

/* ── Titles & text ── */
h1 { color: #b3c5ff !important; font-weight: 800 !important; font-size: 2rem !important; }
h2 { color: #b3c5ff !important; font-weight: 700 !important; }
h3 { color: #dfe2f3 !important; font-weight: 600 !important; }
p, label { color: #c2c6d8 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > label { color: #b3c5ff !important; font-weight: 600 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important; color: #dfe2f3 !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] > label { color: #b3c5ff !important; font-weight: 600 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important; color: #dfe2f3 !important;
}
[data-testid="stTextInput"] input:focus { border-color: #b3c5ff !important; box-shadow: 0 0 0 1px #b3c5ff33 !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: rgba(0, 102, 255, 0.12) !important;
    border: 1px solid rgba(0, 102, 255, 0.4) !important;
    color: #b3c5ff !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: rgba(0, 102, 255, 0.25) !important;
    border-color: #b3c5ff !important; transform: translateY(-1px);
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #b3c5ff !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8c90a1 !important; font-size: 0.75rem !important; }

/* ── Containers / Cards ── */
[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: #8c90a1 !important; font-size: 0.78rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.20); }

/* ── Book card custom classes ── */
.book-card-wrap {
    background: rgba(15,19,31,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.28s cubic-bezier(.23,1,.32,1);
    height: 100%;
}
.book-card-wrap:hover {
    border-color: rgba(179,197,255,0.35);
    transform: translateY(-3px);
    box-shadow: 0 16px 36px rgba(0,0,0,0.5);
}
.match-pill {
    display: inline-block;
    background: rgba(0,241,254,0.12);
    border: 1px solid rgba(0,241,254,0.3);
    color: #00f1fe;
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.10em;
    padding: 3px 10px; border-radius: 99px;
    box-shadow: 0 0 10px rgba(0,241,254,0.14);
    margin-bottom: 6px;
}
.explain-tag {
    font-size: 0.72rem; color: #b3c5ff;
    font-style: italic; font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 4px;
}
.genre-chip {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    color: #8c90a1;
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 2px 8px; border-radius: 5px; margin: 2px 2px 0 0;
}
.persona-chip {
    display: inline-block; cursor: pointer;
    padding: 8px 16px; border-radius: 99px;
    font-size: 0.80rem; font-weight: 600;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
    color: #c2c6d8; margin: 3px;
    transition: all 0.2s;
}
.persona-chip.active {
    background: rgba(0,102,255,0.18);
    border-color: #b3c5ff; color: #b3c5ff;
    box-shadow: 0 0 14px rgba(0,102,255,0.2);
}
.section-head {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: rgba(223,226,243,0.45);
    margin-bottom: 10px;
}
.top-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0 20px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 24px;
}
.logo-text {
    font-size: 1.6rem; font-weight: 800;
    color: #b3c5ff; letter-spacing: -0.5px;
}
.rec-info {
    font-size: 0.75rem; color: rgba(194,198,216,0.5);
    font-style: italic; margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# State & Data Loading
# ─────────────────────────────────────────────────────────────────
init_session_state()

DATA_DIR      = REPO_ROOT / "data" / "processed-dataset"
GRAPH_DIR     = REPO_ROOT / "data" / "processed-graph"
PERSONAS_FILE = DATA_DIR / "personas.csv"
RATINGS_FILE  = DATA_DIR / "Ratings_Final.csv"
USER_MAP_FILE = GRAPH_DIR / "user_mapping.csv"

PERSONA_EMOJI = {
    "Arts & Photography Enthusiast":    "🎨",
    "Biography & History Reader":       "📜",
    "Business & Economics Reader":      "📈",
    "Children's Book Reader":           "🧒",
    "Crafts & Lifestyle Reader":        "🏡",
    "Crime & Mystery Reader":           "🔍",
    "Engineering Student":              "⚙️",
    "Exam Preparation Student":         "📝",
    "Fantasy / Horror / Sci-Fi Reader": "🚀",
    "Health & Self-Help Reader":        "💪",
    "Higher Education Student":         "🎓",
    "History Reader":                   "🏛️",
    "Language & Writing Enthusiast":    "✍️",
    "Law & Governance Reader":          "⚖️",
    "Literature & Fiction Reader":      "📖",
}

@st.cache_data(ttl=600)
def load_demo_personas():
    if not PERSONAS_FILE.exists():
        return {}, "personas.csv not found. Run scripts/03_generate_final_datasets.py"
    df_p = pd.read_csv(PERSONAS_FILE, dtype={"User-ID": int})
    valid_rated  = set(pd.read_csv(RATINGS_FILE,  usecols=["User-ID"])["User-ID"].astype(int)) if RATINGS_FILE.exists()  else set()
    valid_mapped = set(pd.read_csv(USER_MAP_FILE, usecols=["User-ID"])["User-ID"].astype(int)) if USER_MAP_FILE.exists() else set()
    demo, errors = {}, []
    for persona, grp in df_p.groupby("Persona"):
        row = grp.iloc[len(grp) // 2]
        uid = int(row["User-ID"])
        bad = []
        if uid not in valid_rated:  bad.append("Ratings_Final.csv")
        if valid_mapped and uid not in valid_mapped: bad.append("user_mapping.csv")
        if bad:
            errors.append(f"User {uid} ({persona}) missing from {', '.join(bad)}")
            continue
        demo[persona] = {"user_id": uid, "persona": persona,
                         "primary_genre": row["Primary_Genre"],
                         "secondary_genre": row["Secondary_Genre"]}
    return demo, "\n".join(errors) if errors else None

demo_users, _persona_error = load_demo_personas()
persona_labels = list(demo_users.keys())

if "active_persona" not in st.session_state:
    st.session_state.active_persona = persona_labels[0] if persona_labels else None

# ─────────────────────────────────────────────────────────────────
# HEADER ROW
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <div class="logo-text">📚 GraphRec</div>
  <div style="font-size:0.8rem;color:#8c90a1;">Graph Neural Network · Book Discovery</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PERSONA SELECTOR (two columns: chips + controls)
# ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 1])

with left_col:
    st.markdown('<div class="section-head">Active Persona</div>', unsafe_allow_html=True)

    active_persona = st.session_state.active_persona
    active_info    = demo_users.get(active_persona, {})

    if persona_labels:
        chosen = st.selectbox(
            "Switch Persona",
            options=persona_labels,
            index=persona_labels.index(active_persona) if active_persona in persona_labels else 0,
            format_func=lambda p: f"{PERSONA_EMOJI.get(p,'📚')}  {p}",
            key="persona_picker",
            label_visibility="collapsed",
        )
        if chosen != st.session_state.active_persona:
            st.session_state.active_persona = chosen
            st.session_state.selected_user  = demo_users[chosen]["user_id"]
            track_user_view(st.session_state.selected_user)
            st.rerun()

    # Persona info card
    if active_info:
        emoji = PERSONA_EMOJI.get(active_persona, "📚")
        st.markdown(f"""
        <div style="
          background: rgba(0,102,255,0.10);
          border: 1px solid rgba(179,197,255,0.25);
          border-radius: 14px; padding: 14px 18px; margin-top: 10px;
          display: flex; gap: 16px; align-items: center;
        ">
          <div style="font-size:2.2rem;line-height:1;">{emoji}</div>
          <div>
            <div style="font-size:1rem;font-weight:700;color:#b3c5ff;margin-bottom:4px;">{active_persona}</div>
            <div style="font-size:0.8rem;color:#8c90a1;">
              <b style="color:#c2c6d8;">Primary:</b> {active_info.get('primary_genre','—')}<br>
              <b style="color:#c2c6d8;">Secondary:</b> {active_info.get('secondary_genre','—')}<br>
              <b style="color:#c2c6d8;">User ID:</b> {active_info.get('user_id','—')}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-head">Custom User ID</div>', unsafe_allow_html=True)
    custom_id = st.text_input("Custom ID", value="", placeholder="e.g. 10042",
                               label_visibility="collapsed", key="custom_uid")
    if custom_id and custom_id.isdigit():
        cid = int(custom_id)
        if cid != st.session_state.selected_user:
            st.session_state.selected_user = cid
            track_user_view(cid)
            st.rerun()
        st.caption(f"Showing results for User **{cid}**")

    # System stats
    st.markdown('<div class="section-head" style="margin-top:18px;">System Stats</div>', unsafe_allow_html=True)
    stats = fetch_analytics() or {}
    if stats:
        s1, s2 = st.columns(2)
        s1.metric("Books",   f"{stats.get('Total_Books',0):,}")
        s2.metric("Users",   f"{stats.get('Total_Users',0):,}")
        s1.metric("Ratings", f"{stats.get('Total_Ratings',0):,}")
        s2.metric("Genres",  str(stats.get("Unique_Genres", "—")))

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
recs_response  = fetch_recommendations(st.session_state.selected_user, top_k=12)
recs           = recs_response.get("recommendations", []) if recs_response else []
inference_time = recs_response.get("inference_time_ms", 0) if recs_response else 0

# Section header
head_left, head_right = st.columns([4, 1])
with head_left:
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px;">
      <span style="font-size:1.25rem;font-weight:700;color:#dfe2f3;">Recommended for You</span>
      <span class="rec-info">{"✨ " + str(len(recs)) + " books · " + str(inference_time) + "ms" if recs else ""}</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    search_q = st.text_input("🔍 Filter", value="", placeholder="Search title…",
                              label_visibility="collapsed", key="search_filter")

if not recs:
    st.info("No recommendations yet. Make sure the FastAPI backend is running (`uvicorn backend.main:app --reload`) and the model is trained.")
else:
    # Filter
    if search_q:
        recs = [r for r in recs if search_q.lower() in r.get("Book_Title","").lower()
                                 or search_q.lower() in r.get("Author","").lower()]

    # ── Book grid: 4 columns ──────────────────────────────────────
    COLS = 4
    col_objs = st.columns(COLS, gap="medium")

    for i, book in enumerate(recs):
        title      = str(book.get("Book_Title", "Unknown"))
        author     = str(book.get("Author", "Unknown"))
        genre      = str(book.get("Main_Genre", ""))
        sub_genre  = str(book.get("Sub_Genre", ""))
        confidence = str(book.get("Confidence_Score", "N/A"))
        tags       = book.get("Explanation_Tags", [])
        amz_rating = float(book.get("Amazon_Rating") or 0)
        amz_url    = book.get("Amazon_URL") or "#"
        price      = book.get("Price", "N/A")
        isbn       = str(book.get("ISBN", ""))

        cover_url = str(book.get("Book_Cover_URL", "")).strip()
        if not cover_url or cover_url.lower() == "nan":
            safe = urllib.parse.quote(title[:22])
            cover_url = f"https://placehold.co/220x320/101827/b3c5ff?text={safe}"

        explain = tags[0] if tags else "Recommended based on your reading behavior"
        stars   = "★" * int(amz_rating) + "☆" * (5 - int(amz_rating))

        genre_chips = ""
        for g in [genre, sub_genre]:
            if g and g not in ("nan", "Unknown", ""):
                genre_chips += f'<span class="genre-chip">{g}</span>'

        with col_objs[i % COLS]:
            # Cover image
            st.image(cover_url, use_container_width=True)

            # Card body
            st.markdown(f"""
            <div style="padding: 2px 0 8px;">
              <div class="match-pill">{confidence} Match</div>
              <div class="explain-tag">{explain[:60]}{'…' if len(explain)>60 else ''}</div>
              <div style="font-size:0.95rem;font-weight:600;color:#dfe2f3;line-height:1.3;margin-bottom:3px;">{title[:50]}{'…' if len(title)>50 else ''}</div>
              <div style="font-size:0.82rem;color:#8c90a1;margin-bottom:5px;">{author}</div>
              <div style="font-size:0.85rem;color:#ffd700;margin-bottom:6px;">{stars} <span style="color:#8c90a1;font-size:0.75rem;">{amz_rating if amz_rating else ''}</span></div>
              <div style="margin-bottom:8px;">{genre_chips}</div>
            </div>
            """, unsafe_allow_html=True)

            # Buttons
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔍 Details", key=f"det_{isbn}_{i}", use_container_width=True):
                    st.session_state.viewing_book = book
                    st.rerun()
            with b2:
                st.markdown(
                    f'<a href="{amz_url}" target="_blank" style="display:block;text-align:center;background:#ff9900;color:#111;padding:7px 4px;border-radius:10px;text-decoration:none;font-weight:700;font-size:0.78rem;line-height:1.6;">🛒 {price}</a>',
                    unsafe_allow_html=True
                )
            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Book Details Dialog
# ─────────────────────────────────────────────────────────────────
if "viewing_book" in st.session_state:
    show_book_details(st.session_state.viewing_book)
