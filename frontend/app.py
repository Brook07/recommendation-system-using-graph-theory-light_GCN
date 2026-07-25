"""
GraphRec AI — Premium Redesign
Pure Python Streamlit UI with CSS theming.
All layout via st.columns / st.container / st.sidebar.
"""
import streamlit as st
import sys, urllib.parse
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from frontend.utils.api import fetch_recommendations, fetch_analytics
from frontend.utils.state import init_session_state, track_user_view
from frontend.components.dialogs import show_book_details

# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRec AI | Book Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_session_state()

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Variables ── */
:root {
  --bg:       #0B1120;
  --card:     #111827;
  --card2:    #1a2235;
  --primary:  #6366F1;
  --plight:   rgba(99,102,241,0.15);
  --secondary:#06B6D4;
  --accent:   #22C55E;
  --warning:  #F59E0B;
  --danger:   #EF4444;
  --text:     #F1F5F9;
  --muted:    #94A3B8;
  --border:   rgba(255,255,255,0.07);
  --border2:  rgba(99,102,241,0.3);
}

/* ── Base ── */
html, body, [class*="css"], [class*="st-"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Hide chrome ── */
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] { display:none !important; }

/* ── Layout ── */
.block-container {
  padding: 1.5rem 2rem 4rem !important;
  max-width: 1600px !important;
}
section[data-testid="stSidebar"] {
  background: #0d1526 !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.2rem 1rem !important; }

/* ── Sidebar elements ── */
section[data-testid="stSidebar"] label { color: #94A3B8 !important; font-size:0.72rem !important; font-weight:700 !important; text-transform:uppercase; letter-spacing:0.08em; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
section[data-testid="stSidebar"] .stSlider > div { color: var(--primary) !important; }

/* ── Main widgets ── */
[data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
}
[data-testid="stTextInput"] input {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important; color: var(--text) !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
  background: var(--plight) !important;
  border: 1px solid var(--border2) !important;
  color: #a5b4fc !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
  padding: 6px 12px !important;
  transition: all 0.2s !important;
  width: 100% !important;
}
[data-testid="stButton"] > button:hover {
  background: rgba(99,102,241,0.3) !important;
  border-color: var(--primary) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(99,102,241,0.25) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px !important;
  transition: all 0.25s;
}
[data-testid="stMetric"]:hover {
  border-color: var(--border2);
  box-shadow: 0 0 20px rgba(99,102,241,0.12);
  transform: translateY(-2px);
}
[data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 800 !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  color: var(--text) !important;
  font-weight: 600 !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 0.75rem !important; }

/* ── Info/Warning/Error ── */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 8px; }

/* ════════════════════════════════════════════
   BOOK CARD COMPONENTS
   ════════════════════════════════════════════ */

/* Card cover image — always 220px */
.gr-cover {
  height: 220px;
  overflow: hidden;
  position: relative;
  background: #1a2235;
}
.gr-cover img {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.5s cubic-bezier(.23,1,.32,1);
}

/* Card wrapper */
.gr-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s cubic-bezier(.23,1,.32,1);
  cursor: pointer;
  margin-bottom: 4px;
}
.gr-card:hover {
  transform: translateY(-5px) scale(1.02);
  border-color: rgba(99,102,241,0.4);
  box-shadow: 0 24px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.15);
}
.gr-card:hover .gr-cover img { transform: scale(1.08); }

/* Match & confidence badges */
.gr-badge-row {
  display: flex; gap: 6px; flex-wrap: wrap;
  margin-bottom: 8px;
}
.gr-match {
  background: rgba(6,182,212,0.15);
  border: 1px solid rgba(6,182,212,0.35);
  color: #22d3ee;
  font-size: 0.7rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: 0.10em;
  padding: 3px 9px; border-radius: 99px;
}
.gr-conf-high   { background:rgba(34,197,94,0.15); border:1px solid rgba(34,197,94,0.3); color:#4ade80; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:99px; }
.gr-conf-mid    { background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.3); color:#fbbf24; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:99px; }
.gr-conf-low    { background:rgba(239,68,68,0.15);  border:1px solid rgba(239,68,68,0.3);  color:#f87171; font-size:0.68rem; font-weight:700; padding:2px 7px; border-radius:99px; }

/* Card text area */
.gr-body { padding: 12px 12px 8px; }
.gr-title { font-size: 0.95rem; font-weight: 700; color: #F1F5F9; line-height: 1.35; margin-bottom: 3px; }
.gr-author { font-size: 0.78rem; color: #94A3B8; margin-bottom: 8px; }
.gr-tags { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }
.gr-tag-genre { background:rgba(99,102,241,0.15); color:#a5b4fc; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; padding:2px 8px; border-radius:5px; }
.gr-tag-sub   { background:rgba(6,182,212,0.10);  color:#67e8f9; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; padding:2px 8px; border-radius:5px; }

/* Meta grid */
.gr-meta {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 4px 8px; margin-bottom: 8px;
}
.gr-meta-item { font-size: 0.72rem; color: #64748B; }
.gr-meta-item b { color: #94A3B8; }

/* Explanation tags */
.gr-why { margin-bottom: 8px; }
.gr-why-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #6366F1; margin-bottom: 4px; }
.gr-why-tag {
  display:inline-block; background:rgba(99,102,241,0.10);
  border-left: 2px solid #6366F1;
  color: #c7d2fe; font-size: 0.7rem; padding: 2px 6px;
  border-radius: 0 4px 4px 0; margin: 2px 0;
}

/* ── Section headings ── */
.gr-section-head {
  display: flex; align-items: center; gap: 10px;
  font-size: 1.15rem; font-weight: 700; color: #F1F5F9;
  margin-bottom: 16px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.gr-section-badge {
  background: var(--plight); border: 1px solid var(--border2);
  color: #a5b4fc; font-size: 0.68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 2px 8px; border-radius: 6px;
}

/* ── Header badges ── */
.gr-tech-badge {
  display:inline-block; background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  color: #CBD5E1; font-size: 0.75rem; font-weight: 600;
  padding: 4px 12px; border-radius: 99px; margin: 3px;
}

/* ── Stat card ── */
.gr-stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 18px 20px;
  transition: all 0.25s; text-align: center;
}
.gr-stat-card:hover {
  border-color: var(--border2);
  box-shadow: 0 0 24px rgba(99,102,241,0.12);
  transform: translateY(-3px);
}
.gr-stat-icon { font-size: 1.8rem; margin-bottom: 6px; display:block; }
.gr-stat-num  { font-size: 2rem; font-weight: 800; color: var(--primary); display:block; line-height:1; }
.gr-stat-lbl  { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-top: 4px; display:block; }
.gr-stat-desc { font-size: 0.68rem; color: #475569; margin-top: 2px; }

/* ── Sidebar profile ── */
.gr-profile {
  background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.08) 100%);
  border: 1px solid rgba(99,102,241,0.25);
  border-radius: 16px; padding: 16px; margin-bottom: 16px;
  text-align: center;
}
.gr-avatar {
  width: 56px; height: 56px; border-radius: 50%;
  background: linear-gradient(135deg, #6366F1, #06B6D4);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; margin: 0 auto 10px;
  border: 2px solid rgba(99,102,241,0.4);
  box-shadow: 0 0 16px rgba(99,102,241,0.3);
}
.gr-profile-name { font-size: 0.88rem; font-weight: 700; color: #F1F5F9; margin-bottom: 2px; }
.gr-profile-uid  { font-size: 0.72rem; color: #6366F1; font-weight: 600; }
.gr-profile-genre { font-size: 0.7rem; color: #94A3B8; margin-top: 2px; }

/* ── Status dot ── */
.gr-status { display:inline-flex; align-items:center; gap:6px; font-size:0.78rem; font-weight:600; color:#4ade80; }
.gr-dot { width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 6px #22c55e; animation: gr-pulse 2s infinite; }
@keyframes gr-pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }

/* ── Footer ── */
.gr-footer {
  margin-top: 48px; padding: 24px 0 8px;
  border-top: 1px solid var(--border);
  text-align: center;
}
.gr-footer-tech { display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-bottom:12px; }
.gr-footer-badge {
  background: rgba(255,255,255,0.04); border: 1px solid var(--border);
  color: #64748B; font-size: 0.7rem; padding: 3px 10px; border-radius: 6px;
}

/* ── Animations ── */
@keyframes gr-fadein { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:translateY(0);} }
.gr-card  { animation: gr-fadein 0.4s ease both; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────
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
        return {}, None
    df_p = pd.read_csv(PERSONAS_FILE, dtype={"User-ID": int})
    valid_rated  = set(pd.read_csv(RATINGS_FILE,  usecols=["User-ID"])["User-ID"].astype(int)) if RATINGS_FILE.exists()  else set()
    valid_mapped = set(pd.read_csv(USER_MAP_FILE, usecols=["User-ID"])["User-ID"].astype(int)) if USER_MAP_FILE.exists() else set()
    demo = {}
    for persona, grp in df_p.groupby("Persona"):
        row = grp.iloc[len(grp) // 2]
        uid = int(row["User-ID"])
        if uid in valid_rated and (not valid_mapped or uid in valid_mapped):
            demo[persona] = {"user_id": uid, "persona": persona,
                             "primary_genre": row["Primary_Genre"],
                             "secondary_genre": row["Secondary_Genre"]}
    return demo, None

demo_users, _ = load_demo_personas()
persona_labels = list(demo_users.keys())

if "active_persona" not in st.session_state:
    st.session_state.active_persona = persona_labels[0] if persona_labels else None

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Profile ──────────────────────────────────────────────────
    active_persona = st.session_state.active_persona
    active_info    = demo_users.get(active_persona, {})
    emoji          = PERSONA_EMOJI.get(active_persona, "📚")

    st.markdown(f"""
    <div class="gr-profile">
      <div class="gr-avatar">{emoji}</div>
      <div class="gr-profile-name">{active_persona or 'Select Persona'}</div>
      <div class="gr-profile-uid">User ID · {active_info.get('user_id','—')}</div>
      <div class="gr-profile-genre">📖 {active_info.get('primary_genre','—')}</div>
      <div class="gr-profile-genre">📚 {active_info.get('secondary_genre','—')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Persona picker ────────────────────────────────────────────
    st.markdown("**🎭 Select Persona**")
    if persona_labels:
        chosen = st.selectbox(
            "Persona", persona_labels,
            index=persona_labels.index(active_persona) if active_persona in persona_labels else 0,
            format_func=lambda p: f"{PERSONA_EMOJI.get(p,'📚')} {p}",
            label_visibility="collapsed",
        )
        if chosen != st.session_state.active_persona:
            st.session_state.active_persona = chosen
            st.session_state.selected_user = demo_users[chosen]["user_id"]
            track_user_view(st.session_state.selected_user)
            st.rerun()

    st.markdown("**🆔 Custom User ID**")
    custom_uid = st.text_input("Custom UID", value="", placeholder="e.g. 10042",
                                label_visibility="collapsed", key="sb_custom_uid")
    if custom_uid and custom_uid.isdigit():
        cid = int(custom_uid)
        if cid != st.session_state.selected_user:
            st.session_state.selected_user = cid
            track_user_view(cid)
            st.rerun()

    st.markdown("---")

    # ── Model Status ─────────────────────────────────────────────
    st.markdown("**⚡ System Status**")
    stats = fetch_analytics() or {}
    model_ok = bool(stats)
    if model_ok:
        st.markdown('<div class="gr-status"><div class="gr-dot"></div>Backend Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#f87171;font-size:0.78rem;font-weight:600;">🔴 Backend Offline</div>', unsafe_allow_html=True)

    if stats:
        st.markdown(f"""
        <div style="margin-top:10px;">
          <div style="font-size:0.7rem;color:#64748B;margin-bottom:6px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Dataset</div>
          <div style="font-size:0.82rem;color:#94A3B8;">📚 <b style="color:#F1F5F9;">{stats.get('Total_Books',0):,}</b> Books</div>
          <div style="font-size:0.82rem;color:#94A3B8;">👤 <b style="color:#F1F5F9;">{stats.get('Total_Users',0):,}</b> Users</div>
          <div style="font-size:0.82rem;color:#94A3B8;">⭐ <b style="color:#F1F5F9;">{stats.get('Total_Ratings',0):,}</b> Ratings</div>
          <div style="font-size:0.82rem;color:#94A3B8;">🏷️ <b style="color:#F1F5F9;">{stats.get('Unique_Genres','—')}</b> Genres</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Settings ─────────────────────────────────────────────────
    st.markdown("**⚙️ Recommendation Settings**")
    top_k = st.slider("Top K Results", 4, 20, 12, 4)
    sort_by = st.selectbox("Sort By", [
        "Highest Match", "Highest Amazon Rating",
        "Most Popular", "Price: Low to High", "Price: High to Low", "A–Z"
    ], label_visibility="visible")
    show_explain = st.toggle("Show Explanation Tags", value=True)

    st.markdown("---")

    # ── Search ───────────────────────────────────────────────────
    st.markdown("**🔍 Search**")
    search_q = st.text_input("Search", value="", placeholder="Title, author, genre…",
                              label_visibility="collapsed", key="sb_search")

    # Genre filter
    st.markdown("**🏷️ Filter by Genre**")
    genre_filter = st.selectbox("Genre filter", ["All Genres"], label_visibility="collapsed", key="sb_genre")

# ─────────────────────────────────────────────────────────────────
# RESOLVE ACTIVE USER
# ─────────────────────────────────────────────────────────────────
if not custom_uid or not custom_uid.isdigit():
    active_info = demo_users.get(st.session_state.active_persona, {})
    target_user = active_info.get("user_id", st.session_state.get("selected_user", 10000))
    if st.session_state.selected_user != target_user:
        st.session_state.selected_user = target_user
        track_user_view(target_user)

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 10px;">
  <div style="font-size:2.6rem;font-weight:900;background:linear-gradient(135deg,#6366F1,#06B6D4);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:6px;">
    📚 GraphRec AI
  </div>
  <div style="font-size:1rem;color:#94A3B8;font-weight:400;margin-bottom:14px;">
    Graph Neural Network Powered Book Recommendation System
  </div>
  <div>
    <span class="gr-tech-badge">🔬 LightGCN</span>
    <span class="gr-tech-badge">🕸️ Graph Theory</span>
    <span class="gr-tech-badge">🧠 Explainable AI</span>
    <span class="gr-tech-badge">⚡ FastAPI</span>
    <span class="gr-tech-badge">🎈 Streamlit</span>
    <span class="gr-tech-badge">🔥 PyTorch</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ───────────────────────────────────────────────────
sc1, sc2, sc3, sc4 = st.columns(4)
stat_data = [
    (sc1, "📚", f"{stats.get('Total_Books',0):,}",   "Books",   "In the catalog"),
    (sc2, "👤", f"{stats.get('Total_Users',0):,}",   "Users",   "Synthetic personas"),
    (sc3, "⭐", f"{stats.get('Total_Ratings',0):,}",  "Ratings", "User interactions"),
    (sc4, "🏷️", str(stats.get('Unique_Genres','—')), "Genres",  "Content categories"),
]
for col, icon, num, lbl, desc in stat_data:
    col.markdown(f"""
    <div class="gr-stat-card">
      <span class="gr-stat-icon">{icon}</span>
      <span class="gr-stat-num">{num}</span>
      <span class="gr-stat-lbl">{lbl}</span>
      <div class="gr-stat-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FETCH RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────
with st.spinner("Crunching the graph…"):
    recs_response = fetch_recommendations(st.session_state.selected_user, top_k=top_k)

recs = recs_response.get("recommendations", []) if recs_response else []
inference_time = recs_response.get("inference_time_ms", 0) if recs_response else 0

# ─────────────────────────────────────────────────────────────────
# WELCOME ROW — metrics
# ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("🎭 Active Persona", (active_persona or "—")[:22] + ("…" if active_persona and len(active_persona)>22 else ""))
m2.metric("⏱️ Inference Time", f"{inference_time} ms")
m3.metric("📋 Results Loaded", str(len(recs)))
m4.metric("👤 User ID", str(st.session_state.selected_user))

st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# RECOMMENDATIONS SECTION
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gr-section-head">
  📖 Recommended for You
  <span class="gr-section-badge">AI Powered</span>
</div>
""", unsafe_allow_html=True)

if not recs:
    st.warning("No recommendations returned. Make sure the FastAPI backend is running and the LightGCN model is trained.")
else:
    # ── Filter ────────────────────────────────────────────────────
    filtered = recs
    if search_q:
        sq = search_q.lower()
        filtered = [r for r in recs if sq in r.get("Book_Title","").lower()
                                    or sq in r.get("Author","").lower()
                                    or sq in r.get("Main_Genre","").lower()]

    # Update genre filter options after first load
    all_genres = sorted(set(r.get("Main_Genre","") for r in recs if r.get("Main_Genre","")))

    # ── Sort ──────────────────────────────────────────────────────
    def sort_key(r):
        if sort_by == "Highest Amazon Rating": return -(float(r.get("Amazon_Rating") or 0))
        if sort_by == "Most Popular":          return -(float(r.get("Number_of_People_Rated") or 0))
        if sort_by == "Price: Low to High":
            p = str(r.get("Price","")).replace("₹","").replace(",","")
            try: return float(p)
            except: return 9999
        if sort_by == "Price: High to Low":
            p = str(r.get("Price","")).replace("₹","").replace(",","")
            try: return -float(p)
            except: return -9999
        if sort_by == "A–Z":                   return r.get("Book_Title","")
        return -(float(r.get("Recommendation_Score") or 0))  # Highest Match (default)

    filtered = sorted(filtered, key=sort_key)

    st.caption(f"Showing {len(filtered)} of {len(recs)} recommendations · Sorted by: {sort_by} · {inference_time}ms inference")

    # ── Book Grid ─────────────────────────────────────────────────
    COLS = 4
    cols = st.columns(COLS, gap="medium")

    for i, book in enumerate(filtered):
        title      = str(book.get("Book_Title", "Unknown Title"))
        author     = str(book.get("Author", "Unknown Author"))
        genre      = str(book.get("Main_Genre", ""))
        sub_genre  = str(book.get("Sub_Genre", ""))
        book_type  = str(book.get("Book_Type", ""))
        price      = str(book.get("Price", "N/A"))
        amz_rating = float(book.get("Amazon_Rating") or 0)
        n_rated    = int(book.get("Number_of_People_Rated") or 0)
        comm_rating= float(book.get("Dataset_Average_Rating") or 0)
        rec_score  = float(book.get("Recommendation_Score") or 0)
        confidence = str(book.get("Confidence_Score", "N/A"))
        tags       = book.get("Explanation_Tags", [])
        amz_url    = book.get("Amazon_URL") or "#"
        isbn       = str(book.get("ISBN", i))

        # Confidence colour
        conf_pct = float(confidence.replace("%","")) if "%" in confidence else 0
        if conf_pct >= 70:   conf_cls = "gr-conf-high"
        elif conf_pct >= 40: conf_cls = "gr-conf-mid"
        else:                conf_cls = "gr-conf-low"

        cover_url = str(book.get("Book_Cover_URL","")).strip()
        if not cover_url or cover_url.lower() == "nan":
            safe = urllib.parse.quote(title[:20])
            cover_url = f"https://placehold.co/220x320/111827/6366F1?text={safe}"

        stars = "★" * int(amz_rating) + "☆" * (5 - int(amz_rating))

        genre_tag = f'<span class="gr-tag-genre">{genre}</span>' if genre and genre not in ("nan","") else ""
        sub_tag   = f'<span class="gr-tag-sub">{sub_genre}</span>' if sub_genre and sub_genre not in ("nan","Unknown","") else ""

        explain_html = ""
        if show_explain and tags:
            explain_html = '<div class="gr-why"><div class="gr-why-label">Why Recommended</div>'
            for t in tags[:3]:
                explain_html += f'<div class="gr-why-tag">✓ {t}</div>'
            explain_html += "</div>"

        anim_delay = f"{(i % COLS) * 0.08:.2f}s"

        with cols[i % COLS]:
            # Cover image (in HTML for fixed height control)
            st.markdown(f"""
            <div class="gr-card" style="animation-delay:{anim_delay}">
              <div class="gr-cover">
                <img src="{cover_url}" loading="lazy"
                     onerror="this.src='https://placehold.co/220x320/111827/6366F1?text=No+Cover'">
              </div>
              <div class="gr-body">
                <div class="gr-badge-row">
                  <span class="gr-match">{confidence} Match</span>
                  <span class="{conf_cls}">{
                      'High' if conf_pct>=70 else 'Medium' if conf_pct>=40 else 'Low'
                  }</span>
                </div>
                <div class="gr-title">{title}</div>
                <div class="gr-author">✍️ {author}</div>
                <div class="gr-tags">{genre_tag}{sub_tag}</div>
                <div class="gr-meta">
                  <div class="gr-meta-item">⭐ <b>{amz_rating:.1f}</b>/5</div>
                  <div class="gr-meta-item">👥 <b>{comm_rating:.1f}</b> comm.</div>
                  <div class="gr-meta-item">💰 <b>{price}</b></div>
                  <div class="gr-meta-item">📊 <b>{n_rated:,}</b> rated</div>
                </div>
                {explain_html}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons (native Streamlit so they are clickable)
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🔍 Details", key=f"det_{isbn}_{i}", use_container_width=True):
                    st.session_state.viewing_book = book
                    st.rerun()
            with b2:
                st.markdown(
                    f'<a href="{amz_url}" target="_blank" style="display:block;text-align:center;'
                    f'background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.35);'
                    f'color:#fbbf24;padding:6px 4px;border-radius:10px;text-decoration:none;'
                    f'font-weight:700;font-size:0.76rem;line-height:1.6;">🛒 {price}</a>',
                    unsafe_allow_html=True,
                )
            with b3:
                st.markdown(
                    f'<a href="{amz_url}" target="_blank" style="display:block;text-align:center;'
                    f'background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);'
                    f'color:#4ade80;padding:6px 4px;border-radius:10px;text-decoration:none;'
                    f'font-weight:700;font-size:0.76rem;line-height:1.6;">👁️ Preview</a>',
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# ANALYTICS SECTION
# ─────────────────────────────────────────────────────────────────
if recs:
    st.markdown("<div style='margin:32px 0 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="gr-section-head">
      📊 Recommendation Analytics
      <span class="gr-section-badge">Live Data</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📈 View Analytics Dashboard", expanded=False):
        a1, a2, a3 = st.columns(3)

        # Genre distribution pie
        genre_counts = pd.Series([r.get("Main_Genre","Unknown") for r in recs]).value_counts()
        with a1:
            st.markdown("**🏷️ Genre Distribution**")
            if HAS_PLOTLY:
                fig = px.pie(
                    names=genre_counts.index, values=genre_counts.values,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    hole=0.45,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F1F5F9", margin=dict(t=10,b=10,l=10,r=10),
                    showlegend=True, legend=dict(font=dict(size=10)),
                    height=260,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(genre_counts)

        # Recommendation score histogram
        scores = [float(r.get("Recommendation_Score") or 0) for r in recs]
        with a2:
            st.markdown("**📉 Score Distribution**")
            if HAS_PLOTLY:
                fig2 = px.histogram(
                    x=scores, nbins=10,
                    color_discrete_sequence=["#6366F1"],
                    labels={"x":"Recommendation Score","y":"Count"},
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#F1F5F9", margin=dict(t=10,b=10,l=10,r=10),
                    bargap=0.1, height=260,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(pd.Series(scores).value_counts())

        # Confidence bar chart
        conf_values = []
        for r in recs:
            c = str(r.get("Confidence_Score","0%")).replace("%","")
            try: conf_values.append(float(c))
            except: conf_values.append(0)
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

        with a3:
            st.markdown("**🎯 Confidence Metrics**")
            st.metric("Average Confidence", f"{avg_conf:.1f}%")
            st.metric("Max Confidence", f"{max(conf_values):.1f}%")
            st.metric("Min Confidence", f"{min(conf_values):.1f}%")
            if avg_conf > 0:
                st.progress(int(avg_conf))

# ─────────────────────────────────────────────────────────────────
# GRAPH THEORY PANEL
# ─────────────────────────────────────────────────────────────────
st.markdown("<div style='margin:16px 0 0;'></div>", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_graph_stats():
    gs = {}
    user_map_path = GRAPH_DIR / "user_mapping.csv"
    item_map_path = GRAPH_DIR / "item_mapping.csv"
    edge_path     = GRAPH_DIR / "edge_index.pt"
    if user_map_path.exists():
        gs["Users (Nodes)"] = len(pd.read_csv(user_map_path))
    if item_map_path.exists():
        gs["Books (Nodes)"] = len(pd.read_csv(item_map_path))
    gs["Total Nodes"]  = gs.get("Users (Nodes)",0) + gs.get("Books (Nodes)",0)
    if RATINGS_FILE.exists():
        gs["Edges (Ratings)"] = len(pd.read_csv(RATINGS_FILE))
    return gs

with st.expander("🕸️ Graph Theory Panel", expanded=False):
    gs = load_graph_stats()
    gp_cols = st.columns(4)
    gp_items = list(gs.items())
    for j, (k, v) in enumerate(gp_items):
        gp_cols[j % 4].metric(k, f"{v:,}")

    st.markdown("""
    <div style="margin-top:14px;font-size:0.82rem;color:#64748B;line-height:1.8;">
      <b style="color:#94A3B8;">Architecture:</b> LightGCN (Light Graph Convolutional Network) ·
      <b style="color:#94A3B8;">Propagation:</b> Bipartite User-Item Graph ·
      <b style="color:#94A3B8;">Embedding:</b> Collaborative Filtering via Graph Convolution ·
      <b style="color:#94A3B8;">Explainability:</b> Genre + Collaborative Similarity Scores
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# BOOK DETAILS DIALOG
# ─────────────────────────────────────────────────────────────────
if "viewing_book" in st.session_state:
    show_book_details(st.session_state.viewing_book)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="gr-footer">
  <div class="gr-footer-tech">
    <span class="gr-footer-badge">🕸️ Graph Theory</span>
    <span class="gr-footer-badge">🔬 LightGCN</span>
    <span class="gr-footer-badge">🔥 PyTorch</span>
    <span class="gr-footer-badge">⚡ FastAPI</span>
    <span class="gr-footer-badge">🎈 Streamlit</span>
    <span class="gr-footer-badge">📚 Google Books API</span>
  </div>
  <div style="font-size:0.7rem;color:#334155;">
    GraphRec AI · v2.0 · Dataset: {stats.get('Total_Books',0):,} books · {stats.get('Total_Ratings',0):,} ratings
  </div>
</div>
""", unsafe_allow_html=True)
