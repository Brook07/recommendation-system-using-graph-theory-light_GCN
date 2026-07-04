"""GraphRec Streamlit frontend.

Modern discovery UI for the graph-based book recommender. The app calls the
FastAPI backend when available and falls back to the local recommendation
function for development/offline demos.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.book_reco.recommender_pipeline import recommend as local_recommend
except Exception:  # pragma: no cover - Streamlit should still render API mode
    local_recommend = None


API_URL = "http://localhost:8000"
DEFAULT_TOP_K = 12
PROFILE_DATA_PATH = REPO_ROOT / "data" / "processed" / "merged_reduced_lightgcn.csv"

COLORS = {
    "background": "#070d1d",
    "surface": "#0f1628",
    "surface_high": "#171f33",
    "surface_card": "#101827",
    "border": "rgba(179, 197, 255, 0.14)",
    "border_active": "#b3c5ff",
    "primary": "#b3c5ff",
    "cyan": "#00f1fe",
    "cyan_soft": "rgba(0, 241, 254, 0.18)",
    "text": "#eef3ff",
    "muted": "#9aa6c4",
}

SELECTED_USER_IDS = [11676, 153662, 95359]

st.set_page_config(
    page_title="GraphRec | Intelligent Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    """Inject application CSS."""

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&display=swap');

:root {{
    --bg: {COLORS["background"]};
    --surface: {COLORS["surface"]};
    --surface-high: {COLORS["surface_high"]};
    --card: {COLORS["surface_card"]};
    --border: {COLORS["border"]};
    --border-active: {COLORS["border_active"]};
    --primary: {COLORS["primary"]};
    --cyan: {COLORS["cyan"]};
    --cyan-soft: {COLORS["cyan_soft"]};
    --text: {COLORS["text"]};
    --muted: {COLORS["muted"]};
}}

html, body, [class*="st-"] {{
    font-family: 'Geist', Inter, system-ui, sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 5% 5%, rgba(0, 102, 255, 0.14), transparent 34rem),
        radial-gradient(circle at 95% 20%, rgba(0, 241, 254, 0.08), transparent 28rem),
        var(--bg);
    color: var(--text);
}}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {{
    display: none !important;
}}

.block-container {{
    max-width: 1500px;
    padding: 0 1.9rem 5rem;
}}

.top-nav {{
    height: 82px;
    margin: 0 -1.9rem 4.5rem;
    padding: 0 1.9rem;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: rgba(7, 13, 29, 0.72);
    backdrop-filter: blur(18px);
    position: sticky;
    top: 0;
    z-index: 50;
}}

.brand {{
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 800;
    color: var(--primary);
    letter-spacing: -0.04em;
    text-shadow: 0 0 18px rgba(179, 197, 255, 0.22);
}}

.nav-links {{
    display: flex;
    gap: 2.6rem;
    align-items: center;
    justify-content: center;
}}

.nav-link {{
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.35rem 0;
}}

.nav-link.active {{
    color: var(--primary);
    border-bottom: 2px solid var(--primary);
}}

.nav-actions {{
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
    align-items: center;
    color: var(--primary);
}}

.icon-bubble {{
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.04);
}}

.section-title {{
    color: rgba(238, 243, 255, 0.62);
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0 0 1.6rem;
}}

.persona-card {{
    height: 245px;
    border-radius: 2rem;
    padding: 1.75rem 1.85rem;
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid var(--border);
    transition: transform 180ms ease, border-color 180ms ease, background 180ms ease, box-shadow 180ms ease;
    position: relative;
    overflow: hidden;
}}

.persona-card.active {{
    background: linear-gradient(135deg, rgba(179, 197, 255, 0.15), rgba(255, 255, 255, 0.045));
    border-color: var(--border-active);
    box-shadow: 0 0 34px rgba(0, 102, 255, 0.18);
}}

.persona-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(179, 197, 255, 0.38);
}}

.persona-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}}

.profile-id-badge {{
    width: 58px;
    height: 58px;
    border-radius: 50%;
    border: 1px solid rgba(179, 197, 255, 0.42);
    display: grid;
    place-items: center;
    color: var(--cyan);
    font-weight: 900;
    font-size: 0.76rem;
    background: rgba(0, 241, 254, 0.08);
    box-shadow: 0 0 18px rgba(0, 241, 254, 0.16);
}}

.active-pill {{
    background: #0066ff;
    color: #f8f7ff;
    border-radius: 999px;
    padding: 0.34rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}}

.profile-title {{
    color: var(--text);
    font-size: 1.24rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}}

.profile-line {{
    color: #d4daf0;
    font-size: 0.82rem;
    line-height: 1.32;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.profile-line.muted {{
    color: #7f8bad;
}}

.profile-stat-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.4rem;
    margin: 0.65rem 0;
}}

.profile-stat {{
    border-radius: 0.8rem;
    background: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(179, 197, 255, 0.08);
    padding: 0.42rem 0.55rem;
}}

.profile-stat-label {{
    color: #7f8bad;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.06em;
}}

.profile-stat-value {{
    color: var(--text);
    font-size: 0.78rem;
    font-weight: 800;
}}

div[data-testid="stButton"] > button {{
    width: 100%;
    height: 2.45rem;
    border-radius: 999px;
    border: 1px solid rgba(179, 197, 255, 0.22);
    background: rgba(255, 255, 255, 0.045);
    color: var(--primary);
    font-weight: 800;
    letter-spacing: 0.03em;
}}

div[data-testid="stButton"] > button:hover {{
    border-color: var(--cyan);
    color: var(--cyan);
    background: rgba(0, 241, 254, 0.08);
}}

.results-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1rem;
    margin: 1.5rem 0 1.2rem;
}}

.active-user-strip {{
    max-width: 780px;
    margin: 2.7rem auto 2.5rem;
    padding: 1.05rem 1.35rem;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    color: var(--primary);
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid var(--border);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
    font-weight: 800;
    letter-spacing: 0.02em;
}}

.active-user-dot {{
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 14px rgba(0, 241, 254, 0.75);
}}

.results-title {{
    color: var(--text);
    font-size: 1.35rem;
    font-weight: 800;
}}

.results-meta {{
    color: var(--muted);
    font-size: 0.9rem;
}}

.book-card {{
    height: 585px;
    border-radius: 1.8rem;
    overflow: hidden;
    background: rgba(16, 24, 39, 0.78);
    border: 1px solid rgba(179, 197, 255, 0.12);
    box-shadow: 0 16px 42px rgba(0, 0, 0, 0.22);
    transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
    display: flex;
    flex-direction: column;
    margin-bottom: 1.8rem;
}}

.book-card:hover {{
    transform: translateY(-7px);
    border-color: rgba(0, 241, 254, 0.42);
    box-shadow: 0 24px 60px rgba(0, 241, 254, 0.12);
}}

.cover-wrap {{
    position: relative;
    height: 335px;
    flex: 0 0 335px;
    background: #080d18;
    overflow: hidden;
}}

.cover-wrap img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 550ms ease, filter 250ms ease;
}}

.book-card:hover .cover-wrap img {{
    transform: scale(1.055);
    filter: brightness(1.08) saturate(1.08);
}}

.similarity-badge {{
    position: absolute;
    top: 1rem;
    right: 1rem;
    border-radius: 999px;
    padding: 0.58rem 0.9rem;
    background: rgba(15, 22, 40, 0.82);
    border: 1px solid rgba(0, 241, 254, 0.48);
    color: var(--cyan);
    font-size: 0.75rem;
    font-weight: 900;
    letter-spacing: 0.04em;
    box-shadow: 0 0 20px rgba(0, 241, 254, 0.22);
}}

.score-label {{
    display: block;
    color: rgba(238, 243, 255, 0.72);
    font-size: 0.58rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}}

.book-body {{
    padding: 1rem 1.25rem 1.05rem;
    display: grid;
    grid-template-rows: 3rem 1.45rem 2.1rem 3.2rem;
    row-gap: 0.55rem;
    min-height: 250px;
    overflow: hidden;
}}

.book-title {{
    color: var(--text);
    font-size: 1.16rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.18;
    min-height: 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.book-author {{
    color: #d8deef;
    font-size: 0.92rem;
    margin-top: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.metadata-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0;
    padding-top: 0;
    max-height: 2.1rem;
    overflow: hidden;
    align-content: flex-start;
}}

.metadata-chip {{
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.065);
    color: #c9d0e5;
    padding: 0.28rem 0.58rem;
    font-size: 0.69rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}}

.explanation {{
    color: var(--primary);
    font-size: 0.78rem;
    line-height: 1.35;
    margin-top: 0;
    padding-top: 0.65rem;
    border-top: 1px solid rgba(179, 197, 255, 0.1);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}

.status-card {{
    border-radius: 1.25rem;
    padding: 1rem 1.15rem;
    background: rgba(0, 241, 254, 0.08);
    border: 1px solid rgba(0, 241, 254, 0.22);
    color: var(--text);
    margin-bottom: 1.2rem;
}}

.mobile-note {{
    display: none;
}}

@media (max-width: 960px) {{
    .top-nav {{
        grid-template-columns: 1fr auto;
        margin-bottom: 2.4rem;
    }}
    .nav-links {{
        display: none;
    }}
    .book-card {{
        height: 545px;
    }}
    .cover-wrap {{
        height: 300px;
        flex-basis: 300px;
    }}
}}

@media (max-width: 640px) {{
    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    .top-nav {{
        margin-left: -1rem;
        margin-right: -1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    .persona-card {{
        height: auto;
        min-height: 168px;
    }}
    .results-head {{
        display: block;
    }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def normalize_recommendation(rec: dict[str, Any]) -> dict[str, Any]:
    """Normalize backend and local recommendation key styles."""

    return {
        "isbn": str(rec.get("ISBN", "")),
        "title": str(rec.get("Book_Title", rec.get("Book-Title", "Unknown Title"))),
        "author": str(rec.get("Book_Author", rec.get("Book-Author", "Unknown Author"))),
        "year": str(rec.get("Year_Of_Publication", rec.get("Year-Of-Publication", "Unknown"))),
        "publisher": str(rec.get("Publisher", "Unknown")),
        "image": str(rec.get("Image_URL_L", rec.get("Image-URL-L", ""))),
        "score": float(rec.get("Score", 0.0) or 0.0),
        "fallback": bool(rec.get("Is_Fallback", rec.get("Is-Fallback", False))),
    }


def top_values(series: pd.Series, limit: int = 2) -> list[str]:
    """Return the most frequent non-empty values from a metadata column."""

    cleaned = series.dropna().astype(str)
    cleaned = cleaned[cleaned.str.strip().ne("")]
    return cleaned.value_counts().head(limit).index.tolist()


def profile_title(history: pd.DataFrame) -> str:
    """Create a defensible profile title from interaction statistics."""

    total = len(history)
    top_author_counts = history["Book-Author"].dropna().astype(str).value_counts()
    top_author = top_author_counts.index[0] if not top_author_counts.empty else ""
    top_author_share = float(top_author_counts.iloc[0] / total) if total and not top_author_counts.empty else 0.0
    valid_years = pd.to_numeric(history["Year-Of-Publication_Clean"], errors="coerce").dropna()
    median_year = int(valid_years.median()) if not valid_years.empty else 0

    if total >= 500:
        return "Highly Active Reader"
    if top_author and top_author_share >= 0.18:
        return f"{top_author}-Focused Reader"
    if median_year >= 1995:
        return "Contemporary Reader"
    if not valid_years.empty and valid_years.quantile(0.75) < 1985:
        return "Classic-Era Reader"
    return "Diverse Active Reader"


@st.cache_data(show_spinner=False, ttl=600)
def build_user_profiles(user_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    """Build display profiles from the reduced Book-Crossing interaction table."""

    columns = [
        "User-ID",
        "Book-Rating",
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Year-Of-Publication_Clean",
    ]
    history = pd.read_csv(PROFILE_DATA_PATH, usecols=columns, low_memory=False)
    profiles: list[dict[str, Any]] = []

    for user_id in user_ids:
        user_history = history[history["User-ID"] == user_id].copy()
        if user_history.empty:
            profiles.append(
                {
                    "key": str(user_id),
                    "title": "Dataset User",
                    "user_id": user_id,
                    "total_ratings": 0,
                    "avg_rating": 0.0,
                    "activity_level": "No retained interactions",
                    "top_authors": [],
                    "top_publishers": [],
                    "top_books": [],
                    "year_range": "Unknown",
                    "summary": "No retained interactions found in the reduced dataset.",
                }
            )
            continue

        valid_years = pd.to_numeric(user_history["Year-Of-Publication_Clean"], errors="coerce").dropna()
        year_range = f"{int(valid_years.min())}-{int(valid_years.max())}" if not valid_years.empty else "Unknown"
        total_ratings = len(user_history)
        avg_rating = float(user_history["Book-Rating"].mean())
        activity_level = (
            "Very high activity"
            if total_ratings >= 500
            else "High activity"
            if total_ratings >= 150
            else "Moderate activity"
        )
        top_authors = top_values(user_history["Book-Author"], 3)
        top_publishers = top_values(user_history["Publisher"], 2)
        top_books = top_values(user_history["Book-Title"], 2)
        title = profile_title(user_history)
        profiles.append(
            {
                "key": str(user_id),
                "title": title,
                "user_id": user_id,
                "total_ratings": total_ratings,
                "avg_rating": avg_rating,
                "activity_level": activity_level,
                "top_authors": top_authors,
                "top_publishers": top_publishers,
                "top_books": top_books,
                "year_range": year_range,
                "summary": (
                    f"{activity_level}; top author: {top_authors[0] if top_authors else 'Unknown'}; "
                    f"preferred years: {year_range}."
                ),
            }
        )

    return profiles


@st.cache_data(show_spinner=False, ttl=300)
def get_recommendations(user_id: int, top_k: int = DEFAULT_TOP_K) -> tuple[list[dict[str, Any]], str]:
    """Fetch recommendations from backend API, then local fallback if needed."""

    try:
        response = requests.get(
            f"{API_URL}/recommend/{user_id}",
            params={"top_k": top_k},
            timeout=12,
        )
        if response.status_code == 200:
            payload = response.json()
            return [normalize_recommendation(item) for item in payload.get("recommendations", [])], "api"
    except requests.RequestException:
        pass

    if local_recommend is not None:
        local_items = local_recommend(user_id=user_id, top_k=top_k)
        return [normalize_recommendation(item) for item in local_items], "local"

    return [], "unavailable"


def format_similarity_score(score: float) -> str:
    """Format the actual recommender score for display."""

    return f"{score:.3f}"


def metadata_chips(book: dict[str, Any]) -> list[str]:
    """Build chips only from fields present in the Book-Crossing metadata."""

    chips = []
    if book["year"] and book["year"] != "Unknown":
        chips.append(f"Year {book['year']}")
    if book["publisher"] and book["publisher"] != "Unknown":
        chips.append(book["publisher"])
    return chips[:2]


def recommendation_explanation(book: dict[str, Any], rank: int) -> str:
    """Explain the recommendation using the actual inference path."""

    if book["fallback"]:
        return "Popularity fallback: ranked highly among books in the training interactions."
    if rank == 1:
        return "Top-ranked recommendation for this user's learned reading profile."
    return "High embedding similarity in the user-book interaction graph."


def render_nav() -> None:
    """Render the top navigation bar."""

    st.markdown(
        """
<header class="top-nav">
    <div class="brand">GraphRec</div>
    <nav class="nav-links">
        <span class="nav-link active">Discover</span>
        <span class="nav-link">Library</span>
        <span class="nav-link">Analytics</span>
    </nav>
    <div class="nav-actions">
        <div class="icon-bubble">⚙</div>
        <div class="icon-bubble">GR</div>
    </div>
</header>
""",
        unsafe_allow_html=True,
    )


def render_persona_card(persona: dict[str, Any], active: bool) -> None:
    """Render a persona summary card."""

    active_class = " active" if active else ""
    active_badge = '<span class="active-pill">ACTIVE</span>' if active else ""
    html = (
        f'<div class="persona-card{active_class}">'
        '<div class="persona-top">'
        f'<img class="persona-avatar" src="{escape(persona["avatar"])}" alt="{escape(persona["name"])}" />'
        f"{active_badge}"
        "</div>"
        f'<div class="persona-name">{escape(persona["name"])}</div>'
        f'<div class="persona-focus">{escape(persona["focus"])}</div>'
        f'<div class="persona-focus" style="margin-top:.55rem;color:#7f8bad;">User {persona["user_id"]} · {persona["ratings"]} ratings</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_book_card(book: dict[str, Any], persona: dict[str, Any], rank: int, total: int) -> None:
    """Render one uniform recommendation card."""

    image = book["image"] if book["image"].startswith("http") else "https://via.placeholder.com/360x540/101827/b3c5ff?text=GraphRec"
    chips = metadata_chips(book)
    chip_html = "".join(f'<span class="metadata-chip">{escape(chip)}</span>' for chip in chips)
    score_label = "Popularity Score" if book["fallback"] else "Graph Similarity"
    explanation = recommendation_explanation(book, rank)
    title = escape(book["title"])
    author = escape(book["author"])
    year = escape(book["year"])

    st.markdown(
        f"""
<article class="book-card">
    <div class="cover-wrap">
        <img src="{escape(image)}" alt="{title}">
        <div class="similarity-badge"><span class="score-label">{score_label}</span>{format_similarity_score(book["score"])}</div>
    </div>
    <div class="book-body">
        <div class="book-title">{title}</div>
        <div class="book-author">{author} · {year}</div>
        <div class="metadata-chips">{chip_html}</div>
        <div class="explanation">{escape(explanation)}</div>
    </div>
</article>
""",
        unsafe_allow_html=True,
    )


def rerun_app() -> None:
    """Rerun the app across Streamlit versions."""

    if hasattr(st, "rerun"):
        st.rerun()
    else:
        getattr(st, "experimental_rerun")()


def main() -> None:
    """Render GraphRec."""

    inject_styles()
    render_nav()

    if "active_persona" not in st.session_state:
        st.session_state.active_persona = PERSONAS[0]["key"]

    st.markdown('<h2 class="section-title">Active Intelligence Persona</h2>', unsafe_allow_html=True)
    persona_cols = st.columns(3, gap="large")
    for col, persona in zip(persona_cols, PERSONAS):
        active = st.session_state.active_persona == persona["key"]
        with col:
            render_persona_card(persona, active)
            if st.button(f"Select {persona['name']}", key=f"select_{persona['key']}"):
                st.session_state.active_persona = persona["key"]
                rerun_app()

    active_persona = next(item for item in PERSONAS if item["key"] == st.session_state.active_persona)
    active_user_id = int(active_persona["user_id"])
    st.markdown(
        f"""
<div class="active-user-strip">
    <span class="active-user-dot"></span>
    Recommendations are live for {escape(active_persona['name'])} · User {active_user_id}
</div>
""",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading recommendations..."):
        recommendations, source = get_recommendations(active_user_id, DEFAULT_TOP_K)

    ranked_recommendations = sorted(recommendations, key=lambda item: item["score"], reverse=True)
    source_label = "Backend API" if source == "api" else "Local recommender" if source == "local" else "Unavailable"
    st.markdown(
        f"""
<div class="results-head">
    <div>
        <div class="results-title">Discovery Feed</div>
        <div class="results-meta">{escape(active_persona['name'])} · User {active_user_id} · {len(ranked_recommendations)} recommendations</div>
    </div>
    <div class="results-meta">Source: {source_label}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if recommendations and recommendations[0]["fallback"]:
        st.markdown(
            '<div class="status-card">Cold-start or embedding fallback is active. Results are ranked by popularity until LightGCN embeddings are available.</div>',
            unsafe_allow_html=True,
        )

    if not ranked_recommendations:
        st.warning("No recommendations returned for this persona.")
        return

    grid_cols = st.columns(4, gap="large")
    for idx, book in enumerate(ranked_recommendations):
        with grid_cols[idx % 4]:
            render_book_card(book, active_persona, idx + 1, len(ranked_recommendations))


if __name__ == "__main__":
    main()
