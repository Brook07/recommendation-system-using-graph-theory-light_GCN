import streamlit as st
import requests

# Constants
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="GraphRec - Book Recommendations",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for styling cards
st.markdown("""
<style>
    .book-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
        color: #333;
    }
    .book-title {
        font-size: 1.1em;
        font-weight: bold;
        margin-top: 10px;
        color: #1f77b4;
    }
    .book-author {
        font-size: 0.9em;
        color: #555;
    }
    .book-score {
        margin-top: 10px;
        font-size: 0.9em;
        font-weight: bold;
        color: #d62728;
    }
    .fallback-alert {
        padding: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 GraphRec: Book Recommendation Engine")
st.markdown("Powered by LightGCN (Graph Neural Networks)")

# Sidebar for input
with st.sidebar:
    st.header("User Settings")
    user_id = st.number_input("Enter User ID", min_value=1, value=276964, step=1)
    generate_btn = st.button("Generate Recommendations", type="primary")
    
    st.markdown("---")
    st.markdown("**Sample Users:**")
    st.markdown("- `276964` (Known User)")
    st.markdown("- `277042` (Known User)")
    st.markdown("- `999999` (Unknown User - triggers Fallback)")

if generate_btn:
    with st.spinner(f"Fetching recommendations for User {user_id}..."):
        try:
            response = requests.get(f"{API_URL}/recommend/{user_id}", params={"top_k": 10})
            if response.status_code == 200:
                data = response.json()
                recs = data.get("recommendations", [])
                
                if recs and recs[0].get("Is_Fallback"):
                    st.markdown("""
                    <div class="fallback-alert">
                        <strong>New or Unknown User!</strong> We don't have enough history for you yet, so we're recommending our all-time most popular and highly-rated books.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success(f"Top 10 Personalized Recommendations for User {user_id}")

                # Display in grid
                cols = st.columns(5)
                for idx, rec in enumerate(recs):
                    col = cols[idx % 5]
                    with col:
                        # Image handling
                        img_url = rec.get("Image_URL_L")
                        if img_url and str(img_url).startswith("http"):
                            st.image(img_url, use_container_width=True)
                        else:
                            # Placeholder image
                            st.image("https://via.placeholder.com/150x200?text=No+Cover", use_container_width=True)
                        
                        st.markdown(f"""
                        <div class="book-card">
                            <div class="book-title">{rec['Book_Title']}</div>
                            <div class="book-author">by {rec['Book_Author']}</div>
                            <div class="book-author" style="font-size:0.8em">{rec['Publisher']} ({rec['Year_Of_Publication']})</div>
                            <div class="book-score">Score: {rec['Score']:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Is the FastAPI server running?")
