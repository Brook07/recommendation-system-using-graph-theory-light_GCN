from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import time
import logging
from backend.inference import InferenceEngine

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="GraphRec Explainable API", 
    description="Book Recommendation API using LightGCN with Explainable AI mechanics.", 
    version="2.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def load_model():
    global engine
    engine = InferenceEngine()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"{request.method} {request.url.path} - Completed in {process_time:.4f}s - Status: {response.status_code}")
    return response

# --- Pydantic Models ---

class ExplainableRecommendation(BaseModel):
    ISBN: str
    Book_Title: str
    Author: str
    Main_Genre: str
    Sub_Genre: str
    Book_Type: str
    Price: str
    Amazon_Rating: float
    Number_of_People_Rated: int
    Dataset_Average_Rating: float
    Recommendation_Score: float
    Confidence_Score: str
    Explanation_Tags: List[str]
    Book_Cover_URL: str
    Amazon_URL: str
    
    # Backwards compatibility fields for Streamlit (if it expects them)
    Book_Author: Optional[str] = Field(None, alias='Author')
    Score: Optional[float] = Field(None, alias='Recommendation_Score')

class RecommendationResponse(BaseModel):
    user_id: int
    inference_time_ms: float
    recommendations: List[ExplainableRecommendation]

class UserProfileResponse(BaseModel):
    Favorite_Genres: List[str]
    Favorite_Authors: List[str]
    Books_Rated: int
    Average_Rating: float
    Reading_Diversity: int
    Total_Recommendations_Generated: int

class AnalyticsResponse(BaseModel):
    Total_Users: int
    Total_Books: int
    Total_Ratings: int
    Most_Popular_Genres: List[str]
    Most_Active_Users: List[int]
    Most_Recommended_Books: List[str]


# --- Endpoints ---

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(user_id: int, top_k: int = 10):
    """
    Returns top-k recommendations for a given user, complete with metadata and explainable tags.
    """
    t0 = time.time()
    try:
        recs = engine.get_recommendations(user_id, top_k=top_k)
        if not recs:
            raise HTTPException(status_code=404, detail="User not found or model not loaded.")
            
        formatted_recs = []
        for r in recs:
            # Create object mapping. Include fallback fields for existing UI if necessary.
            formatted_recs.append(ExplainableRecommendation(**r))
            
        inference_time_ms = (time.time() - t0) * 1000
        logging.info(f"Generated {len(formatted_recs)} recommendations for User {user_id}")
        
        return RecommendationResponse(
            user_id=user_id, 
            inference_time_ms=round(inference_time_ms, 2),
            recommendations=formatted_recs
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/profile", response_model=UserProfileResponse, tags=["Users"])
def get_user_profile(user_id: int):
    """
    Returns the user's reading profile based on their interactions.
    """
    profile = engine.get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User profile not found. User may have zero positive ratings.")
    return UserProfileResponse(**profile)

@app.get("/books/{isbn}", tags=["Books"])
def get_book_metadata(isbn: str):
    """
    Returns complete metadata for a given ISBN.
    """
    book = engine.get_book_metadata(isbn)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found.")
    return book

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics():
    """
    Returns dataset and system analytics.
    """
    return AnalyticsResponse(**engine.get_analytics())

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
