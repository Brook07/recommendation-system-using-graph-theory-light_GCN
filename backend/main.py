from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from backend.inference import InferenceEngine

app = FastAPI(title="GraphRec API", description="Book Recommendation API using LightGCN", version="1.0.0")

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
    engine = InferenceEngine(model_path="models/best_lightgcn_model.pt")

class Recommendation(BaseModel):
    ISBN: str
    Book_Title: str
    Book_Author: str
    Year_Of_Publication: str
    Publisher: str
    Image_URL_L: str
    Score: float
    Is_Fallback: bool

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, top_k: int = 10):
    try:
        recs = engine.get_recommendations(user_id, top_k=top_k)
        formatted_recs = []
        for r in recs:
            formatted_recs.append(Recommendation(
                ISBN=r["ISBN"],
                Book_Title=r["Book-Title"],
                Book_Author=r["Book-Author"],
                Year_Of_Publication=str(r["Year-Of-Publication"]),
                Publisher=r["Publisher"],
                Image_URL_L=r["Image-URL-L"],
                Score=round(r["Score"], 4),
                Is_Fallback=r["Is-Fallback"]
            ))
        return RecommendationResponse(user_id=user_id, recommendations=formatted_recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
