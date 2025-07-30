from fastapi import APIRouter, HTTPException
from .schemas import RecommendationResponse, SimilarMoviesResponse

router = APIRouter()

@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, n: int = 10):
    """Fetch top-N recommendations for a user."""
    from src.serving.inference import recommend_for_user
    recs = recommend_for_user(user_id, n)
    if recs is None:
        raise HTTPException(status_code=404, detail="User not found or no recommendations available")
    return {"user_id": user_id, "recommendations": recs}

@router.get("/similar_movies/{movie_id}", response_model=SimilarMoviesResponse)
def get_similar_movies(movie_id: int, n: int = 10):
    """Fetch top-N similar movies."""
    from src.serving.inference import similar_movies_for_id
    sims = similar_movies_for_id(movie_id, n)
    if sims is None:
        raise HTTPException(status_code=404, detail="Movie not found or no similar movies available")
    return {"movie_id": movie_id, "similar_movies": sims}
