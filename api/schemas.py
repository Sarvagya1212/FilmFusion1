from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class MovieScore(BaseModel):
    movie_id: int
    title: str
    score: float
    year: Optional[int] = None
    genres: Optional[List[str]] = None

class UserStats(BaseModel):
    total_ratings: int
    avg_rating: float
    favorite_genres: List[str]
    rating_distribution: dict

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieScore]
    user_stats: Optional[UserStats] = None
    timestamp: str

class SimilarMoviesResponse(BaseModel):
    movie_id: int
    similar_movies: List[MovieScore]
    original_movie: Optional[dict] = None
    timestamp: str
