from pydantic import BaseModel
from typing import List

class MovieScore(BaseModel):
    movie_id: int
    title: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieScore]

class SimilarMoviesResponse(BaseModel):
    movie_id: int
    similar_movies: List[MovieScore]
