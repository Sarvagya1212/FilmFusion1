import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging
from scipy.sparse import csr_matrix

from ..base_recommender import BaseRecommender

class TFIDFRecommender(BaseRecommender):
    """Content-based recommender using TF-IDF vectors from movie descriptions"""
    
    def __init__(self, similarity_threshold=0.1):
        super().__init__("TF-IDF Content-Based Recommender")
        self.similarity_threshold = similarity_threshold
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movies_df = None
        
    def fit(self, ratings_df: pd.DataFrame, tfidf_matrix: csr_matrix, 
            movies_df: pd.DataFrame, **kwargs):
        """Train content-based model using TF-IDF features"""
        self.logger.info(f"Training {self.name}...")
        
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.tfidf_matrix = tfidf_matrix
        
        # Create movie ID to index mapping
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Compute cosine similarity matrix
        self.logger.info("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Apply similarity threshold
        self.similarity_matrix[self.similarity_matrix < self.similarity_threshold] = 0
        
        self.is_fitted = True
        self.logger.info(f"✅ {self.name} training completed")
        self.logger.info(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        
        return self
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get most similar movies to a given movie"""
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Get indices of most similar movies
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_similar+1]  # Exclude self
        
        similar_movies = []
        for idx in similar_indices:
            if similarity_scores[idx] > 0:  # Only include movies above threshold
                similar_movie_id = self.idx_to_movie_id[idx]
                similar_movies.append((similar_movie_id, similarity_scores[idx]))
        
        return similar_movies
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using content-based approach"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get user's rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return 3.0  # Default rating for new users
        
        # Get similar movies to the target item
        similar_movies = self.get_similar_movies(item_id, n_similar=50)
        
        if not similar_movies:
            # Fall back to user's average rating
            return user_ratings['rating'].mean()
        
        # Calculate weighted average rating based on similarity
        numerator = 0
        denominator = 0
        
        for similar_movie_id, similarity in similar_movies:
            user_movie_rating = user_ratings[user_ratings['movieId'] == similar_movie_id]
            
            if len(user_movie_rating) > 0:
                rating = user_movie_rating.iloc[0]['rating']
                numerator += similarity * rating
                denominator += similarity
        
        if denominator == 0:
            return user_ratings['rating'].mean()
        
        predicted_rating = numerator / denominator
        
        # Ensure rating is within valid range
        return max(0.5, min(5.0, predicted_rating))
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate content-based recommendations"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get user's rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            # For new users, recommend popular movies
            return self._recommend_popular_movies(n_recommendations)
        
        # Get movies already seen by user
        seen_items = set()
        if exclude_seen:
            seen_items = self.get_user_seen_items(user_id, self.ratings_df)
        
        # Find movies similar to user's highly rated movies
        candidate_movies = {}
        
        # Focus on movies rated >= 4.0
        high_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]
        
        for _, rating_row in high_rated_movies.iterrows():
            movie_id = rating_row['movieId']
            user_rating = rating_row['rating']
            
            # Get similar movies
            similar_movies = self.get_similar_movies(movie_id, n_similar=20)
            
            for similar_movie_id, similarity in similar_movies:
                if similar_movie_id not in seen_items:
                    # Weight similarity by user's rating for the source movie
                    weighted_score = similarity * (user_rating / 5.0)
                    
                    if similar_movie_id in candidate_movies:
                        candidate_movies[similar_movie_id] += weighted_score
                    else:
                        candidate_movies[similar_movie_id] = weighted_score
        
        # Sort candidates by score and return top N
        if not candidate_movies:
            return self._recommend_popular_movies(n_recommendations)
        
        recommendations = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _recommend_popular_movies(self, n_recommendations: int) -> List[Tuple[int, float]]:
        """Fallback: recommend popular movies for new users"""
        movie_popularity = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        movie_popularity.columns = ['rating_count', 'avg_rating']
        movie_popularity = movie_popularity[movie_popularity['rating_count'] >= 50]
        movie_popularity['popularity_score'] = (
            movie_popularity['avg_rating'] * np.log(movie_popularity['rating_count'])
        )
        
        popular_movies = movie_popularity.sort_values('popularity_score', ascending=False)
        
        recommendations = []
        for movie_id, row in popular_movies.head(n_recommendations).iterrows():
            recommendations.append((movie_id, row['popularity_score']))
        
        return recommendations
