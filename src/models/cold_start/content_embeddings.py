import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import logging

class ContentEmbeddingRecommender:
    """Content-based recommendations using embeddings for cold-start items"""
    
    def __init__(self, similarity_threshold: float = 0.1):
        self.similarity_threshold = similarity_threshold
        self.content_embeddings = None
        self.movie_features = None
        self.genre_encoder = MultiLabelBinarizer()
        self.cast_encoder = MultiLabelBinarizer()
        self.director_encoder = MultiLabelBinarizer()
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        
    def _extract_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        """Extract content features from movies"""
        
        features = []
        
        # Genre features
        genre_lists = []
        for _, movie in movies_df.iterrows():
            if 'genres_tmdb' in movie and isinstance(movie['genres_tmdb'], str):
                try:
                    genres = eval(movie['genres_tmdb'])
                    if isinstance(genres, list):
                        genre_lists.append(genres)
                    else:
                        genre_lists.append([])
                except:
                    genre_lists.append([])
            else:
                genre_lists.append([])
        
        genre_features = self.genre_encoder.fit_transform(genre_lists)
        features.append(genre_features)
        
        # Cast features (top 5 actors)
        cast_lists = []
        for _, movie in movies_df.iterrows():
            if 'cast' in movie and isinstance(movie['cast'], str):
                try:
                    cast = eval(movie['cast'])
                    if isinstance(cast, list):
                        cast_lists.append(cast[:5])  # Top 5 actors
                    else:
                        cast_lists.append([])
                except:
                    cast_lists.append([])
            else:
                cast_lists.append([])
        
        cast_features = self.cast_encoder.fit_transform(cast_lists)
        features.append(cast_features)
        
        # Director features
        director_lists = []
        for _, movie in movies_df.iterrows():
            if 'directors' in movie and isinstance(movie['directors'], str):
                try:
                    directors = eval(movie['directors'])
                    if isinstance(directors, list):
                        director_lists.append(directors)
                    else:
                        director_lists.append([])
                except:
                    director_lists.append([])
            else:
                director_lists.append([])
        
        director_features = self.director_encoder.fit_transform(director_lists)
        features.append(director_features)
        
        # Numerical features (normalized)
        numerical_features = []
        
        # Year feature (normalized)
        if 'year' in movies_df.columns:
            years = movies_df['year'].fillna(movies_df['year'].median())
            year_normalized = (years - years.min()) / (years.max() - years.min())
            numerical_features.append(year_normalized.values.reshape(-1, 1))
        
        # Rating features (if available)
        if 'vote_average' in movies_df.columns:
            ratings = movies_df['vote_average'].fillna(movies_df['vote_average'].median())
            rating_normalized = ratings / 10.0
            numerical_features.append(rating_normalized.values.reshape(-1, 1))
        
        # Runtime features (if available)
        if 'runtime' in movies_df.columns:
            runtime = movies_df['runtime'].fillna(movies_df['runtime'].median())
            runtime_normalized = runtime / runtime.max()
            numerical_features.append(runtime_normalized.values.reshape(-1, 1))
        
        if numerical_features:
            numerical_matrix = np.hstack(numerical_features)
            features.append(numerical_matrix)
        
        # Combine all features
        combined_features = np.hstack(features)
        
        self.logger.info(f"Extracted features shape: {combined_features.shape}")
        return combined_features
    
    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame = None):
        """Train the content embedding recommender"""
        
        self.logger.info("Training content embedding recommender...")
        
        self.movie_features = movies_df.copy()
        
        # Extract content features
        self.content_embeddings = self._extract_features(movies_df)
        
        # Create movie ID to index mapping
        self.movie_id_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])
        }
        self.idx_to_movie_id = {
            idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()
        }
        
        self.is_fitted = True
        self.logger.info("✅ Content embedding recommender trained")
        
        return self
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar movies for a given movie"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar movies")
        
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        movie_embedding = self.content_embeddings[movie_idx].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(movie_embedding, self.content_embeddings)[0]
        
        # Get most similar movies (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        similar_movies = []
        for idx in similar_indices:
            if similarities[idx] > self.similarity_threshold:
                similar_movie_id = self.idx_to_movie_id[idx]
                similar_movies.append((similar_movie_id, similarities[idx]))
        
        return similar_movies
    
    def recommend_for_new_movie(self, new_movie_features: Dict, 
                               user_preferences: Dict, 
                               n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Recommend similar movies for a new movie"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Create feature vector for new movie
        new_movie_embedding = self._create_new_movie_embedding(new_movie_features)
        
        # Calculate similarities with existing movies
        similarities = cosine_similarity(
            new_movie_embedding.reshape(1, -1), 
            self.content_embeddings
        )[0]
        
        # Get most similar movies
        similar_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in similar_indices:
            if similarities[idx] > self.similarity_threshold:
                movie_id = self.idx_to_movie_id[idx]
                recommendations.append((movie_id, similarities[idx]))
        
        return recommendations
    
    def _create_new_movie_embedding(self, movie_features: Dict) -> np.ndarray:
        """Create embedding for a new movie"""
        
        # Extract and encode features similar to training
        features = []
        
        # Genre features
        genres = movie_features.get('genres', [])
        genre_features = self.genre_encoder.transform([genres])
        features.append(genre_features)
        
        # Cast features
        cast = movie_features.get('cast', [])[:5]  # Top 5 actors
        cast_features = self.cast_encoder.transform([cast])
        features.append(cast_features)
        
        # Director features
        directors = movie_features.get('directors', [])
        director_features = self.director_encoder.transform([directors])
        features.append(director_features)
        
        # Numerical features
        numerical_features = []
        
        # Year (normalized using training data statistics)
        year = movie_features.get('year', 2020)
        year_stats = self.movie_features['year'].agg(['min', 'max'])
        year_normalized = (year - year_stats['min']) / (year_stats['max'] - year_stats['min'])
        numerical_features.append([year_normalized])
        
        # Rating (if provided)
        if 'vote_average' in movie_features:
            rating_normalized = movie_features['vote_average'] / 10.0
            numerical_features.append([rating_normalized])
        
        # Runtime (if provided)
        if 'runtime' in movie_features:
            runtime_stats = self.movie_features['runtime'].max()
            runtime_normalized = movie_features['runtime'] / runtime_stats
            numerical_features.append([runtime_normalized])
        
        if numerical_features:
            numerical_matrix = np.array(numerical_features).reshape(1, -1)
            features.append(numerical_matrix)
        
        # Combine features
        combined_features = np.hstack(features)
        return combined_features.flatten()
