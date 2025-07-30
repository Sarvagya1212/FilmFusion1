import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Create metadata-based content recommender
print("🏷️ BUILDING METADATA-BASED CONTENT MODEL")
print("=" * 40)

class MetadataContentRecommender:
    """Content-based recommender using movie metadata"""
    
    def __init__(self):
        self.similarity_matrix = None
        self.movies_df = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        
    def create_metadata_features(self, movies_df):
        """Create feature matrix from movie metadata"""
        features = []
        
        # Genre features (one-hot encoding)
        all_genres = set()
        for _, row in movies_df.iterrows():
            if isinstance(row['genres_tmdb'], str) and row['genres_tmdb'] != '[]':
                try:
                    genres = eval(row['genres_tmdb'])
                    if isinstance(genres, list):
                        all_genres.update(genres)
                except:
                    pass
        
        all_genres = sorted(list(all_genres))
        print(f"Found {len(all_genres)} unique genres: {all_genres[:10]}...")
        
        # Create genre matrix
        genre_matrix = np.zeros((len(movies_df), len(all_genres)))
        for i, (_, row) in enumerate(movies_df.iterrows()):
            if isinstance(row['genres_tmdb'], str) and row['genres_tmdb'] != '[]':
                try:
                    genres = eval(row['genres_tmdb'])
                    if isinstance(genres, list):
                        for genre in genres:
                            if genre in all_genres:
                                genre_idx = all_genres.index(genre)
                                genre_matrix[i, genre_idx] = 1
                except:
                    pass
        
        features.append(genre_matrix)
        feature_names = [f"genre_{g}" for g in all_genres]
        
        # Year features (normalized)
        if 'year' in movies_df.columns:
            years = movies_df['year'].fillna(movies_df['year'].median()).values
            years_normalized = (years - years.min()) / (years.max() - years.min())
            features.append(years_normalized.reshape(-1, 1))
            feature_names.append('year_normalized')
        
        # Rating features
        if 'vote_average' in movies_df.columns:
            ratings = movies_df['vote_average'].fillna(movies_df['vote_average'].median()).values
            ratings_normalized = ratings / 10.0  # Normalize to 0-1
            features.append(ratings_normalized.reshape(-1, 1))
            feature_names.append('vote_average_normalized')
        
        # Runtime features
        if 'runtime' in movies_df.columns:
            runtime = movies_df['runtime'].fillna(movies_df['runtime'].median()).values
            runtime_normalized = runtime / runtime.max()
            features.append(runtime_normalized.reshape(-1, 1))
            feature_names.append('runtime_normalized')
        
        # Combine all features
        feature_matrix = np.hstack(features)
        
        print(f"Created metadata feature matrix: {feature_matrix.shape}")
        print(f"Features: {len(feature_names)} total")
        
        return feature_matrix, feature_names
    
    def fit(self, movies_df, ratings_df):
        """Train metadata-based model"""
        self.movies_df = movies_df.copy()
        
        # Create movie mappings
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies_df['movieId'])}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Create metadata features
        feature_matrix, self.feature_names = self.create_metadata_features(movies_df)
        
        # Compute similarity matrix
        print("Computing metadata similarity matrix...")
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        print(f"Metadata model trained successfully")
        
    def get_similar_movies(self, movie_id, n_similar=10):
        """Get similar movies based on metadata"""
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Get most similar movies
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_similar+1]
        
        similar_movies = []
        for idx in similar_indices:
            similar_movie_id = self.idx_to_movie_id[idx]
            similar_movies.append((similar_movie_id, similarity_scores[idx]))
        
        return similar_movies
    
    def recommend(self, user_id, ratings_df, n_recommendations=10):
        """Generate recommendations based on metadata similarity"""
        # Get user's highly rated movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]
        
        if len(high_rated) == 0:
            return []
        
        # Get similar movies for each highly rated movie
        candidate_movies = {}
        seen_movies = set(user_ratings['movieId'])
        
        for _, row in high_rated.iterrows():
            movie_id = row['movieId']
            user_rating = row['rating']
            
            similar_movies = self.get_similar_movies(movie_id, n_similar=20)
            
            for similar_movie_id, similarity in similar_movies:
                if similar_movie_id not in seen_movies:
                    weighted_score = similarity * (user_rating / 5.0)
                    
                    if similar_movie_id in candidate_movies:
                        candidate_movies[similar_movie_id] += weighted_score
                    else:
                        candidate_movies[similar_movie_id] = weighted_score
        
        # Sort and return top recommendations
        recommendations = sorted(candidate_movies.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]