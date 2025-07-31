import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

class PopularByGenreRecommender:
    """Popular-by-genre fallback for new users"""
    
    def __init__(self, min_ratings: int = 50):
        self.min_ratings = min_ratings
        self.genre_popularity = {}
        self.global_popular_movies = []
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """Train the popularity-based recommender"""
        
        self.logger.info("Training popularity-based recommender...")
        
        # Calculate movie popularity scores
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_stats.columns = ['rating_count', 'avg_rating']
        
        # Filter movies with minimum ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= self.min_ratings]
        
        # Calculate popularity score (weighted by count and average rating)
        popular_movies['popularity_score'] = (
            popular_movies['avg_rating'] * np.log1p(popular_movies['rating_count'])
        )
        
        # Merge with movie metadata
        movies_with_stats = movies_df.merge(popular_movies, on='movieId', how='inner')
        
        # Calculate genre-based popularity
        self._calculate_genre_popularity(movies_with_stats)
        
        # Global popular movies (fallback)
        self.global_popular_movies = popular_movies.sort_values(
            'popularity_score', ascending=False
        ).head(100).index.tolist()
        
        self.is_fitted = True
        self.logger.info("✅ Popularity-based recommender trained")
        
        return self
    
    def _calculate_genre_popularity(self, movies_with_stats: pd.DataFrame):
        """Calculate popularity by genre"""
        
        self.genre_popularity = {}
        
        for _, movie in movies_with_stats.iterrows():
            # Parse genres
            if 'genres_tmdb' in movie and isinstance(movie['genres_tmdb'], str):
                try:
                    genres = eval(movie['genres_tmdb'])
                    if isinstance(genres, list):
                        for genre in genres:
                            if genre not in self.genre_popularity:
                                self.genre_popularity[genre] = []
                            
                            self.genre_popularity[genre].append({
                                'movieId': movie['movieId'],
                                'popularity_score': movie['popularity_score'],
                                'avg_rating': movie['avg_rating'],
                                'rating_count': movie['rating_count']
                            })
                except:
                    pass
        
        # Sort movies within each genre by popularity
        for genre in self.genre_popularity:
            self.genre_popularity[genre] = sorted(
                self.genre_popularity[genre],
                key=lambda x: x['popularity_score'],
                reverse=True
            )
        
        self.logger.info(f"Calculated popularity for {len(self.genre_popularity)} genres")
    
    def recommend_by_genre(self, preferred_genres: List[str], 
                          n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Recommend popular movies by preferred genres"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = []
        seen_movies = set()
        
        # Distribute recommendations across preferred genres
        recs_per_genre = max(1, n_recommendations // len(preferred_genres))
        
        for genre in preferred_genres:
            if genre in self.genre_popularity:
                genre_movies = self.genre_popularity[genre]
                
                for movie_data in genre_movies:
                    movie_id = movie_data['movieId']
                    if movie_id not in seen_movies:
                        recommendations.append((movie_id, movie_data['popularity_score']))
                        seen_movies.add(movie_id)
                        
                        if len([r for r in recommendations if r[0] in [m['movieId'] for m in genre_movies[:recs_per_genre]]]) >= recs_per_genre:
                            break
        
        # Fill remaining slots with global popular movies
        while len(recommendations) < n_recommendations:
            for movie_id in self.global_popular_movies:
                if movie_id not in seen_movies:
                    recommendations.append((movie_id, 1.0))  # Default score
                    seen_movies.add(movie_id)
                    break
            else:
                break  # No more movies to add
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def recommend_for_new_user(self, user_preferences: Dict = None, 
                              n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Recommend for a completely new user"""
        
        if user_preferences and 'preferred_genres' in user_preferences:
            return self.recommend_by_genre(
                user_preferences['preferred_genres'], 
                n_recommendations
            )
        else:
            # Return globally popular movies
            recommendations = []
            for movie_id in self.global_popular_movies[:n_recommendations]:
                recommendations.append((movie_id, 1.0))
            return recommendations
    
    def get_popular_genres(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most popular genres"""
        
        genre_counts = {genre: len(movies) for genre, movies in self.genre_popularity.items()}
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_genres[:top_n]
