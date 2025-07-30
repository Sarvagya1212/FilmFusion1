class PopularityBaseline:
    """Simple popularity-based baseline"""
    
    def __init__(self):
        self.popular_movies = None
        
    def fit(self, train_df):
        # Calculate movie popularity
        movie_popularity = train_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        movie_popularity.columns = ['count', 'avg_rating']
        
        # Popularity score: count * avg_rating
        movie_popularity['popularity_score'] = (
            movie_popularity['count'] * movie_popularity['avg_rating']
        )
        
        self.popular_movies = movie_popularity.sort_values(
            'popularity_score', ascending=False
        ).index.tolist()
    
    def recommend(self, user_id, n_recommendations=10, exclude_seen=True):
        if self.popular_movies is None:
            return []
        
        # Return top popular movies
        recommendations = []
        for movie_id in self.popular_movies[:n_recommendations]:
            recommendations.append((movie_id, 1.0))  # Dummy score
        
        return recommendations
