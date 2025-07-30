import numpy as np

class RandomBaseline:
    """Random recommendation baseline"""
    
    def __init__(self):
        self.all_movies = None
    
    def fit(self, train_df):
        self.all_movies = train_df['movieId'].unique().tolist()
    
    def recommend(self, user_id, n_recommendations=10, exclude_seen=True):
        if self.all_movies is None:
            return []
        
        # Return random movies
        selected_movies = np.random.choice(
            self.all_movies, 
            size=min(n_recommendations, len(self.all_movies)), 
            replace=False
        )
        
        recommendations = [(movie_id, np.random.random()) for movie_id in selected_movies]
        return recommendations