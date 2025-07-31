import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVDpp, accuracy
from surprise.model_selection import train_test_split
from typing import List, Tuple
import pickle
from pathlib import Path
import logging

from ..base_recommender import BaseRecommender

class MatrixFactorizationRecommender(BaseRecommender):
    """SVD++ Matrix Factorization using Surprise library with save/load functionality"""
    
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        super().__init__("SVD++ Matrix Factorization")
        
        # SVD++ parameters
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        
        # Initialize SVD++ model
        self.model = SVDpp(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        
        # Store training data for recommendations
        self.trainset = None
        self.ratings_df = None
    
    def fit(self, ratings_df: pd.DataFrame, **kwargs):
        """Train SVD++ model"""
        self.logger.info(f"Training {self.name}...")
        
        # Store ratings for later use
        self.ratings_df = ratings_df.copy()
        
        # Create surprise dataset
        reader = Reader(rating_scale=(0.5, 5.0))
        dataset = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Build full trainset
        self.trainset = dataset.build_full_trainset()
        
        # Create mappings
        self.create_mappings(ratings_df)
        
        # Train model
        self.model.fit(self.trainset)
        
        self.is_fitted = True
        self.logger.info(f"{self.name} training completed")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        prediction = self.model.predict(user_id, item_id)
        return prediction.est
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get all items
        all_items = set(self.ratings_df['movieId'].unique())
        
        # Get items already seen by user
        seen_items = set()
        if exclude_seen:
            seen_items = self.get_user_seen_items(user_id, self.ratings_df)
        
        # Get candidate items
        candidate_items = all_items - seen_items
        
        # Predict ratings for all candidate items
        predictions = []
        for item_id in candidate_items:
            try:
                predicted_rating = self.predict(user_id, item_id)
                predictions.append((item_id, predicted_rating))
            except:
                continue
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def evaluate_on_testset(self, testset):
        """Evaluate model on test set"""
        predictions = self.model.test(testset)
        
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions
        }
    
    def save_model(self, path: str):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'trainset': self.trainset,
            'user_mapping': getattr(self, 'user_mapping', {}),
            'item_mapping': getattr(self, 'item_mapping', {}),
            'reverse_user_mapping': getattr(self, 'reverse_user_mapping', {}),
            'reverse_item_mapping': getattr(self, 'reverse_item_mapping', {}),
            'ratings_df': self.ratings_df,
            'parameters': {
                'n_factors': self.n_factors,
                'n_epochs': self.n_epochs,
                'lr_all': self.lr_all,
                'reg_all': self.reg_all
            },
            'is_fitted': self.is_fitted,
            'n_users': getattr(self, 'n_users', 0),
            'n_items': getattr(self, 'n_items', 0)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.trainset = model_data['trainset']
            self.user_mapping = model_data.get('user_mapping', {})
            self.item_mapping = model_data.get('item_mapping', {})
            self.reverse_user_mapping = model_data.get('reverse_user_mapping', {})
            self.reverse_item_mapping = model_data.get('reverse_item_mapping', {})
            self.ratings_df = model_data.get('ratings_df')
            self.n_users = model_data.get('n_users', len(self.user_mapping))
            self.n_items = model_data.get('n_items', len(self.item_mapping))
            
            # Restore parameters
            params = model_data.get('parameters', {})
            self.n_factors = params.get('n_factors', self.n_factors)
            self.n_epochs = params.get('n_epochs', self.n_epochs)
            self.lr_all = params.get('lr_all', self.lr_all)
            self.reg_all = params.get('reg_all', self.reg_all)
            
            self.is_fitted = model_data.get('is_fitted', True)
            
            self.logger.info(f"Model loaded from {path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
