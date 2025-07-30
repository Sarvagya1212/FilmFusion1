from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

class BaseRecommender(ABC):
    """Base class for all recommendation models"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
    
    def create_mappings(self, ratings_df: pd.DataFrame):
        """Create user and item ID mappings"""
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['movieId'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        self.logger.info(f"Created mappings: {self.n_users} users, {self.n_items} items")
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        pass
    
    def get_user_seen_items(self, user_id: int, ratings_df: pd.DataFrame) -> set:
        """Get items already rated by user"""
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        return set(user_ratings['movieId'].values)
    
    def batch_recommend(self, user_ids: List[int], n_recommendations: int = 10,
                       exclude_seen: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        """Generate recommendations for multiple users"""
        recommendations = {}
        for user_id in user_ids:
            try:
                recs = self.recommend(user_id, n_recommendations, exclude_seen)
                recommendations[user_id] = recs
            except Exception as e:
                self.logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
                recommendations[user_id] = []
        
        return recommendations
