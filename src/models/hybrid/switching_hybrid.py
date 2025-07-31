import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class SwitchingHybridRecommender:
    """
    Switching hybrid recommender that selects models based on context
    """
    
    def __init__(self, models: Dict[str, object], switching_strategy='user_profile'):
        self.models = models
        self.switching_strategy = switching_strategy
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        
        # Thresholds for switching decisions
        self.min_ratings_cf = 10  # Minimum ratings for collaborative filtering
        self.min_content_features = 5  # Minimum content features for content-based
        
    def _decide_model(self, user_id: int, context: Dict = None) -> str:
        """Decide which model to use based on switching strategy"""
        
        if self.switching_strategy == 'user_profile':
            return self._user_profile_switching(user_id)
        elif self.switching_strategy == 'item_features':
            return self._item_feature_switching(context)
        elif self.switching_strategy == 'confidence_based':
            return self._confidence_based_switching(user_id, context)
        else:
            return list(self.models.keys())[0]  # Default to first model
    
    def _user_profile_switching(self, user_id: int) -> str:
        """Switch based on user profile characteristics"""
        
        try:
            # Get user's rating history
            user_ratings = self.train_df[self.train_df['userId'] == user_id]
            num_ratings = len(user_ratings)
            
            if num_ratings >= self.min_ratings_cf:
                # User has enough ratings for collaborative filtering
                return 'svd_model'  # Assume SVD is the collaborative model
            else:
                # New user or sparse data - use content-based
                return 'tfidf_model'  # Assume TF-IDF is the content model
                
        except Exception as e:
            self.logger.warning(f"Error in user profile switching: {e}")
            return 'tfidf_model'  # Default to content-based
    
    def _confidence_based_switching(self, user_id: int, context: Dict = None) -> str:
        """Switch based on model confidence scores"""
        
        model_confidences = {}
        
        for name, model in self.models.items():
            try:
                # Get sample recommendations and estimate confidence
                sample_recs = model.recommend(user_id, n_recommendations=5)
                
                if sample_recs:
                    # Use score variance as confidence measure
                    scores = [score for _, score in sample_recs]
                    confidence = np.std(scores)  # Higher variance = more confident
                    model_confidences[name] = confidence
                else:
                    model_confidences[name] = 0
                    
            except Exception as e:
                self.logger.warning(f"Confidence calculation failed for {name}: {e}")
                model_confidences[name] = 0
        
        # Select model with highest confidence
        best_model = max(model_confidences.keys(), key=lambda x: model_confidences[x])
        return best_model
    
    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame = None):
        """Train the switching hybrid recommender"""
        
        self.train_df = train_df  # Store for switching decisions
        
        self.logger.info("Training individual models...")
        
        # Train all component models
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            try:
                model.fit(train_df)
                self.logger.info(f"{name} model trained successfully")
            except Exception as e:
                self.logger.error(f"Failed to train {name} model: {e}")
                raise
        
        self.is_fitted = True
        self.logger.info("Switching hybrid recommender training completed")
        
        return self
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True, context: Dict = None) -> List[Tuple[int, float]]:
        """Generate recommendations using model switching"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Decide which model to use
        selected_model_name = self._decide_model(user_id, context)
        selected_model = self.models[selected_model_name]
        
        self.logger.debug(f"Selected {selected_model_name} for user {user_id}")
        
        try:
            return selected_model.recommend(user_id, n_recommendations, exclude_seen)
        except Exception as e:
            self.logger.warning(f"Selected model {selected_model_name} failed: {e}")
            
            # Fallback to another model
            for name, model in self.models.items():
                if name != selected_model_name:
                    try:
                        return model.recommend(user_id, n_recommendations, exclude_seen)
                    except:
                        continue
            
            return []  # All models failed
    
    def predict(self, user_id: int, item_id: int, context: Dict = None) -> float:
        """Predict rating using selected model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        selected_model_name = self._decide_model(user_id, context)
        selected_model = self.models[selected_model_name]
        
        try:
            return selected_model.predict(user_id, item_id)
        except Exception as e:
            self.logger.warning(f"Prediction failed for {selected_model_name}: {e}")
            return 3.0  # Default prediction
