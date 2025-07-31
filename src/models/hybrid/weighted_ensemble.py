import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

class WeightedHybridRecommender:
    """
    Weighted ensemble hybrid recommender combining multiple recommendation models
    """
    
    def __init__(self, models: Dict[str, object], optimization_method='grid_search'):
        self.models = models
        self.weights = {name: 1.0/len(models) for name in models.keys()}
        self.optimization_method = optimization_method
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
        
    def optimize_weights(self, validation_df: pd.DataFrame, k_values=[5, 10, 20]):
        """Optimize weights using validation data"""
        
        if self.optimization_method == 'grid_search':
            return self._grid_search_optimization(validation_df, k_values)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(validation_df, k_values)
        else:
            return self._equal_weights()
    
    def _grid_search_optimization(self, validation_df: pd.DataFrame, k_values: List[int]):
        """Grid search for optimal weights"""
        
        model_names = list(self.models.keys())
        best_weights = None
        best_score = -np.inf
        
        # Generate weight combinations
        weight_combinations = []
        for w1 in np.arange(0.1, 1.0, 0.1):
            for w2 in np.arange(0.1, 1.0 - w1, 0.1):
                if len(model_names) == 2:
                    weight_combinations.append([w1, w2])
                elif len(model_names) == 3:
                    w3 = 1.0 - w1 - w2
                    if w3 > 0:
                        weight_combinations.append([w1, w2, w3])
        
        self.logger.info(f"Testing {len(weight_combinations)} weight combinations")
        
        for weights in weight_combinations:
            # Create weight dictionary
            weight_dict = {name: weights[i] for i, name in enumerate(model_names)}
            
            # Evaluate with these weights
            score = self._evaluate_weights(weight_dict, validation_df, k_values)
            
            if score > best_score:
                best_score = score
                best_weights = weight_dict
        
        self.weights = best_weights
        self.logger.info(f"Optimal weights: {self.weights}, Score: {best_score:.4f}")
        
        return best_weights
    
    def _evaluate_weights(self, weights: Dict[str, float], 
                         validation_df: pd.DataFrame, k_values: List[int]) -> float:
        """Evaluate a specific weight configuration"""
        
        total_score = 0
        total_users = 0
        
        # Sample users for efficiency
        sample_users = validation_df['userId'].unique()[:100]
        
        for user_id in sample_users:
            user_test = validation_df[validation_df['userId'] == user_id]
            if len(user_test) == 0:
                continue
            
            # Get hybrid recommendations
            hybrid_recs = self._get_hybrid_recommendations(user_id, weights, n_recommendations=20)
            
            if not hybrid_recs:
                continue
            
            # Calculate precision@k for different k values
            relevant_items = set(user_test[user_test['rating'] >= 4.0]['movieId'])
            
            for k in k_values:
                recommended_items = [item_id for item_id, _ in hybrid_recs[:k]]
                precision = len(set(recommended_items) & relevant_items) / k
                total_score += precision
                
            total_users += 1
        
        return total_score / (total_users * len(k_values)) if total_users > 0 else 0
    
    def _get_hybrid_recommendations(self, user_id: int, weights: Dict[str, float], 
                                   n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations using specified weights"""
        
        model_recommendations = {}
        
        # Get recommendations from each model
        for model_name, model in self.models.items():
            try:
                recs = model.recommend(user_id, n_recommendations * 2)  # Get more to combine
                model_recommendations[model_name] = recs
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed for user {user_id}: {e}")
                model_recommendations[model_name] = []
        
        # Combine recommendations using weights
        combined_scores = {}
        
        for model_name, recs in model_recommendations.items():
            weight = weights.get(model_name, 0)
            
            for movie_id, score in recs:
                if movie_id in combined_scores:
                    combined_scores[movie_id] += weight * score
                else:
                    combined_scores[movie_id] = weight * score
        
        # Sort by combined score
        sorted_recommendations = sorted(combined_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return sorted_recommendations[:n_recommendations]
    
    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, 
        tfidf_matrix=None, movies_df=None):
        """Train the hybrid recommender with proper TF-IDF support"""
        
        self.logger.info("Training individual models...")
        
        # Train all component models
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")
            try:
                if name == 'tfidf' and hasattr(model, 'fit'):
                    # TF-IDF model needs special handling
                    if tfidf_matrix is not None and movies_df is not None:
                        model.fit(
                            ratings_df=train_df,
                            tfidf_matrix=tfidf_matrix,
                            movies_df=movies_df
                        )
                    else:
                        self.logger.error("TF-IDF model requires tfidf_matrix and movies_df")
                        raise ValueError("TF-IDF model requires tfidf_matrix and movies_df")
                else:
                    # Standard model training (SVD++, etc.)
                    model.fit(train_df)
                    
                self.logger.info(f"{name} model trained successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name} model: {e}")
                raise
        
        # Optimize weights
        self.logger.info("Optimizing ensemble weights...")
        self.optimize_weights(validation_df)
        
        self.is_fitted = True
        self.logger.info("Hybrid recommender training completed")
        
        return self

    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate hybrid recommendations"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        return self._get_hybrid_recommendations(user_id, self.weights, n_recommendations)
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating using weighted ensemble"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        weighted_prediction = 0
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                prediction = model.predict(user_id, item_id)
                weight = self.weights[name]
                weighted_prediction += weight * prediction
                total_weight += weight
            except Exception as e:
                self.logger.warning(f"Prediction failed for model {name}: {e}")
                continue
        
        return weighted_prediction / total_weight if total_weight > 0 else 3.0
