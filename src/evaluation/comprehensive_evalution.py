import numpy as np
import sys
import pandas as pd
sys.path.append('../')
from src.evaluation.metrics import RecommendationMetrics
from pathlib import Path

enriched_movies = pd.read_csv(Path('../data/enriched') / 'movies_enriched.csv')

class ComprehensiveEvaluator:
    def __init__(self):
        self.metrics_calculator = RecommendationMetrics()
        self.results = {}
    
    def evaluate_model(self, model, model_name, test_df, train_df, 
                      n_recommendations=10, sample_size=200):
        """Comprehensive evaluation of a single model"""
        print(f"\nEvaluating {model_name}...")
        
        # Sample test users for faster evaluation
        test_users = test_df['userId'].unique()
        if len(test_users) > sample_size:
            test_users = np.random.choice(test_users, sample_size, replace=False)
        
        print(f"Evaluating on {len(test_users)} users...")
        
        # Generate recommendations
        recommendations = {}
        successful_users = 0
        failed_users = 0
        
        for user_id in test_users:
            try:
                if hasattr(model, 'recommend'):
                    # Handle different model interfaces
                    if model_name in ["TF-IDF", "Metadata"]:
                        if model_name == "Metadata":
                            recs = model.recommend(user_id, train_df, n_recommendations)
                        else:
                            recs = model.recommend(user_id, n_recommendations, exclude_seen=True)
                    else:
                        recs = model.recommend(user_id, n_recommendations, exclude_seen=True)
                    
                    if recs and len(recs) > 0:
                        recommendations[user_id] = recs
                        successful_users += 1
                    else:
                        failed_users += 1
                else:
                    failed_users += 1
                    
            except Exception as e:
                failed_users += 1
                continue
        
        print(f"Generated recommendations: {successful_users}/{len(test_users)} users")
        
        if successful_users == 0:
            print(f"No recommendations generated for {model_name}")
            return None
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(recommendations, test_df, train_df, n_recommendations)
        
        # Add meta information
        metrics['model_name'] = model_name
        metrics['users_evaluated'] = len(test_users)
        metrics['successful_users'] = successful_users
        metrics['failed_users'] = failed_users
        metrics['success_rate'] = successful_users / len(test_users)
        
        print(f"{model_name} evaluation completed")
        
        return metrics
    
    def calculate_all_metrics(self, recommendations, test_df, train_df, k=10):
        """Calculate all evaluation metrics"""
        
        # Initialize metric collections
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []
        
        # Calculate item popularity for novelty
        item_popularity = train_df.groupby('movieId').size()
        max_popularity = item_popularity.max()
        item_popularity_normalized = item_popularity / max_popularity
        item_popularity_dict = item_popularity_normalized.to_dict()
        
        # Per-user metrics
        for user_id, user_recs in recommendations.items():
            # Get user's test ratings
            user_test = test_df[test_df['userId'] == user_id]
            
            if len(user_test) == 0:
                continue
            
            # Binary relevance (rating >= 4.0)
            relevant_items_binary = set(user_test[user_test['rating'] >= 4.0]['movieId'])
            
            # Graded relevance for NDCG
            relevant_items_graded = {}
            for _, row in user_test.iterrows():
                # Normalize to 0-3 scale
                relevant_items_graded[row['movieId']] = max(0, (row['rating'] - 2) / 3)
            
            # Extract recommended item IDs
            recommended_items = [item_id for item_id, _ in user_recs[:k]]
            
            # Calculate metrics
            if len(relevant_items_binary) > 0:
                precision = self.metrics_calculator.precision_at_k(recommended_items, relevant_items_binary, k)
                recall = self.metrics_calculator.recall_at_k(recommended_items, relevant_items_binary, k)
                f1 = self.metrics_calculator.f1_at_k(recommended_items, relevant_items_binary, k)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
            
            if len(relevant_items_graded) > 0:
                ndcg = self.metrics_calculator.ndcg_at_k(recommended_items, relevant_items_graded, k)
                ndcg_scores.append(ndcg)
        
        # Calculate aggregate metrics
        metrics = {
            'precision_at_k': np.mean(precision_scores) if precision_scores else 0,
            'recall_at_k': np.mean(recall_scores) if recall_scores else 0,
            'f1_at_k': np.mean(f1_scores) if f1_scores else 0,
            'ndcg_at_k': np.mean(ndcg_scores) if ndcg_scores else 0,
            'precision_std': np.std(precision_scores) if precision_scores else 0,
            'recall_std': np.std(recall_scores) if recall_scores else 0,
            'f1_std': np.std(f1_scores) if f1_scores else 0,
            'ndcg_std': np.std(ndcg_scores) if ndcg_scores else 0
        }
        
        # Coverage metrics
        all_recommended_items = set()
        all_recommendations_flat = []
        
        for user_recs in recommendations.values():
            user_items = [item_id for item_id, _ in user_recs]
            all_recommended_items.update(user_items)
            all_recommendations_flat.extend(user_items)
        
        total_items = len(enriched_movies)
        metrics['catalog_coverage'] = len(all_recommended_items) / total_items
        
        # Diversity (average pairwise dissimilarity)
        if len(all_recommendations_flat) > 1:
            # Simple diversity based on recommendation frequency
            item_counts = pd.Series(all_recommendations_flat).value_counts()
            diversity = 1 - (item_counts.max() / len(all_recommendations_flat))
            metrics['diversity'] = diversity
        else:
            metrics['diversity'] = 0
        
        # Novelty (average inverse popularity)
        novelty_scores = []
        for user_recs in recommendations.values():
            user_novelty = []
            for item_id, _ in user_recs:
                if item_id in item_popularity_dict:
                    novelty = 1 - item_popularity_dict[item_id]
                    user_novelty.append(novelty)
            if user_novelty:
                novelty_scores.append(np.mean(user_novelty))
        
        metrics['novelty'] = np.mean(novelty_scores) if novelty_scores else 0
        
        return metrics