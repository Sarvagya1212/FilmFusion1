import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
import logging

class RecommendationMetrics:
    """Comprehensive evaluation metrics for recommendation systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def precision_at_k(self, recommended_items: List[int], 
                      relevant_items: Set[int], k: int = 10) -> float:
        """Calculate Precision@K"""
        if k <= 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & relevant_items)
        
        return relevant_recommended / min(k, len(recommended_k))
    
    def recall_at_k(self, recommended_items: List[int], 
                   relevant_items: Set[int], k: int = 10) -> float:
        """Calculate Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & relevant_items)
        
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(self, recommended_items: List[int], 
               relevant_items: Set[int], k: int = 10) -> float:
        """Calculate F1@K"""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended_items: List[int], 
                 relevant_items: Dict[int, float], k: int = 10) -> float:
        """Calculate NDCG@K"""
        recommended_k = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended_k):
            if item_id in relevant_items:
                relevance = relevant_items[item_id]
                dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # Calculate IDCG
        sorted_relevance = sorted(relevant_items.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance):
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def diversity_score(self, recommended_items: List[int], 
                       similarity_matrix: np.ndarray, 
                       item_mapping: Dict[int, int]) -> float:
        """Calculate diversity score (1 - average pairwise similarity)"""
        if len(recommended_items) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                item1_idx = item_mapping.get(recommended_items[i])
                item2_idx = item_mapping.get(recommended_items[j])
                
                if item1_idx is not None and item2_idx is not None:
                    sim = similarity_matrix[item1_idx, item2_idx]
                    similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def novelty_score(self, recommended_items: List[int], 
                     item_popularity: Dict[int, float]) -> float:
        """Calculate novelty score (1 - average item popularity)"""
        if not recommended_items:
            return 0.0
        
        popularities = []
        for item_id in recommended_items:
            if item_id in item_popularity:
                popularities.append(item_popularity[item_id])
        
        if not popularities:
            return 0.0
        
        avg_popularity = np.mean(popularities)
        return 1.0 - avg_popularity
    
    def coverage_score(self, all_recommendations: List[List[int]], 
                      total_items: int) -> float:
        """Calculate catalog coverage"""
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items
    
    def evaluate_recommendations(self, recommendations: Dict[int, List[Tuple[int, float]]], 
                               test_ratings: pd.DataFrame,
                               similarity_matrix: np.ndarray = None,
                               item_mapping: Dict[int, int] = None,
                               item_popularity: Dict[int, float] = None,
                               k_values: List[int] = [5, 10, 20]) -> Dict:
        """Comprehensive evaluation of recommendations"""
        
        metrics_results = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'f1': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'diversity': {k: [] for k in k_values},
            'novelty': {k: [] for k in k_values}
        }
        
        # Calculate per-user metrics
        for user_id, user_recs in recommendations.items():
            # Get user's test items (relevant items)
            user_test_ratings = test_ratings[test_ratings['userId'] == user_id]
            
            if len(user_test_ratings) == 0:
                continue
            
            # Binary relevance (rating >= 4.0)
            relevant_items_binary = set(
                user_test_ratings[user_test_ratings['rating'] >= 4.0]['movieId']
            )
            
            # Graded relevance for NDCG
            relevant_items_graded = {}
            for _, row in user_test_ratings.iterrows():
                # Normalize ratings to 0-3 scale for NDCG
                relevant_items_graded[row['movieId']] = max(0, row['rating'] - 2) / 3
            
            # Extract recommended item IDs
            recommended_item_ids = [item_id for item_id, _ in user_recs]
            
            # Calculate metrics for different k values
            for k in k_values:
                # Precision, Recall, F1
                precision = self.precision_at_k(recommended_item_ids, relevant_items_binary, k)
                recall = self.recall_at_k(recommended_item_ids, relevant_items_binary, k)
                f1 = self.f1_at_k(recommended_item_ids, relevant_items_binary, k)
                
                metrics_results['precision'][k].append(precision)
                metrics_results['recall'][k].append(recall)
                metrics_results['f1'][k].append(f1)
                
                # NDCG
                ndcg = self.ndcg_at_k(recommended_item_ids, relevant_items_graded, k)
                metrics_results['ndcg'][k].append(ndcg)
                
                # Diversity
                if similarity_matrix is not None and item_mapping is not None:
                    diversity = self.diversity_score(
                        recommended_item_ids[:k], similarity_matrix, item_mapping
                    )
                    metrics_results['diversity'][k].append(diversity)
                
                # Novelty
                if item_popularity is not None:
                    novelty = self.novelty_score(recommended_item_ids[:k], item_popularity)
                    metrics_results['novelty'][k].append(novelty)
        
        # Calculate averages
        final_metrics = {}
        for metric_name, k_results in metrics_results.items():
            final_metrics[metric_name] = {}
            for k, values in k_results.items():
                if values:
                    final_metrics[metric_name][k] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
                else:
                    final_metrics[metric_name][k] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'count': 0
                    }
        
        # Calculate coverage
        all_recs = [recs for recs in recommendations.values()]
        if all_recs:
            total_items = len(set(test_ratings['movieId']))
            coverage = self.coverage_score(
                [[item_id for item_id, _ in recs] for recs in all_recs], 
                total_items
            )
            final_metrics['coverage'] = coverage
        
        return final_metrics
