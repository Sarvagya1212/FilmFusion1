import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import logging
from tqdm import tqdm

from ..base_recommender import BaseRecommender

class NCFDataset(Dataset):
    """Dataset for Neural Collaborative Filtering"""
    
    def __init__(self, ratings_df, user_mapping, item_mapping, negative_sampling=True):
        self.ratings_df = ratings_df
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping
        self.negative_sampling = negative_sampling
        
        # Prepare data
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare training data with negative sampling"""
        positive_samples = []
        
        for _, row in self.ratings_df.iterrows():
            user_idx = self.user_mapping[row['userId']]
            item_idx = self.item_mapping[row['movieId']]
            rating = 1.0 if row['rating'] >= 3.5 else 0.0  # Binary feedback
            
            positive_samples.append([user_idx, item_idx, rating])
        
        self.samples = positive_samples
        
        # Add negative samples
        if self.negative_sampling:
            self.add_negative_samples()
    
    def add_negative_samples(self):
        """Add negative samples for training"""
        user_item_set = set()
        for sample in self.samples:
            user_item_set.add((sample[0], sample[1]))
        
        negative_samples = []
        n_negative = len(self.samples)  # Equal number of negative samples
        
        users = list(self.user_mapping.values())
        items = list(self.item_mapping.values())
        
        while len(negative_samples) < n_negative:
            user_idx = np.random.choice(users)
            item_idx = np.random.choice(items)
            
            if (user_idx, item_idx) not in user_item_set:
                negative_samples.append([user_idx, item_idx, 0.0])
                user_item_set.add((user_idx, item_idx))
        
        self.samples.extend(negative_samples)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'user': torch.tensor(sample[0], dtype=torch.long),
            'item': torch.tensor(sample[1], dtype=torch.long),
            'rating': torch.tensor(sample[2], dtype=torch.float32)
        }

class NCFModel(nn.Module):
    """Neural Collaborative Filtering Model"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64]):
        super(NCFModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        mlp_layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding layers"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(concat_emb)
        
        return output.squeeze()

class NeuralCFRecommender(BaseRecommender):
    """Neural Collaborative Filtering Recommender"""
    
    def __init__(self, embedding_dim=64, hidden_dims=[128, 64], 
                 learning_rate=0.001, batch_size=256, epochs=50):
        super().__init__("Neural Collaborative Filtering")
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Model components
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def fit(self, ratings_df: pd.DataFrame, **kwargs):
        """Train Neural CF model"""
        self.logger.info(f"Training {self.name}...")
        
        # Create mappings
        self.create_mappings(ratings_df)
        self.ratings_df = ratings_df.copy()
        
        # Create dataset and dataloader
        dataset = NCFDataset(ratings_df, self.user_mapping, self.item_mapping)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = NCFModel(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for batch in pbar:
                    users = batch['user'].to(self.device)
                    items = batch['item'].to(self.device)
                    ratings = batch['rating'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    predictions = self.model(users, items)
                    loss = criterion(predictions, ratings)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / batch_count
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        self.logger.info(f"✅ {self.name} training completed")
        
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return 2.5  # Default prediction for unknown users/items
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
            
            prediction = self.model(user_tensor, item_tensor)
            
            # Convert probability to rating scale (0.5 to 5.0)
            rating = 0.5 + (prediction.cpu().item() * 4.5)
        
        return rating
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            return []  # Return empty list for unknown users
        
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
