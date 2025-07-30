# Enhanced Neural CF Model Implementation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import ParameterGrid
import torch
from torch.utils.data import Dataset, DataLoader

class ImprovedNCFModel(nn.Module):
    """Enhanced Neural CF with dropout and better architecture"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[128, 64], dropout=0.2):
        super(ImprovedNCFModel, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers with better initialization
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # MLP layers with dropout
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch normalization
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(input_dim, 1))
        mlp_layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(concat_emb)
        
        return output.squeeze()

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience


class NCFDataset(Dataset):
    """Enhanced Dataset for Neural Collaborative Filtering"""
    
    def __init__(self, ratings_df, user_mapping=None, item_mapping=None):
        self.ratings_df = ratings_df.copy()
        
        # Create mappings if not provided
        if user_mapping is None:
            unique_users = ratings_df['userId'].unique()
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        else:
            self.user_mapping = user_mapping
            
        if item_mapping is None:
            unique_items = ratings_df['movieId'].unique()
            self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        else:
            self.item_mapping = item_mapping
        
        # Prepare data
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare training data"""
        self.samples = []
        
        for _, row in self.ratings_df.iterrows():
            user_idx = self.user_mapping.get(row['userId'])
            item_idx = self.item_mapping.get(row['movieId'])
            
            if user_idx is not None and item_idx is not None:
                # Normalize rating to 0-1 scale for sigmoid output
                rating_normalized = (row['rating'] - 0.5) / 4.5
                
                self.samples.append({
                    'user': user_idx,
                    'item': item_idx,
                    'rating': rating_normalized,
                    'rating_original': row['rating']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['user'], dtype=torch.long),
            torch.tensor(sample['item'], dtype=torch.long),
            torch.tensor(sample['rating_original'], dtype=torch.float32)  # Use original scale for loss
        )

