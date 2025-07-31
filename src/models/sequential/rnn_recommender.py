import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

class SessionDataset(Dataset):
    """Dataset for session-based recommendation"""
    
    def __init__(self, sessions: List[List[int]], seq_length: int = 10):
        self.sessions = sessions
        self.seq_length = seq_length
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """Create input-target pairs from sessions"""
        samples = []
        
        for session in self.sessions:
            if len(session) < 2:
                continue
                
            for i in range(1, len(session)):
                # Input sequence
                start_idx = max(0, i - self.seq_length)
                input_seq = session[start_idx:i]
                
                # Pad if necessary
                if len(input_seq) < self.seq_length:
                    input_seq = [0] * (self.seq_length - len(input_seq)) + input_seq
                
                # Target item
                target = session[i]
                
                samples.append((input_seq, target))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class RNNRecommenderModel(nn.Module):
    """RNN-based sequential recommendation model"""
    
    def __init__(self, n_items: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.2):
        super(RNNRecommenderModel, self).__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embedding layer
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # RNN layers
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, n_items)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
    
    def forward(self, input_seq, hidden=None):
        # Embedding
        embedded = self.item_embedding(input_seq)
        
        # RNN
        rnn_output, hidden = self.rnn(embedded, hidden)
        
        # Take the last output
        last_output = rnn_output[:, -1, :]
        
        # Dropout and final layer
        output = self.dropout(last_output)
        logits = self.output_layer(output)
        
        return logits, hidden

class RNNSequentialRecommender:
    """RNN-based sequential recommender for session-based recommendations"""
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, 
                 n_layers: int = 2, seq_length: int = 10, learning_rate: float = 0.001,
                 batch_size: int = 256, epochs: int = 50):
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.logger = logging.getLogger(__name__)
        
    def _prepare_sessions(self, ratings_df: pd.DataFrame) -> List[List[int]]:
        """Prepare session data from ratings"""
        
        # Group by user and sort by timestamp
        if 'timestamp' in ratings_df.columns:
            ratings_df = ratings_df.sort_values(['userId', 'timestamp'])
        else:
            ratings_df = ratings_df.sort_values(['userId'])
        
        sessions = []
        
        # Create sessions per user
        for user_id, user_data in ratings_df.groupby('userId'):
            # Convert to item sequence (only items rated >= 3.5)
            user_items = user_data[user_data['rating'] >= 3.5]['movieId'].tolist()
            
            if len(user_items) >= 2:  # Need at least 2 items for a session
                sessions.append(user_items)
        
        self.logger.info(f"Created {len(sessions)} sessions")
        return sessions
    
    def _create_item_mapping(self, ratings_df: pd.DataFrame):
        """Create item ID mappings"""
        unique_items = ratings_df['movieId'].unique()
        self.item_mapping = {item_id: idx + 1 for idx, item_id in enumerate(unique_items)}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        self.n_items = len(unique_items)
    
    def fit(self, ratings_df: pd.DataFrame):
        """Train the RNN sequential recommender"""
        
        self.logger.info("Training RNN Sequential Recommender...")
        
        # Create item mappings
        self._create_item_mapping(ratings_df)
        
        # Prepare sessions
        sessions = self._prepare_sessions(ratings_df)
        
        # Map item IDs to indices
        mapped_sessions = []
        for session in sessions:
            mapped_session = [self.item_mapping[item_id] for item_id in session 
                            if item_id in self.item_mapping]
            if len(mapped_session) >= 2:
                mapped_sessions.append(mapped_session)
        
        # Create dataset and dataloader
        dataset = SessionDataset(mapped_sessions, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = RNNRecommenderModel(
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            batch_count = 0
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for batch_input, batch_target in pbar:
                    batch_input = batch_input.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    logits, _ = self.model(batch_input)
                    loss = criterion(logits, batch_target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / batch_count
            self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        self.logger.info("RNN Sequential Recommender training completed")
        return self
    
    def predict_next_items(self, session: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Predict next items for a given session"""
        
        if not self.model:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Map session items to indices
        mapped_session = [self.item_mapping.get(item_id, 0) for item_id in session]
        
        # Prepare input sequence
        if len(mapped_session) > self.seq_length:
            input_seq = mapped_session[-self.seq_length:]
        else:
            input_seq = [0] * (self.seq_length - len(mapped_session)) + mapped_session
        
        # Convert to tensor
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, n_recommendations + len(session))
            
            recommendations = []
            seen_items = set(session)
            
            for i in range(len(top_indices[0])):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                
                if idx in self.reverse_item_mapping:
                    item_id = self.reverse_item_mapping[idx]
                    if item_id not in seen_items:  # Exclude already seen items
                        recommendations.append((item_id, prob))
                        
                        if len(recommendations) >= n_recommendations:
                            break
        
        return recommendations
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user (compatibility with other models)"""
        
        # This is a simplified implementation - in practice, you'd need user session data
        # For now, return empty recommendations as this model is session-based
        return []
