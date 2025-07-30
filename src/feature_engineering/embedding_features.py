import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from gensim.models import Word2Vec
from collections import defaultdict
import logging
import pickle

class EmbeddingFeatureExtractor:
    """Create embeddings for actors, directors, and other categorical features"""
    
    def __init__(self, embedding_dim=100, min_count=2):
        self.logger = logging.getLogger(__name__)
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        
        # Initialize encoders
        self.actor_encoder = MultiLabelBinarizer()
        self.director_encoder = MultiLabelBinarizer()
        self.genre_encoder = MultiLabelBinarizer()
        
        # Word2Vec models for embeddings
        self.actor_w2v = None
        self.director_w2v = None
        
    def prepare_sequences(self, movies_df):
        """Prepare sequences for Word2Vec training"""
        # Actor sequences (movies as documents, actors as words)
        actor_sequences = []
        director_sequences = []
        
        for idx, row in movies_df.iterrows():
            # Process cast
            if isinstance(row['cast'], str) and row['cast'] != '[]':
                try:
                    cast = eval(row['cast'])  # Convert string representation to list
                    if isinstance(cast, list) and len(cast) > 0:
                        actor_sequences.append(cast)
                except:
                    pass
            
            # Process directors
            if isinstance(row['directors'], str) and row['directors'] != '[]':
                try:
                    directors = eval(row['directors'])
                    if isinstance(directors, list) and len(directors) > 0:
                        director_sequences.append(directors)
                except:
                    pass
        
        return actor_sequences, director_sequences
    
    def train_embeddings(self, movies_df):
        """Train Word2Vec embeddings for actors and directors"""
        self.logger.info("Training actor and director embeddings...")
        
        actor_sequences, director_sequences = self.prepare_sequences(movies_df)
        
        # Train actor embeddings
        if actor_sequences:
            self.actor_w2v = Word2Vec(
                sentences=actor_sequences,
                vector_size=self.embedding_dim,
                window=5,
                min_count=self.min_count,
                workers=4,
                epochs=10
            )
            self.logger.info(f"Trained actor embeddings: {len(self.actor_w2v.wv)} actors")
        
        # Train director embeddings
        if director_sequences:
            self.director_w2v = Word2Vec(
                sentences=director_sequences,
                vector_size=self.embedding_dim,
                window=3,
                min_count=1,  # Directors appear less frequently
                workers=4,
                epochs=10
            )
            self.logger.info(f"Trained director embeddings: {len(self.director_w2v.wv)} directors")
    
    def get_movie_embeddings(self, movies_df):
        """Get aggregated embeddings for each movie"""
        movie_embeddings = []
        
        for idx, row in movies_df.iterrows():
            movie_embedding = np.zeros(self.embedding_dim * 2)  # Actor + Director embeddings
            
            # Actor embeddings
            actor_emb = np.zeros(self.embedding_dim)
            if isinstance(row['cast'], str) and row['cast'] != '[]':
                try:
                    cast = eval(row['cast'])
                    if isinstance(cast, list):
                        actor_vectors = []
                        for actor in cast[:5]:  # Top 5 actors
                            if self.actor_w2v and actor in self.actor_w2v.wv:
                                actor_vectors.append(self.actor_w2v.wv[actor])
                        
                        if actor_vectors:
                            actor_emb = np.mean(actor_vectors, axis=0)
                except:
                    pass
            
            # Director embeddings
            director_emb = np.zeros(self.embedding_dim)
            if isinstance(row['directors'], str) and row['directors'] != '[]':
                try:
                    directors = eval(row['directors'])
                    if isinstance(directors, list):
                        director_vectors = []
                        for director in directors:
                            if self.director_w2v and director in self.director_w2v.wv:
                                director_vectors.append(self.director_w2v.wv[director])
                        
                        if director_vectors:
                            director_emb = np.mean(director_vectors, axis=0)
                except:
                    pass
            
            # Combine embeddings
            movie_embedding[:self.embedding_dim] = actor_emb
            movie_embedding[self.embedding_dim:] = director_emb
            movie_embeddings.append(movie_embedding)
        
        return np.array(movie_embeddings)
    
    def create_categorical_features(self, movies_df):
        """Create one-hot encoded features for categorical variables"""
        features = {}
        
        # Genre features
        genre_lists = []
        for idx, row in movies_df.iterrows():
            if 'genres_tmdb' in row and isinstance(row['genres_tmdb'], str):
                try:
                    genres = eval(row['genres_tmdb'])
                    genre_lists.append(genres if isinstance(genres, list) else [])
                except:
                    genre_lists.append([])
            else:
                genre_lists.append([])
        
        if genre_lists:
            genre_features = self.genre_encoder.fit_transform(genre_lists)
            features['genres'] = genre_features
            self.logger.info(f"Created genre features: {genre_features.shape}")
        
        return features
    
    def save_embeddings(self, path_prefix):
        """Save trained embeddings"""
        if self.actor_w2v:
            self.actor_w2v.save(f"{path_prefix}_actor_w2v.model")
        
        if self.director_w2v:
            self.director_w2v.save(f"{path_prefix}_director_w2v.model")
        
        # Save encoders
        with open(f"{path_prefix}_encoders.pkl", 'wb') as f:
            pickle.dump({
                'genre_encoder': self.genre_encoder,
                'actor_encoder': self.actor_encoder,
                'director_encoder': self.director_encoder
            }, f)
        
        self.logger.info(f"Embeddings and encoders saved with prefix: {path_prefix}")
