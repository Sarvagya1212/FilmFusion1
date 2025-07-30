import pandas as pd
import numpy as np
from typing import Dict, List, Set
import logging
import json
from pathlib import Path

class GenreNormalizer:
    """Normalize and standardize genre taxonomies"""
    
    def __init__(self, custom_mapping_file: str = None):
        self.logger = logging.getLogger(__name__)
        self.genre_mapping = self._load_genre_mapping(custom_mapping_file)
        self.genre_stats = {}
    
    def _load_genre_mapping(self, custom_file: str = None) -> Dict[str, str]:
        """Load genre mapping for normalization"""
        
        # Default mapping for common variations and synonyms
        default_mapping = {
            # Case variations
            'sci-fi': 'Science Fiction',
            'sci fi': 'Science Fiction',
            'scifi': 'Science Fiction',
            
            # Common synonyms
            'thriller': 'Thriller',
            'suspense': 'Thriller',
            'action': 'Action',
            'adventure': 'Adventure',
            'comedy': 'Comedy',
            'drama': 'Drama',
            'horror': 'Horror',
            'romance': 'Romance',
            'romantic': 'Romance',
            'documentary': 'Documentary',
            'animation': 'Animation',
            'animated': 'Animation',
            'fantasy': 'Fantasy',
            'mystery': 'Mystery',
            'crime': 'Crime',
            'war': 'War',
            'western': 'Western',
            'musical': 'Musical',
            'music': 'Musical',
            'family': 'Family',
            'children': "Children's",
            "children's": "Children's",
            'kids': "Children's",
            
            # Specific mappings
            'film-noir': 'Film-Noir',
            'noir': 'Film-Noir',
            'imax': 'IMAX',
            '(no genres listed)': 'Unknown'
        }
        
        # Load custom mapping if provided
        if custom_file and Path(custom_file).exists():
            try:
                with open(custom_file, 'r') as f:
                    custom_mapping = json.load(f)
                default_mapping.update(custom_mapping)
                self.logger.info(f"Loaded custom genre mapping from {custom_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load custom mapping: {e}")
        
        return default_mapping
    
    def get_genre_statistics(self) -> Dict:
        """Get genre normalization statistics with JSON serializable types"""
        if not self.genre_stats:
            return {}
        
        # Convert numpy types to native Python types
        serializable_stats = {}
        for key, value in self.genre_stats.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                serializable_stats[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                serializable_stats[key] = float(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries (like most_common_genres)
                serializable_stats[key] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                                        for k, v in value.items()}
            else:
                serializable_stats[key] = value
        
        return serializable_stats

    
    def normalize_genre_name(self, genre: str) -> str:
        """Normalize a single genre name"""
        if pd.isna(genre) or not isinstance(genre, str):
            return 'Unknown'
        
        # Clean and normalize
        clean_genre = genre.strip().lower()
        
        # Apply mapping
        return self.genre_mapping.get(clean_genre, genre.strip())
    
    def parse_genre_string(self, genre_string: str) -> List[str]:
        """Parse pipe-separated genre string into normalized list"""
        if pd.isna(genre_string) or not isinstance(genre_string, str):
            return ['Unknown']
        
        # Split by pipe and normalize each genre
        genres = [self.normalize_genre_name(g) for g in genre_string.split('|')]
        
        # Remove duplicates while preserving order
        seen = set()
        normalized_genres = []
        for genre in genres:
            if genre not in seen:
                normalized_genres.append(genre)
                seen.add(genre)
        
        return normalized_genres if normalized_genres else ['Unknown']
    
    def normalize_movie_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize genres in movies DataFrame"""
        self.logger.info("Starting genre normalization...")
        
        df_normalized = df.copy()
        
        # Parse and normalize genres
        df_normalized['genres_list'] = df_normalized['genres'].apply(self.parse_genre_string)
        
        # Create normalized genre string
        df_normalized['genres_normalized'] = df_normalized['genres_list'].apply(
            lambda x: '|'.join(x)
        )
        
        # Calculate genre statistics
        all_genres = []
        for genre_list in df_normalized['genres_list']:
            all_genres.extend(genre_list)
        
        genre_counts = pd.Series(all_genres).value_counts()
        
        self.genre_stats = {
            'total_movies': len(df_normalized),
            'unique_genres': len(genre_counts),
            'most_common_genres': genre_counts.head(10).to_dict(),
            'avg_genres_per_movie': df_normalized['genres_list'].apply(len).mean(),
            'movies_without_genres': (df_normalized['genres_list'].apply(lambda x: x == ['Unknown'])).sum()
        }
        
        self.logger.info(f"Genre normalization completed. Found {self.genre_stats['unique_genres']} unique genres")
        
        return df_normalized
    
    def create_genre_taxonomy(self) -> Dict[str, List[str]]:
        """Create hierarchical genre taxonomy"""
        taxonomy = {
            'Action & Adventure': ['Action', 'Adventure', 'War'],
            'Comedy & Family': ['Comedy', 'Family', "Children's"],
            'Drama & Romance': ['Drama', 'Romance'],
            'Horror & Thriller': ['Horror', 'Thriller', 'Mystery'],
            'Science Fiction & Fantasy': ['Science Fiction', 'Fantasy'],
            'Documentary & Biography': ['Documentary'],
            'Animation': ['Animation'],
            'Musical & Arts': ['Musical'],
            'Crime & Noir': ['Crime', 'Film-Noir'],
            'Western': ['Western'],
            'Other': ['Unknown', 'IMAX']
        }
        
        return taxonomy
    
    def get_genre_statistics(self) -> Dict:
        """Get genre normalization statistics"""
        return self.genre_stats
