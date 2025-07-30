import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict
import os
from urllib.parse import urljoin

class MovieLensDataFetcher:
    """Fetch and process MovieLens datasets with comprehensive error handling"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://files.grouplens.org/datasets/movielens/"
        
        # Available dataset sizes
        self.available_datasets = {
            'latest-small': 'ml-latest-small.zip',
            'latest': 'ml-latest.zip',
            '25m': 'ml-25m.zip',
            '20m': 'ml-20m.zip'
        }
    
    def download_dataset(self, size: str = "latest-small", force_download: bool = False) -> bool:
        """Download MovieLens dataset with validation"""
        if size not in self.available_datasets:
            self.logger.error(f"Invalid dataset size: {size}. Available: {list(self.available_datasets.keys())}")
            return False
        
        zip_filename = self.available_datasets[size]
        zip_path = self.data_dir / zip_filename
        extract_dir = self.data_dir / f"ml-{size}"
        
        # Check if already downloaded and extracted
        if not force_download and extract_dir.exists() and any(extract_dir.iterdir()):
            self.logger.info(f"Dataset {size} already exists, skipping download")
            return True
        
        try:
            url = urljoin(self.base_url, zip_filename)
            self.logger.info(f"Downloading MovieLens {size} dataset from {url}")
            
            # Download with progress tracking
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if int(progress) % 10 == 0:  # Log every 10%
                                self.logger.info(f"Download progress: {progress:.1f}%")
            
            # Extract zip file
            self.logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Cleanup zip file
            zip_path.unlink()
            
            self.logger.info(f"Successfully downloaded and extracted {size} dataset")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error downloading dataset: {e}")
            return False
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid zip file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading dataset: {e}")
            return False
    
    def load_ratings(self, size: str = "latest-small") -> Optional[pd.DataFrame]:
        """Load and validate ratings data"""
        try:
            ratings_path = self.data_dir / f"ml-{size}" / "ratings.csv"
            
            if not ratings_path.exists():
                self.logger.error(f"Ratings file not found: {ratings_path}")
                return None
            
            df = pd.read_csv(ratings_path)
            
            # Validate required columns
            required_columns = ['userId', 'movieId', 'rating', 'timestamp']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in ratings data. Expected: {required_columns}")
                return None
            
            self.logger.info(f"Loaded {len(df)} ratings from {ratings_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading ratings data: {e}")
            return None
    
    def load_movies(self, size: str = "latest-small") -> Optional[pd.DataFrame]:
        """Load and validate movies data"""
        try:
            movies_path = self.data_dir / f"ml-{size}" / "movies.csv"
            
            if not movies_path.exists():
                self.logger.error(f"Movies file not found: {movies_path}")
                return None
            
            df = pd.read_csv(movies_path)
            
            # Validate required columns
            required_columns = ['movieId', 'title', 'genres']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in movies data. Expected: {required_columns}")
                return None
            
            self.logger.info(f"Loaded {len(df)} movies from {movies_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading movies data: {e}")
            return None
    
    def load_tags(self, size: str = "latest-small") -> Optional[pd.DataFrame]:
        """Load and validate tags data if available"""
        try:
            tags_path = self.data_dir / f"ml-{size}" / "tags.csv"
            
            if not tags_path.exists():
                self.logger.info(f"Tags file not found: {tags_path} (this is normal for some datasets)")
                return None
            
            df = pd.read_csv(tags_path)
            
            # Validate required columns
            required_columns = ['userId', 'movieId', 'tag', 'timestamp']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in tags data. Expected: {required_columns}")
                return None
            
            self.logger.info(f"Loaded {len(df)} tags from {tags_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading tags data: {e}")
            return None
    
    def get_dataset_info(self, size: str = "latest-small") -> Dict[str, int]:
        """Get basic information about the dataset"""
        info = {}
        
        ratings_df = self.load_ratings(size)
        movies_df = self.load_movies(size)
        tags_df = self.load_tags(size)
        
        if ratings_df is not None:
            info.update({
                'total_ratings': len(ratings_df),
                'unique_users': ratings_df['userId'].nunique(),
                'unique_movies_rated': ratings_df['movieId'].nunique(),
                'rating_range': f"{ratings_df['rating'].min()}-{ratings_df['rating'].max()}",
                'avg_rating': round(ratings_df['rating'].mean(), 2)
            })
        
        if movies_df is not None:
            info.update({
                'total_movies': len(movies_df),
                'unique_genres': len(set(genre.strip() for genres in movies_df['genres'].dropna() 
                                       for genre in genres.split('|') if genre.strip()))
            })
        
        if tags_df is not None:
            info.update({
                'total_tags': len(tags_df),
                'unique_tags': tags_df['tag'].nunique()
            })
        
        return info
