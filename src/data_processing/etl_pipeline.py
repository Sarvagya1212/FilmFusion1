import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from src.data_ingestion.movielens_fetcher import MovieLensDataFetcher
from src.data_ingestion.tmdb_fetcher import TMDBApiFetcher
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.genre_normalizer import GenreNormalizer
from src.database.db_connector import DatabaseConnector

class ETLPipeline:
    """Comprehensive ETL pipeline with monitoring and validation"""
    
    def __init__(self, dataset_size: str = "latest-small"):
        self.logger = logging.getLogger(__name__)
        self.dataset_size = dataset_size
        
        # Initialize components
        self.movielens_fetcher = MovieLensDataFetcher()
        self.tmdb_fetcher = TMDBApiFetcher()
        self.data_cleaner = DataCleaner()
        self.genre_normalizer = GenreNormalizer()
        self.db_connector = DatabaseConnector()
        
        # Pipeline state
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'duration': None,
            'status': 'initialized',
            'errors': [],
            'warnings': []
        }
        
        # Create processed data directory
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_data(self) -> Dict[str, pd.DataFrame]:
        """Extract data from all sources with validation"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING DATA EXTRACTION")
        self.logger.info("=" * 50)
        
        extracted_data = {}
        
        try:
            # 1. Download and extract MovieLens data
            self.logger.info(f"Downloading MovieLens {self.dataset_size} dataset...")
            success = self.movielens_fetcher.download_dataset(self.dataset_size)
            
            if not success:
                raise Exception("Failed to download MovieLens dataset")
            
            # 2. Load datasets
            self.logger.info("Loading datasets...")
            
            # Load ratings
            ratings_df = self.movielens_fetcher.load_ratings(self.dataset_size)
            if ratings_df is not None:
                extracted_data['ratings'] = ratings_df
                self.logger.info(f"[SUCCESS] Loaded {len(ratings_df):,} ratings")
            else:
                self.pipeline_stats['errors'].append("Failed to load ratings data")
            
            # Load movies
            movies_df = self.movielens_fetcher.load_movies(self.dataset_size)
            if movies_df is not None:
                extracted_data['movies'] = movies_df
                self.logger.info(f"[SUCCESS] Loaded {len(movies_df):,} movies")
            else:
                self.pipeline_stats['errors'].append("Failed to load movies data")
            
            # Load tags if available
            tags_df = self.movielens_fetcher.load_tags(self.dataset_size)
            if tags_df is not None:
                extracted_data['tags'] = tags_df
                self.logger.info(f"[SUCCESS] Loaded {len(tags_df):,} tags")
            
            # 3. Get dataset info
            dataset_info = self.movielens_fetcher.get_dataset_info(self.dataset_size)
            self.logger.info(f"Dataset summary: {dataset_info}")
            
            self.logger.info("[SUCCESS] DATA EXTRACTION COMPLETED")
            return extracted_data
            
        except Exception as e:
            error_msg = f"Data extraction failed: {e}"
            self.logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def transform_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Transform and clean all extracted data"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING DATA TRANSFORMATION")
        self.logger.info("=" * 50)
        
        transformed_data = {}
        
        try:
            # 1. Clean ratings data
            if 'ratings' in raw_data:
                self.logger.info("Cleaning ratings data...")
                cleaned_ratings = self.data_cleaner.clean_ratings(raw_data['ratings'])
                transformed_data['ratings'] = cleaned_ratings
                
                # Save processed data
                ratings_path = self.processed_dir / "ratings_cleaned.csv"
                cleaned_ratings.to_csv(ratings_path, index=False)
                self.logger.info(f"💾 Saved cleaned ratings to {ratings_path}")
            
            # 2. Clean and normalize movies data
            if 'movies' in raw_data:
                self.logger.info("Cleaning movies data...")
                cleaned_movies = self.data_cleaner.clean_movies(raw_data['movies'])
                
                self.logger.info("Normalizing genres...")
                normalized_movies = self.genre_normalizer.normalize_movie_genres(cleaned_movies)
                transformed_data['movies'] = normalized_movies
                
                # Save processed data
                movies_path = self.processed_dir / "movies_cleaned.csv"
                normalized_movies.to_csv(movies_path, index=False)
                self.logger.info(f"💾 Saved cleaned movies to {movies_path}")
                
                # Save genre statistics with JSON serialization fix
                genre_stats = self.genre_normalizer.get_genre_statistics()
                genre_stats_serializable = self._make_json_serializable(genre_stats)
                genre_stats_path = self.processed_dir / "genre_statistics.json"
                with open(genre_stats_path, 'w') as f:
                    json.dump(genre_stats_serializable, f, indent=2)
            
            # 3. Clean tags data if available
            if 'tags' in raw_data and raw_data['tags'] is not None:
                self.logger.info("Cleaning tags data...")
                # Implement tag cleaning (similar to existing clean_tags method)
                transformed_data['tags'] = raw_data['tags']  # Placeholder
            
            # 4. Generate data quality report with JSON serialization fix
            quality_report = self.data_cleaner.generate_data_quality_report(transformed_data)
            quality_report_serializable = self._make_json_serializable(quality_report)
            report_path = self.processed_dir / "data_quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(quality_report_serializable, f, indent=2)
            
            self.logger.info(f"📊 Data quality report saved to {report_path}")
            self.logger.info("✅ DATA TRANSFORMATION COMPLETED")
            
            return transformed_data
            
        except Exception as e:
            error_msg = f"Data transformation failed: {e}"
            self.logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            raise

    def _make_json_serializable(self, obj):
        """Convert numpy/pandas data types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj

    def load_data(self, transformed_data: Dict[str, pd.DataFrame]) -> bool:
        """Load transformed data into database"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING DATA LOADING")
        self.logger.info("=" * 50)
        
        try:
            # 1. Test database connection
            if not self.db_connector.test_connection():
                raise Exception("Database connection failed")
            
            # 2. Create tables
            if not self.db_connector.create_tables():
                raise Exception("Failed to create database tables")
            
            # 3. Load data in order (users first, then movies, then ratings)
            load_order = ['movies', 'ratings']  # Users will be extracted from ratings
            
            for table_name in load_order:
                if table_name in transformed_data:
                    df = transformed_data[table_name]
                    self.logger.info(f"Loading {len(df):,} records into {table_name}...")
                    
                    if table_name == 'movies':
                        success = self.db_connector.bulk_insert_movies(df)
                    elif table_name == 'ratings':
                        # First ensure users exist
                        self.db_connector.bulk_insert_users(df)
                        success = self.db_connector.bulk_insert_ratings(df)
                    else:
                        success = True  # Placeholder for other tables
                    
                    if success:
                        self.logger.info(f"[SUCCESS] Successfully loaded {table_name}")
                    else:
                        raise Exception(f"Failed to load {table_name}")
            
            # 4. Validate loaded data
            summary = self.db_connector.get_data_summary()
            self.logger.info(f"[STATS] Database summary: {summary}")
            
            # Save summary
            summary_path = self.processed_dir / "database_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info("[SUCCESS] DATA LOADING COMPLETED")
            return True
            
        except Exception as e:
            error_msg = f"Data loading failed: {e}"
            self.logger.error(error_msg)
            self.pipeline_stats['errors'].append(error_msg)
            return False
    
    def run_full_pipeline(self) -> bool:
        """Execute the complete ETL pipeline with comprehensive monitoring"""
        self.pipeline_stats['start_time'] = datetime.now()
        self.pipeline_stats['status'] = 'running'
        
        self.logger.info("[START] STARTING FILMFUSION ETL PIPELINE")
        self.logger.info(f"Dataset: {self.dataset_size}")
        self.logger.info(f"Start time: {self.pipeline_stats['start_time']}")
        
        try:
            # Extract
            raw_data = self.extract_data()
            
            # Transform
            transformed_data = self.transform_data(raw_data)
            
            # Load
            load_success = self.load_data(transformed_data)
            
            if load_success:
                self.pipeline_stats['status'] = 'completed'
                self.logger.info("[COMPLETE] ETL PIPELINE COMPLETED SUCCESSFULLY!")
            else:
                self.pipeline_stats['status'] = 'failed'
                self.logger.error("[ERROR] ETL PIPELINE FAILED AT LOADING STAGE")
            
            return load_success
            
        except Exception as e:
            self.pipeline_stats['status'] = 'failed'
            self.logger.error(f"[ERROR] ETL PIPELINE FAILED: {e}")
            return False
            
        finally:
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['duration'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            
            # Save pipeline statistics
            stats_path = self.processed_dir / "pipeline_stats.json"
            stats_to_save = self.pipeline_stats.copy()
            stats_to_save['start_time'] = stats_to_save['start_time'].isoformat()
            stats_to_save['end_time'] = stats_to_save['end_time'].isoformat()
            
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            
            self.logger.info(f"[TIMING]  Pipeline duration: {self.pipeline_stats['duration']:.2f} seconds")
            self.logger.info(f"[INFO] Pipeline stats saved to {stats_path}")
