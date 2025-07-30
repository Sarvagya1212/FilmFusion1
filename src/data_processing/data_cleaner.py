import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime

class DataCleaner:
    """Comprehensive data cleaning with validation and quality checks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleaning_stats = {}
    
    def clean_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ratings data with comprehensive validation"""
        self.logger.info("Starting ratings data cleaning...")
        original_count = len(df)
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Remove duplicates
        duplicates_before = df_clean.duplicated(subset=['userId', 'movieId']).sum()
        df_clean = df_clean.drop_duplicates(subset=['userId', 'movieId'], keep='last')
        duplicates_removed = duplicates_before - df_clean.duplicated(subset=['userId', 'movieId']).sum()
        
        # 2. Validate and clean rating values
        invalid_ratings = ~df_clean['rating'].between(0.5, 5.0)
        invalid_count = invalid_ratings.sum()
        df_clean = df_clean[~invalid_ratings]
        
        # 3. Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        df_clean = df_clean.dropna()
        missing_removed = missing_before - df_clean.isnull().sum().sum()
        
        # 4. Convert timestamp to datetime
        if 'timestamp' in df_clean.columns:
            try:
                df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], unit='s')
                # Remove future timestamps (data quality check)
                future_timestamps = df_clean['timestamp'] > pd.Timestamp.now()
                future_count = future_timestamps.sum()
                df_clean = df_clean[~future_timestamps]
            except Exception as e:
                self.logger.warning(f"Error converting timestamps: {e}")
        
        # 5. Ensure proper data types
        df_clean['userId'] = df_clean['userId'].astype('int32')
        df_clean['movieId'] = df_clean['movieId'].astype('int32')
        df_clean['rating'] = df_clean['rating'].astype('float32')
        
        # Store cleaning statistics
        self.cleaning_stats['ratings'] = {
            'original_count': original_count,
            'final_count': len(df_clean),
            'duplicates_removed': duplicates_removed,
            'invalid_ratings_removed': invalid_count,
            'missing_values_removed': missing_removed,
            'rows_removed': original_count - len(df_clean),
            'data_quality_score': len(df_clean) / original_count
        }
        
        self.logger.info(f"Ratings cleaning completed: {original_count} → {len(df_clean)} rows "
                        f"({self.cleaning_stats['ratings']['data_quality_score']:.2%} retained)")
        
        return df_clean
    
    def clean_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean movies data with title parsing and genre handling"""
        self.logger.info("Starting movies data cleaning...")
        original_count = len(df)
        
        df_clean = df.copy()
        
        # 1. Remove movies without titles
        missing_titles = df_clean['title'].isnull() | (df_clean['title'].str.strip() == '')
        missing_title_count = missing_titles.sum()
        df_clean = df_clean[~missing_titles]
        
        # 2. Extract year from title using regex
        year_pattern = r'\((\d{4})\)(?:\s*$)'
        df_clean['year'] = df_clean['title'].str.extract(year_pattern)[0].astype('Int64')
        
        # 3. Clean title (remove year and extra whitespace)
        df_clean['clean_title'] = (df_clean['title']
                                  .str.replace(year_pattern, '', regex=True)
                                  .str.strip())
        
        # 4. Handle genres
        df_clean['genres'] = df_clean['genres'].fillna('(no genres listed)')
        
        # 5. Add genre count
        df_clean['genre_count'] = (df_clean['genres']
                                  .str.split('|')
                                  .apply(lambda x: len([g for g in x if g.strip() and g != '(no genres listed)'])))
        
        # 6. Validate years (reasonable range)
        current_year = datetime.now().year
        invalid_years = (df_clean['year'] < 1888) | (df_clean['year'] > current_year + 5)  # Allow some future releases
        invalid_year_count = invalid_years.sum()
        # Don't remove, but log the anomalies
        if invalid_year_count > 0:
            self.logger.warning(f"Found {invalid_year_count} movies with questionable years")
        
        # 7. Ensure proper data types
        df_clean['movieId'] = df_clean['movieId'].astype('int32')
        
        # Store cleaning statistics
        self.cleaning_stats['movies'] = {
            'original_count': original_count,
            'final_count': len(df_clean),
            'missing_titles_removed': missing_title_count,
            'years_extracted': df_clean['year'].notna().sum(),
            'movies_with_genres': (df_clean['genres'] != '(no genres listed)').sum(),
            'avg_genres_per_movie': df_clean['genre_count'].mean(),
            'data_quality_score': len(df_clean) / original_count
        }
        
        self.logger.info(f"Movies cleaning completed: {original_count} → {len(df_clean)} rows")
        
        return df_clean
    
    def generate_data_quality_report(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Generate comprehensive data quality report with JSON serializable types"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'overall_quality': {}
        }
        
        total_original_rows = 0
        total_final_rows = 0
        
        for dataset_name, df in data_dict.items():
            if dataset_name in self.cleaning_stats:
                stats = self.cleaning_stats[dataset_name].copy()
                
                # Convert numpy types to native Python types
                for key, value in stats.items():
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        stats[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        stats[key] = float(value)
                
                report['datasets'][dataset_name] = stats
                total_original_rows += stats['original_count']
                total_final_rows += stats['final_count']
        
        # Calculate overall statistics
        if total_original_rows > 0:
            report['overall_quality'] = {
                'total_original_rows': int(total_original_rows),
                'total_final_rows': int(total_final_rows),
                'overall_retention_rate': float(total_final_rows / total_original_rows),
                'datasets_processed': len(report['datasets'])
            }
        
        return report
