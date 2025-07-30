import requests
import time
import pandas as pd
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

class TMDBApiFetcher:
    """Enhanced TMDB API fetcher with better error handling and environment variable loading"""
    
    def __init__(self):
        # Load environment variables
        self._load_environment()
        
        # Initialize API configuration
        self.api_key = os.getenv('TMDB_API_KEY')
        self.base_url = 'https://api.themoviedb.org/3'
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Rate limiting
        self.requests_per_second = 4  # TMDB allows 40 requests per 10 seconds
        self.last_request_time = 0
        
        # Validate API key
        self._validate_api_key()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        # Try multiple paths for .env file
        possible_env_paths = [
            Path.cwd() / '.env',
            Path(__file__).parent.parent.parent / '.env',
            Path.cwd().parent / '.env'
        ]
        
        env_loaded = False
        for env_path in possible_env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                print(f"📁 Loaded environment from: {env_path}")
                env_loaded = True
                break
        
        if not env_loaded:
            print("⚠️  No .env file found in expected locations")
            print(f"Searched paths: {[str(p) for p in possible_env_paths]}")
    
    def _validate_api_key(self):
        """Validate that API key is available and properly formatted"""
        if not self.api_key:
            error_msg = (
                "TMDB API key not found. Please:\n"
                "1. Create a .env file in your project root\n"
                "2. Add: TMDB_API_KEY=your_actual_api_key\n"
                "3. Get your API key from: https://www.themoviedb.org/settings/api\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Environment variables loaded: {bool(os.getenv('TMDB_API_KEY'))}"
            )
            raise ValueError(error_msg)
        
        if len(self.api_key.strip()) < 10:
            raise ValueError("TMDB API key appears to be invalid (too short)")
        
        self.logger.info(f"✅ TMDB API key loaded successfully: {self.api_key[:8]}...")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with comprehensive error handling"""
        if params is None:
            params = {}
        
        params['api_key'] = self.api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        self._rate_limit()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            # Check for API key issues
            if response.status_code == 401:
                raise ValueError("Invalid TMDB API key. Please check your API key in .env file")
            
            response.raise_for_status()
            
            data = response.json()
            self.logger.debug(f"Successfully fetched data from {endpoint}")
            return data
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                self.logger.warning(f"Resource not found: {endpoint}")
            elif response.status_code == 429:
                self.logger.warning("Rate limit exceeded, backing off...")
                time.sleep(10)  # Back off for 10 seconds
            else:
                self.logger.error(f"HTTP error {response.status_code}: {e}")
            return None
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for {endpoint}: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test TMDB API connection"""
        try:
            result = self._make_request("configuration")
            if result:
                self.logger.info("✅ TMDB API connection successful")
                return True
            else:
                self.logger.error("❌ TMDB API connection failed")
                return False
        except Exception as e:
            self.logger.error(f"❌ TMDB API connection test failed: {e}")
            return False
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search for movies by title with optional year filter"""
        params = {'query': title}
        if year:
            params['year'] = year
        
        return self._make_request("search/movie", params)
    
    def fetch_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch comprehensive movie details"""
        params = {'append_to_response': 'credits,reviews,keywords,videos'}
        return self._make_request(f"movie/{tmdb_id}", params)
    
    def fetch_movie_credits(self, tmdb_id: int) -> Optional[Dict]:
        """Fetch cast and crew information"""
        return self._make_request(f"movie/{tmdb_id}/credits")
    
    def fetch_movie_reviews(self, tmdb_id: int, page: int = 1) -> Optional[Dict]:
        """Fetch user reviews with pagination"""
        params = {'page': page}
        return self._make_request(f"movie/{tmdb_id}/reviews", params)
    
    def batch_search_movies(self, movie_titles: List[str], 
                           progress_callback=None) -> List[Dict]:
        """Batch search multiple movies with progress tracking"""
        results = []
        total = len(movie_titles)
        
        for i, title in enumerate(movie_titles, 1):
            result = self.search_movie(title)
            results.append({
                'original_title': title,
                'tmdb_data': result,
                'search_timestamp': datetime.now().isoformat()
            })
            
            if progress_callback and i % 10 == 0:
                progress_callback(i, total)
            
            # Log progress
            if i % 50 == 0:
                self.logger.info(f"Processed {i}/{total} movie searches")
        
        return results
