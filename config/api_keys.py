import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEYS = {
    'tmdb': os.getenv('e52ee85b9fa9966a3e9db5aa141ef9cc'),
    'omdb': os.getenv('OMDB_API_KEY'),
}

API_ENDPOINTS = {
    'tmdb_base': 'https://api.themoviedb.org/3',
    'omdb_base': 'http://www.omdbapi.com/',
    'movielens_base': 'https://files.grouplens.org/datasets/movielens/'
}

# Debug: Print API key status (remove in production)
if API_KEYS['tmdb']:
    print(f" TMDB API key loaded: {API_KEYS['tmdb'][:8]}...")
else:
    print(" TMDB API key not found in environment variables")

if API_KEYS['omdb']:
    print(f" OMDB API key loaded: {API_KEYS['omdb'][:8]}...")
else:
    print("  OMDB API key not found in environment variables")
