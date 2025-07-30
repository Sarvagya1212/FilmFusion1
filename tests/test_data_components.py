import unittest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.movielens_fetcher import MovieLensDataFetcher
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.genre_normalizer import GenreNormalizer

class TestMovieLensDataFetcher(unittest.TestCase):
    """Test MovieLens data fetching functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = MovieLensDataFetcher(data_dir=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_download_dataset(self):
        """Test dataset download functionality"""
        success = self.fetcher.download_dataset(size="latest-small")
        self.assertTrue(success, "Dataset download should succeed")
        
        # Check if extracted files exist
        extracted_dir = Path(self.temp_dir) / "ml-latest-small"
        self.assertTrue(extracted_dir.exists(), "Extracted directory should exist")
        
        required_files = ["movies.csv", "ratings.csv"]
        for file_name in required_files:
            file_path = extracted_dir / file_name
            self.assertTrue(file_path.exists(), f"{file_name} should exist")
    
    def test_load_data_files(self):
        """Test loading of data files"""
        # First download the dataset
        self.fetcher.download_dataset(size="latest-small")
        
        # Test loading ratings
        ratings_df = self.fetcher.load_ratings(size="latest-small")
        self.assertIsNotNone(ratings_df, "Ratings DataFrame should not be None")
        self.assertGreater(len(ratings_df), 0, "Ratings should contain data")
        
        # Test required columns
        required_columns = ['userId', 'movieId', 'rating', 'timestamp']
        for col in required_columns:
            self.assertIn(col, ratings_df.columns, f"Column {col} should exist in ratings")
        
        # Test loading movies
        movies_df = self.fetcher.load_movies(size="latest-small")
        self.assertIsNotNone(movies_df, "Movies DataFrame should not be None")
        self.assertGreater(len(movies_df), 0, "Movies should contain data")

class TestDataCleaner(unittest.TestCase):
    """Test data cleaning functionality"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
    
    def create_sample_ratings_data(self):
        """Create sample ratings data for testing"""
        return pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3, 1],  # Duplicate user-movie pair
            'movieId': [1, 2, 1, 3, 2, 4, 1], # Duplicate user-movie pair
            'rating': [4.5, 3.0, 6.0, 2.5, 1.0, 4.0, 4.5],  # Invalid rating (6.0)
            'timestamp': [1234567890, 1234567891, 1234567892, 1234567893, 1234567894, 1234567895, 1234567890]
        })
    
    def create_sample_movies_data(self):
        """Create sample movies data for testing"""
        return pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5],
            'title': ['Movie A (2020)', 'Movie B (1995)', 'Movie C', '', 'Movie D (2021)'],
            'genres': ['Action|Drama', 'Comedy', 'Sci-Fi|Thriller', 'Drama', None]
        })
    
    def test_clean_ratings(self):
        """Test ratings cleaning functionality"""
        sample_data = self.create_sample_ratings_data()
        cleaned_data = self.cleaner.clean_ratings(sample_data)
        
        # Check that duplicates are removed
        self.assertFalse(cleaned_data.duplicated(subset=['userId', 'movieId']).any(),
                        "No duplicates should remain")
        
        # Check that invalid ratings are removed
        self.assertTrue((cleaned_data['rating'] >= 0.5).all(), "All ratings should be >= 0.5")
        self.assertTrue((cleaned_data['rating'] <= 5.0).all(), "All ratings should be <= 5.0")
        
        # Check that timestamp is converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_data['timestamp']),
                       "Timestamp should be datetime type")
    
    def test_clean_movies(self):
        """Test movies cleaning functionality"""
        sample_data = self.create_sample_movies_data()
        cleaned_data = self.cleaner.clean_movies(sample_data)
        
        # Check that empty titles are removed
        self.assertFalse(cleaned_data['clean_title'].isin(['', None]).any(),
                        "No empty titles should remain")
        
        # Check year extraction
        self.assertEqual(cleaned_data.loc[0, 'year'], 2020, "Year should be extracted correctly")
        self.assertEqual(cleaned_data.loc[1, 'year'], 1995, "Year should be extracted correctly")
        
        # Check clean title
        self.assertEqual(cleaned_data.loc[0, 'clean_title'], 'Movie A', "Title should be cleaned")

class TestGenreNormalizer(unittest.TestCase):
    """Test genre normalization functionality"""
    
    def setUp(self):
        self.normalizer = GenreNormalizer()
    
    def test_normalize_genre_name(self):
        """Test single genre normalization"""
        # Test case normalization
        self.assertEqual(self.normalizer.normalize_genre_name('sci-fi'), 'Science Fiction')
        self.assertEqual(self.normalizer.normalize_genre_name('SCI-FI'), 'Science Fiction')
        
        # Test synonym mapping
        self.assertEqual(self.normalizer.normalize_genre_name('thriller'), 'Thriller')
        self.assertEqual(self.normalizer.normalize_genre_name('suspense'), 'Thriller')
    
    def test_parse_genre_string(self):
        """Test genre string parsing"""
        genre_string = "Action|Sci-Fi|Thriller"
        parsed = self.normalizer.parse_genre_string(genre_string)
        
        expected = ['Action', 'Science Fiction', 'Thriller']
        self.assertEqual(parsed, expected, "Genres should be parsed and normalized correctly")
    
    def test_normalize_movie_genres(self):
        """Test movie genre normalization"""
        sample_data = pd.DataFrame({
            'movieId': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genres': ['Action|Sci-Fi', 'Comedy|Romance', 'Thriller|Mystery']
        })
        
        normalized_data = self.normalizer.normalize_movie_genres(sample_data)
        
        # Check that new columns are added
        self.assertIn('genres_list', normalized_data.columns, 
                     "genres_list column should be added")
        self.assertIn('genres_normalized', normalized_data.columns,
                     "genres_normalized column should be added")
        
        # Check normalization
        first_genres = normalized_data.loc[0, 'genres_list']
        self.assertIn('Science Fiction', first_genres, "Sci-Fi should be normalized to Science Fiction")

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMovieLensDataFetcher))
    test_suite.addTest(unittest.makeSuite(TestDataCleaner))
    test_suite.addTest(unittest.makeSuite(TestGenreNormalizer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
