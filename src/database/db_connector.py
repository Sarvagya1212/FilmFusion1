import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect
from config.database import engine, SessionLocal, Base
from src.database.models import User, Movie, Genre, Rating, Tag, Review, movie_genre_association
import logging
from typing import Dict, List, Optional
from contextlib import contextmanager

class DatabaseConnector:
    """Enhanced database operations with comprehensive error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.engine = engine
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("Database connection successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create all database tables with error handling"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
            
            # Verify tables were created
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            expected_tables = ['users', 'movies', 'genres', 'ratings', 'tags', 'reviews', 'movie_genres']
            
            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                self.logger.warning(f"Some tables may not have been created: {missing_tables}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            return False
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def bulk_insert_users(self, ratings_df: pd.DataFrame) -> bool:
        """Insert unique users from ratings data"""
        try:
            with self.get_session() as session:
                # Get unique user IDs
                unique_users = ratings_df['userId'].unique()
                
                # Create user records
                users_data = [{'user_id': int(user_id)} for user_id in unique_users]
                
                # Bulk insert
                session.execute(
                    User.__table__.insert(),
                    users_data
                )
                
                self.logger.info(f"Inserted {len(users_data)} users")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert users: {e}")
            return False
    
    def bulk_insert_movies(self, movies_df: pd.DataFrame) -> bool:
        """Insert movies with genre relationships"""
        try:
            with self.get_session() as session:
                # Insert movies
                movies_data = []
                for _, row in movies_df.iterrows():
                    movie_data = {
                        'movie_id': int(row['movieId']),
                        'title': row['title'],
                        'clean_title': row.get('clean_title', row['title']),
                        'year': int(row['year']) if pd.notna(row.get('year')) else None
                    }
                    movies_data.append(movie_data)
                
                session.execute(Movie.__table__.insert(), movies_data)
                
                # Handle genres
                if 'genres_list' in movies_df.columns:
                    self._insert_genres_and_relationships(session, movies_df)
                
                self.logger.info(f"Inserted {len(movies_data)} movies")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert movies: {e}")
            return False
    
    def _insert_genres_and_relationships(self, session: Session, movies_df: pd.DataFrame):
        """Insert genres and movie-genre relationships"""
        # Collect all unique genres
        all_genres = set()
        for genres_list in movies_df['genres_list']:
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
        
        # Insert genres
        genres_data = [{'name': genre} for genre in all_genres]
        session.execute(Genre.__table__.insert(), genres_data)
        
        # Get genre ID mapping
        genre_map = {g.name: g.id for g in session.query(Genre).all()}
        movie_map = {m.movie_id: m.id for m in session.query(Movie).all()}
        
        # Create movie-genre relationships
        relationships = []
        for _, row in movies_df.iterrows():
            movie_id = movie_map.get(int(row['movieId']))
            if movie_id and isinstance(row.get('genres_list'), list):
                for genre_name in row['genres_list']:
                    genre_id = genre_map.get(genre_name)
                    if genre_id:
                        relationships.append({
                            'movie_id': movie_id,
                            'genre_id': genre_id
                        })
        
        if relationships:
            session.execute(movie_genre_association.insert(), relationships)
    
    def bulk_insert_ratings(self, ratings_df: pd.DataFrame) -> bool:
        """Insert ratings with proper foreign key relationships"""
        try:
            with self.get_session() as session:
                # Get ID mappings
                user_map = {u.user_id: u.id for u in session.query(User).all()}
                movie_map = {m.movie_id: m.id for m in session.query(Movie).all()}
                
                # Prepare ratings data
                ratings_data = []
                skipped_count = 0
                
                for _, row in ratings_df.iterrows():
                    user_id = user_map.get(int(row['userId']))
                    movie_id = movie_map.get(int(row['movieId']))
                    
                    if user_id and movie_id:
                        rating_data = {
                            'user_id': user_id,
                            'movie_id': movie_id,
                            'rating': float(row['rating']),
                            'timestamp': row.get('timestamp')
                        }
                        ratings_data.append(rating_data)
                    else:
                        skipped_count += 1
                
                # Bulk insert ratings
                if ratings_data:
                    session.execute(Rating.__table__.insert(), ratings_data)
                
                self.logger.info(f"Inserted {len(ratings_data)} ratings, skipped {skipped_count}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert ratings: {e}")
            return False
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of loaded data"""
        try:
            with self.get_session() as session:
                summary = {
                    'users': session.query(User).count(),
                    'movies': session.query(Movie).count(),
                    'genres': session.query(Genre).count(),
                    'ratings': session.query(Rating).count(),
                    'tags': session.query(Tag).count(),
                    'reviews': session.query(Review).count()
                }
                
                # Additional statistics
                if summary['ratings'] > 0:
                    avg_rating = session.execute(text("SELECT AVG(rating) FROM ratings")).scalar()
                    summary['average_rating'] = round(float(avg_rating), 2)
                
                return summary
        except Exception as e:
            self.logger.error(f"Failed to get data summary: {e}")
            return {}
