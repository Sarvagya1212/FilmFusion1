from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import relationship
from config.database import Base

# Association table for many-to-many relationship between movies and genres
movie_genre_association = Table(
    'movie_genres',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id')),
    Column('genre_id', Integer, ForeignKey('genres.id'))
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True)  # Original MovieLens user ID
    created_at = Column(DateTime)
    
    # Relationships
    ratings = relationship("Rating", back_populates="user")
    tags = relationship("Tag", back_populates="user")

class Movie(Base):
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, unique=True, index=True)  # Original MovieLens movie ID
    title = Column(String, index=True)
    clean_title = Column(String, index=True)
    year = Column(Integer)
    imdb_id = Column(String)
    tmdb_id = Column(Integer)
    
    # Relationships
    ratings = relationship("Rating", back_populates="movie")
    tags = relationship("Tag", back_populates="movie")
    genres = relationship("Genre", secondary=movie_genre_association, back_populates="movies")

class Genre(Base):
    __tablename__ = 'genres'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    
    # Relationships
    movies = relationship("Movie", secondary=movie_genre_association, back_populates="genres")

class Rating(Base):
    __tablename__ = 'ratings'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_id = Column(Integer, ForeignKey('movies.id'))
    rating = Column(Float)
    timestamp = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

class Tag(Base):
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_id = Column(Integer, ForeignKey('movies.id'))
    tag = Column(String)
    timestamp = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="tags")
    movie = relationship("Movie", back_populates="tags")

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, ForeignKey('movies.id'))
    source = Column(String)  # 'tmdb', 'imdb', etc.
    author = Column(String)
    content = Column(Text)
    rating = Column(Float)
    sentiment_score = Column(Float)
    created_at = Column(DateTime)
    
    # Relationships
    movie = relationship("Movie")
