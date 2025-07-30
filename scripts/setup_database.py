#!/usr/bin/env python3
"""
Database setup script for FilmFusion
Location: scripts/setup_database.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from config.database import engine, Base
    from src.database.models import User, Movie, Genre, Rating, Tag, Review
    from src.database.db_connector import DatabaseConnector
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Make sure you've created the database models and config files first")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_database():
    """Create all database tables"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating database tables...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("[SUCCESS] Database tables created successfully!")
        
        # Test connection
        db_connector = DatabaseConnector()
        session = db_connector.get_session()
        
        logger.info("[SUCCESS] Database connection test successful!")
        session.close()
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Database setup failed: {e}")
        return False

def main():
    """Main execution function"""
    print("[START] Starting FilmFusion database setup...")
    
    success = setup_database()
    
    if success:
        print("[SUCCESS] Database setup completed successfully!")
    else:
        print("[ERROR] Database setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
