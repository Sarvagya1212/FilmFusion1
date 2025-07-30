#!/usr/bin/env python3
"""
Enhanced initial setup script for FilmFusion Phase 1
Location: scripts/run_initial_setup.py
"""

import sys
import os
import logging
from pathlib import Path
import argparse
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from src.database.db_connector import DatabaseConnector
from src.data_processing.etl_pipeline import ETLPipeline

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FilmFusion Initial Setup')
    parser.add_argument('--dataset-size', choices=['latest-small', 'latest', '25m', '20m'],
                       default='latest-small', help='MovieLens dataset size')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip component testing')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of dataset')
    parser.add_argument('--test-only', action='store_true',
                       help='Run tests only, skip ETL pipeline')
    
    return parser.parse_args()

def run_component_tests():
    """Run component tests"""
    logger = logging.getLogger(__name__)
    logger.info("[TEST] Running component tests...")
    
    try:
        # Import and run tests
        from tests.test_data_components import run_tests
        success = run_tests()
        
        if success:
            logger.info("[SUCCESS] All component tests passed!")
            return True
        else:
            logger.error("[ERROR] Some component tests failed!")
            return False
            
    except ImportError as e:
        logger.warning(f"[WARNING]  Could not import test module: {e}")
        logger.warning("Continuing without tests...")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Error running tests: {e}")
        return False

def check_prerequisites():
    """Check system prerequisites"""
    logger = logging.getLogger(__name__)
    logger.info("[CHECK] Checking prerequisites...")
    
    # Check environment variables
    required_env_vars = ['DATABASE_URL']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"[ERROR] Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file")
        return False
    
    # Check database connection
    db_connector = DatabaseConnector()
    if not db_connector.test_connection():
        logger.error("[ERROR] Database connection failed")
        logger.error("Please ensure PostgreSQL is running and accessible")
        return False
    
    logger.info("[SUCCESS] Prerequisites check passed!")
    return True

def main():
    """Run complete Phase 1 setup with enhanced error handling"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    start_time = time.time()
    
    logger.info("[START] STARTING FILMFUSION PHASE 1 SETUP")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {args.dataset_size}")
    logger.info(f"Force download: {args.force_download}")
    logger.info(f"Skip tests: {args.skip_tests}")
    logger.info(f"Test only: {args.test_only}")
    logger.info("=" * 60)
    
    try:
        # 1. Check prerequisites
        if not check_prerequisites():
            logger.error("[ERROR] Prerequisites check failed")
            return False
        
        # 2. Run component tests (unless skipped)
        if not args.skip_tests:
            test_success = run_component_tests()
            if not test_success and not args.force_download:
                logger.error("[ERROR] Component tests failed. Use --force-download to continue anyway.")
                return False
        
        # 3. If test-only mode, exit here
        if args.test_only:
            logger.info("[SUCCESS] Test-only mode completed successfully!")
            return True
        
        # 4. Create database tables
        logger.info("[DATABASE]  Setting up database...")
        db_connector = DatabaseConnector()
        if not db_connector.create_tables():
            logger.error("[ERROR] Failed to create database tables")
            return False
        
        # 5. Run ETL pipeline
        logger.info("[PIPELINE]  Running ETL pipeline...")
        etl = ETLPipeline(dataset_size=args.dataset_size)
        success = etl.run_full_pipeline()
        
        if success:
            # 6. Final validation
            logger.info("[CHECK] Running final validation...")
            summary = db_connector.get_data_summary()
            
            if summary.get('ratings', 0) > 0 and summary.get('movies', 0) > 0:
                logger.info("[SUCCESS] PHASE 1 SETUP COMPLETED SUCCESSFULLY!")
                logger.info("[INFO] Final Summary:")
                for key, value in summary.items():
                    logger.info(f"   {key.capitalize()}: {value:,}")
                
                # Calculate and log total time
                total_time = time.time() - start_time
                logger.info(f"[TIMING]  Total setup time: {total_time:.2f} seconds")
                
                # Next steps guidance
                logger.info("\n[NEXT] NEXT STEPS:")
                logger.info("1. Explore data: jupyter notebook notebooks/01_initial_eda.ipynb")
                logger.info("2. Check data quality: data/processed/data_quality_report.json")
                logger.info("3. Review pipeline stats: data/processed/pipeline_stats.json")
                logger.info("4. Access Airflow UI: http://localhost:8080")
                
                return True
            else:
                logger.error("[ERROR] Data validation failed - insufficient data loaded")
                return False
        else:
            logger.error("[ERROR] ETL pipeline failed")
            return False
            
    except KeyboardInterrupt:
        logger.warning("[WARNING]  Setup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error during setup: {e}")
        logger.exception("Full traceback:")
        return False
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
