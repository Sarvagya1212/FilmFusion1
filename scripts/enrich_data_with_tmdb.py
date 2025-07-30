#!/usr/bin/env python3
"""
Enhanced TMDB data enrichment script with batch processing and resume capability
Location: scripts/enrich_data_with_tmdb.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import json
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.tmdb_fetcher import TMDBApiFetcher
from config.logging_config import setup_logging

class EnhancedMovieDataEnricher:
    def __init__(self, batch_size=50, delay_between_batches=10):
        self.logger = setup_logging()
        self.tmdb_fetcher = TMDBApiFetcher()
        self.processed_dir = Path("data/processed")
        self.enriched_dir = Path("data/enriched")
        self.enriched_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        
        # Progress tracking
        self.progress_file = self.enriched_dir / "enrichment_progress.json"
        self.checkpoint_file = self.enriched_dir / "enrichment_checkpoint.csv"
        
    def load_progress(self):
        """Load previous progress if exists"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed_movies': [], 'last_batch_index': 0}
    
    def save_progress(self, progress_data):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_movies_data(self):
        """Load cleaned movies data from Phase 1"""
        movies_path = self.processed_dir / "movies_cleaned.csv"
        if not movies_path.exists():
            raise FileNotFoundError("Movies data not found. Run Phase 1 first.")
        return pd.read_csv(movies_path)
    
    def enrich_movie_batch(self, movies_batch):
        """Enrich a batch of movies"""
        enriched_movies = []
        failed_movies = []
        
        for _, row in movies_batch.iterrows():
            try:
                # Search by title and year
                title = row.get('clean_title', row['title'])
                year = int(row['year']) if pd.notna(row.get('year')) else None
                
                self.logger.debug(f"Searching for: {title} ({year})")
                
                # Search for movie
                search_result = self.tmdb_fetcher.search_movie(title, year)
                
                if search_result and search_result.get('results'):
                    # Take the first (most relevant) result
                    tmdb_movie = search_result['results'][0]
                    
                    # Get detailed movie information
                    movie_details = self.tmdb_fetcher.fetch_movie_details(tmdb_movie['id'])
                    
                    if movie_details:
                        # Extract cast and crew information
                        credits = movie_details.get('credits', {})
                        cast = credits.get('cast', [])
                        crew = credits.get('crew', [])
                        
                        enriched_movie = {
                            'movieId': row['movieId'],
                            'original_title': row['title'],
                            'clean_title': title,
                            'year': row.get('year'),
                            'genres_original': row.get('genres', ''),
                            
                            # TMDB data
                            'tmdb_id': tmdb_movie['id'],
                            'tmdb_title': tmdb_movie.get('title', ''),
                            'overview': movie_details.get('overview', ''),
                            'tagline': movie_details.get('tagline', ''),
                            
                            # Genres from TMDB
                            'genres_tmdb': [g['name'] for g in movie_details.get('genres', [])],
                            'genres_tmdb_ids': [g['id'] for g in movie_details.get('genres', [])],
                            
                            # Movie details
                            'runtime': movie_details.get('runtime'),
                            'vote_average': movie_details.get('vote_average'),
                            'vote_count': movie_details.get('vote_count'),
                            'popularity': movie_details.get('popularity'),
                            'release_date': movie_details.get('release_date'),
                            'budget': movie_details.get('budget', 0),
                            'revenue': movie_details.get('revenue', 0),
                            'status': movie_details.get('status', ''),
                            'original_language': movie_details.get('original_language', ''),
                            
                            # Production details
                            'production_companies': [pc['name'] for pc in movie_details.get('production_companies', [])],
                            'production_countries': [pc['name'] for pc in movie_details.get('production_countries', [])],
                            'spoken_languages': [sl['english_name'] for sl in movie_details.get('spoken_languages', [])],
                            
                            # Cast (top 10)
                            'cast': [actor['name'] for actor in cast[:10]],
                            'cast_ids': [actor['id'] for actor in cast[:10]],
                            'cast_characters': [actor.get('character', '') for actor in cast[:10]],
                            
                            # Directors
                            'directors': [person['name'] for person in crew if person['job'] == 'Director'],
                            'director_ids': [person['id'] for person in crew if person['job'] == 'Director'],
                            
                            # Other key crew
                            'producers': [person['name'] for person in crew if person['job'] == 'Producer'][:5],
                            'writers': [person['name'] for person in crew if person['department'] == 'Writing'][:5],
                            'cinematographers': [person['name'] for person in crew if person['job'] == 'Director of Photography'][:2],
                            'composers': [person['name'] for person in crew if person['department'] == 'Sound' and 'Composer' in person['job']][:2],
                            
                            # Additional metadata
                            'adult': movie_details.get('adult', False),
                            'video': movie_details.get('video', False),
                            'homepage': movie_details.get('homepage', ''),
                            'imdb_id': movie_details.get('imdb_id', ''),
                            
                            # Keywords
                            'keywords': [kw['name'] for kw in movie_details.get('keywords', {}).get('keywords', [])],
                            
                            # Processing metadata
                            'enrichment_date': datetime.now().isoformat(),
                            'tmdb_api_version': '3'
                        }
                        
                        enriched_movies.append(enriched_movie)
                        self.logger.debug(f"✅ Enriched: {title}")
                        
                    else:
                        failed_movies.append({
                            'movieId': row['movieId'],
                            'title': title,
                            'reason': 'details_fetch_failed',
                            'tmdb_id': tmdb_movie['id']
                        })
                else:
                    failed_movies.append({
                        'movieId': row['movieId'],
                        'title': title,
                        'reason': 'no_search_results'
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing movie {row['movieId']}: {e}")
                failed_movies.append({
                    'movieId': row['movieId'],
                    'title': row.get('title', 'Unknown'),
                    'reason': f'exception: {str(e)}'
                })
        
        return enriched_movies, failed_movies
    
    def run_enrichment(self, movies_df, sample_size=None, resume=True):
        """Run the complete enrichment process"""
        # Load previous progress
        progress = self.load_progress() if resume else {'processed_movies': [], 'last_batch_index': 0}
        
        # Filter movies if sample_size is specified
        if sample_size:
            movies_df = movies_df.sample(n=min(sample_size, len(movies_df)), random_state=42).reset_index(drop=True)
            self.logger.info(f"Processing sample of {len(movies_df)} movies")
        
        # Skip already processed movies
        processed_movie_ids = set(progress['processed_movies'])
        remaining_movies = movies_df[~movies_df['movieId'].isin(processed_movie_ids)]
        
        if len(remaining_movies) == 0:
            self.logger.info("All movies already processed!")
            return self.load_results()
        
        self.logger.info(f"Processing {len(remaining_movies)} remaining movies")
        
        # Process in batches
        all_enriched = []
        all_failed = []
        
        start_batch = progress['last_batch_index']
        total_batches = (len(remaining_movies) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(start_batch, total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(remaining_movies))
            
            movies_batch = remaining_movies.iloc[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(movies_batch)} movies)")
            
            # Enrich batch
            enriched_batch, failed_batch = self.enrich_movie_batch(movies_batch)
            
            all_enriched.extend(enriched_batch)
            all_failed.extend(failed_batch)
            
            # Update progress
            progress['processed_movies'].extend(movies_batch['movieId'].tolist())
            progress['last_batch_index'] = batch_idx + 1
            progress['total_enriched'] = len(all_enriched)
            progress['total_failed'] = len(all_failed)
            progress['last_updated'] = datetime.now().isoformat()
            
            self.save_progress(progress)
            
            # Save intermediate results
            if len(all_enriched) > 0:
                self.save_intermediate_results(all_enriched, all_failed)
            
            # Delay between batches to respect rate limits
            if batch_idx < total_batches - 1:  # Don't delay after last batch
                self.logger.info(f"Waiting {self.delay_between_batches} seconds before next batch...")
                time.sleep(self.delay_between_batches)
        
        # Save final results
        return self.save_final_results(all_enriched, all_failed)
    
    def save_intermediate_results(self, enriched_movies, failed_movies):
        """Save intermediate results as checkpoint"""
        if enriched_movies:
            enriched_df = pd.DataFrame(enriched_movies)
            enriched_df.to_csv(self.checkpoint_file, index=False)
    
    def save_final_results(self, enriched_movies, failed_movies):
        """Save final enrichment results"""
        # Save successful enrichments
        if enriched_movies:
            enriched_df = pd.DataFrame(enriched_movies)
            enriched_path = self.enriched_dir / "movies_enriched.csv"
            enriched_df.to_csv(enriched_path, index=False)
            self.logger.info(f"💾 Enriched data saved to {enriched_path}")
        else:
            enriched_df = pd.DataFrame()
        
        # Save failed searches
        if failed_movies:
            failed_df = pd.DataFrame(failed_movies)
            failed_path = self.enriched_dir / "failed_searches.csv"
            failed_df.to_csv(failed_path, index=False)
            self.logger.info(f"⚠️  Failed searches saved to {failed_path}")
        
        # Generate and save summary
        summary = self.generate_summary(enriched_movies, failed_movies)
        summary_path = self.enriched_dir / "enrichment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("📊 ENRICHMENT SUMMARY:")
        self.logger.info(f"  Total processed: {summary['total_processed']}")
        self.logger.info(f"  Successfully enriched: {summary['successful_enrichments']}")
        self.logger.info(f"  Failed searches: {summary['failed_searches']}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
        
        return enriched_df, failed_movies
    
    def generate_summary(self, enriched_movies, failed_movies):
        """Generate enrichment summary statistics"""
        total_processed = len(enriched_movies) + len(failed_movies)
        
        summary = {
            'total_processed': total_processed,
            'successful_enrichments': len(enriched_movies),
            'failed_searches': len(failed_movies),
            'success_rate': (len(enriched_movies) / total_processed * 100) if total_processed > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'enrichment_date': datetime.now().strftime('%Y-%m-%d'),
            'failure_reasons': {}
        }
        
        # Analyze failure reasons
        if failed_movies:
            failure_reasons = {}
            for failed in failed_movies:
                reason = failed['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            summary['failure_reasons'] = failure_reasons
        
        # Movie statistics
        if enriched_movies:
            enriched_df = pd.DataFrame(enriched_movies)
            
            summary['content_statistics'] = {
                'movies_with_overview': int((enriched_df['overview'] != '').sum()),
                'movies_with_cast': int((enriched_df['cast'].astype(str) != '[]').sum()),
                'movies_with_directors': int((enriched_df['directors'].astype(str) != '[]').sum()),
                'movies_with_genres': int((enriched_df['genres_tmdb'].astype(str) != '[]').sum()),
                'movies_with_budget': int((enriched_df['budget'] > 0).sum()),
                'movies_with_revenue': int((enriched_df['revenue'] > 0).sum()),
                'avg_vote_average': float(enriched_df['vote_average'].mean()) if 'vote_average' in enriched_df else 0,
                'avg_vote_count': float(enriched_df['vote_count'].mean()) if 'vote_count' in enriched_df else 0
            }
        
        return summary
    
    def load_results(self):
        """Load existing enrichment results"""
        enriched_path = self.enriched_dir / "movies_enriched.csv"
        if enriched_path.exists():
            enriched_df = pd.read_csv(enriched_path)
            self.logger.info(f"Loaded {len(enriched_df)} enriched movies from {enriched_path}")
            return enriched_df, []
        else:
            raise FileNotFoundError("No enriched data found. Run enrichment first.")

def main():
    parser = argparse.ArgumentParser(description='Enrich MovieLens data with TMDB metadata')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of movies to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of movies per batch')
    parser.add_argument('--delay', type=int, default=10,
                       help='Delay between batches (seconds)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start from beginning, ignore previous progress')
    
    args = parser.parse_args()
    
    enricher = EnhancedMovieDataEnricher(
        batch_size=args.batch_size,
        delay_between_batches=args.delay
    )
    
    try:
        # Load movies data
        movies_df = enricher.load_movies_data()
        enricher.logger.info(f"Loaded {len(movies_df)} movies for enrichment")
        
        # Run enrichment
        enriched_df, failed_movies = enricher.run_enrichment(
            movies_df,
            sample_size=args.sample_size,
            resume=not args.no_resume
        )
        
        enricher.logger.info("🎉 Data enrichment completed successfully!")
        return True
        
    except Exception as e:
        enricher.logger.error(f"❌ Enrichment failed: {e}")
        enricher.logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
