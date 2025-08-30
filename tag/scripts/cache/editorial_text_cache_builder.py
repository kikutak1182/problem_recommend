#!/usr/bin/env python3
"""
Problem Text Cache Builder

Pre-fetches and caches problem text + editorial text for all problems to enable efficient batch embedding generation.
"""

import json
import os
import sys
import logging
import time
import re
import requests
from typing import Dict, List, Optional
from datetime import datetime
import concurrent.futures
from threading import Lock
from bs4 import BeautifulSoup

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config
from scripts.cache.editorial_text_extractor import EditorialTextExtractor

class ProblemTextExtractor:
    """Extract problem statement text from AtCoder problem URLs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AtCoder Problem Text Extractor (Educational Purpose)'
        })
        self.logger = logging.getLogger('problem_text_extractor')
    
    def extract_problem_text(self, problem_url: str) -> Optional[str]:
        """Extract problem statement text from problem URL"""
        try:
            self.logger.debug(f"Extracting problem text from: {problem_url}")
            response = self.session.get(problem_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find problem statement section
            problem_section = soup.find('div', {'id': 'task-statement'})
            if not problem_section:
                self.logger.warning(f"No task statement found in {problem_url}")
                return None
            
            # Extract text content, clean up
            text = problem_section.get_text(separator='\n', strip=True)
            
            # Basic cleanup
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            if len(text) > 50:  # Minimum length check
                return text
            else:
                self.logger.warning(f"Problem text too short from {problem_url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract problem text from {problem_url}: {e}")
            return None

class ProblemTextCacheBuilder:
    """Build cache of problem + editorial texts for efficient batch processing"""
    
    def __init__(self, max_workers: int = None):
        self.editorial_extractor = EditorialTextExtractor()
        self.problem_extractor = ProblemTextExtractor()
        self.logger = self._setup_logger()
        self.max_workers = max_workers or inference_config.concurrent_workers
        self.cache_lock = Lock()
        
        # Paths
        self.comprehensive_data_path = os.path.join(
            inference_config.base_dir, "editorial_crawler", "data", "comprehensive_problem_data.json"
        )
        self.combined_cache_path = os.path.join(
            inference_config.base_dir, "data", "problem_combined_text_cache.json"
        )
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('editorial_cache_builder')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_comprehensive_data(self) -> List[Dict]:
        """Load comprehensive problem data"""
        with open(self.comprehensive_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['problems']
    
    def load_existing_cache(self) -> Dict[str, Dict[str, str]]:
        """Load existing combined text cache"""
        if os.path.exists(self.combined_cache_path):
            try:
                with open(self.combined_cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                self.logger.info(f"Loaded {len(cache)} cached problem texts")
                return cache
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def save_cache(self, cache: Dict[str, Dict[str, str]]):
        """Save combined text cache to file"""
        with self.cache_lock:
            try:
                with open(self.combined_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved cache with {len(cache)} entries")
            except Exception as e:
                self.logger.error(f"Failed to save cache: {e}")
    
    def extract_combined_text_safe(self, problem: Dict, cache: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Safely extract problem + editorial text for a problem"""
        problem_id = problem['problem_id']
        
        # Check if already in cache
        if problem_id in cache:
            return cache[problem_id]
        
        # Skip if missing URLs
        editorial_url = problem.get('editorial_url')
        problem_url = problem.get('problem_url')
        
        if not editorial_url or not problem_url:
            self.logger.warning(f"Missing URLs for {problem_id}")
            return None
        
        try:
            # Extract both texts
            self.logger.info(f"Extracting problem + editorial for {problem_id}...")
            
            problem_text = self.problem_extractor.extract_problem_text(problem_url)
            editorial_text = self.editorial_extractor.extract_editorial_text(editorial_url)
            
            if problem_text and editorial_text:
                # Create combined text entry
                combined_entry = {
                    "problem_text": problem_text,
                    "editorial_text": editorial_text,
                    "combined_text": f"問題: {problem_text}\n解説: {editorial_text}"
                }
                
                # Add to cache
                with self.cache_lock:
                    cache[problem_id] = combined_entry
                
                total_chars = len(problem_text) + len(editorial_text)
                self.logger.info(f"✓ {problem_id}: {total_chars} chars (problem: {len(problem_text)}, editorial: {len(editorial_text)})")
                
                # Save cache periodically (every 10 entries)
                if len(cache) % 10 == 0:
                    self.save_cache(cache)
                
                return combined_entry
            else:
                missing = []
                if not problem_text:
                    missing.append("problem")
                if not editorial_text:
                    missing.append("editorial")
                self.logger.warning(f"✗ {problem_id}: Failed to extract {', '.join(missing)}")
                return None
                
        except Exception as e:
            self.logger.error(f"✗ {problem_id}: {e}")
            return None
    
    def build_cache_sequential(self, problems: List[Dict], start_index: int = 0):
        """Build cache sequentially (safe, slower)"""
        cache = self.load_existing_cache()
        
        # Filter problems not yet cached
        uncached_problems = [
            p for p in problems 
            if p['problem_id'] not in cache and p.get('editorial_url')
        ]
        
        if start_index > 0:
            uncached_problems = uncached_problems[start_index:]
        
        self.logger.info(f"Building cache for {len(uncached_problems)} problems (starting from index {start_index})")
        
        success_count = 0
        error_count = 0
        
        for i, problem in enumerate(uncached_problems, start_index + 1):
            result = self.extract_combined_text_safe(problem, cache)
            
            if result:
                success_count += 1
            else:
                error_count += 1
            
            # Progress report every 50 problems
            if i % 50 == 0:
                self.logger.info(f"Progress: {i}/{len(uncached_problems)} ({success_count} success, {error_count} errors)")
            
            # Be respectful to AtCoder servers
            time.sleep(0.5)
        
        # Final save
        self.save_cache(cache)
        
        self.logger.info(f"Cache building completed: {success_count} success, {error_count} errors")
        self.logger.info(f"Total cached entries: {len(cache)}")
        
        return cache
    
    def build_cache_concurrent(self, problems: List[Dict], start_index: int = 0):
        """Build cache concurrently (faster, but more server load)"""
        cache = self.load_existing_cache()
        
        # Filter problems not yet cached
        uncached_problems = [
            p for p in problems 
            if p['problem_id'] not in cache and p.get('editorial_url')
        ]
        
        if start_index > 0:
            uncached_problems = uncached_problems[start_index:]
        
        self.logger.info(f"Building cache concurrently for {len(uncached_problems)} problems")
        self.logger.info(f"Using {self.max_workers} workers")
        
        success_count = 0
        error_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.extract_combined_text_safe, problem, cache): problem
                for problem in uncached_problems
            }
            
            # Process completed tasks
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    problem = futures[future]
                    self.logger.error(f"Task failed for {problem['problem_id']}: {e}")
                    error_count += 1
                
                # Progress report
                if i % 50 == 0:
                    self.logger.info(f"Progress: {i}/{len(uncached_problems)} ({success_count} success, {error_count} errors)")
        
        # Final save
        self.save_cache(cache)
        
        self.logger.info(f"Concurrent cache building completed: {success_count} success, {error_count} errors")
        self.logger.info(f"Total cached entries: {len(cache)}")
        
        return cache
    
    def build_abc_cache(self, start_contest: int = None, end_contest: int = None, 
                       concurrent: bool = None, start_index: int = 0,
                       target_problems: List[str] = None):
        """Build editorial cache for ABC contests in range"""
        
        # Use config defaults if not specified
        if start_contest is None:
            start_contest = inference_config.default_start_contest
        if end_contest is None:
            end_contest = inference_config.default_end_contest
        if concurrent is None:
            concurrent = inference_config.use_concurrent
        if target_problems is None:
            target_problems = inference_config.target_problems
        
        self.logger.info(f"Building editorial cache for ABC{start_contest}-{end_contest}")
        self.logger.info(f"Target problems: {target_problems} (difficulty {inference_config.difficulty_threshold}+ only)")
        
        # Load comprehensive data
        problems = self.load_comprehensive_data()
        
        # Filter ABC problems in range and by target problem types
        abc_problems = [
            p for p in problems 
            if p['problem_id'].startswith('abc') and 
               start_contest <= int(p['problem_id'].split('_')[0][3:]) <= end_contest and
               p['problem_id'].split('_')[1] in target_problems  # Filter by problem type (a,b,c,d,e,f)
        ]
        
        self.logger.info(f"Found {len(abc_problems)} ABC problems in range (filtered by difficulty)")
        
        if concurrent:
            return self.build_cache_concurrent(abc_problems, start_index)
        else:
            return self.build_cache_sequential(abc_problems, start_index)
    
    def get_cache_status(self):
        """Show cache status"""
        cache = self.load_existing_cache()
        problems = self.load_comprehensive_data()
        
        # Count by contest range
        abc_problems = [p for p in problems if p['problem_id'].startswith('abc')]
        abc_cached = [pid for pid in cache.keys() if pid.startswith('abc')]
        
        self.logger.info(f"Cache Status:")
        self.logger.info(f"  Total cached: {len(cache)}")
        self.logger.info(f"  ABC problems available: {len(abc_problems)}")
        self.logger.info(f"  ABC problems cached: {len(abc_cached)}")
        if abc_problems:
            self.logger.info(f"  Coverage: {len(abc_cached)/len(abc_problems)*100:.1f}%")
        
        # Show contest range coverage
        if abc_cached:
            cached_numbers = sorted([int(pid.split('_')[0][3:]) for pid in abc_cached if pid.split('_')[0][3:].isdigit()])
            self.logger.info(f"  Cached contest range: ABC{min(cached_numbers)}-{max(cached_numbers)}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Editorial Text Cache Builder")
    parser.add_argument('--start', type=int, default=None, help=f'Start contest number (default: {inference_config.default_start_contest})')
    parser.add_argument('--end', type=int, default=None, help=f'End contest number (default: {inference_config.default_end_contest})')
    parser.add_argument('--concurrent', action='store_true', help=f'Use concurrent processing (default: {inference_config.use_concurrent})')
    parser.add_argument('--workers', type=int, default=None, help=f'Number of concurrent workers (default: {inference_config.concurrent_workers})')
    parser.add_argument('--target-problems', nargs='+', default=None, help=f'Target problem types (default: {inference_config.target_problems})')
    parser.add_argument('--start-index', type=int, default=0, help='Start from this index (for resuming)')
    parser.add_argument('--status', action='store_true', help='Show cache status')
    
    args = parser.parse_args()
    
    builder = ProblemTextCacheBuilder(max_workers=args.workers)
    
    if args.status:
        builder.get_cache_status()
    else:
        builder.build_abc_cache(
            start_contest=args.start,
            end_contest=args.end,
            concurrent=args.concurrent,
            start_index=args.start_index,
            target_problems=args.target_problems
        )


if __name__ == "__main__":
    main()