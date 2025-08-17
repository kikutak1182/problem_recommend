#!/usr/bin/env python3
"""
Batch Tag Processor - Process multiple problems for tag inference

This script processes problems from the editorial crawler database
and generates tags using o4-mini model.
"""

import json
import os
import sys
import logging
from typing import Dict, List, Optional
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.enhanced_tag_inference import EnhancedTagInference
from scripts.tag_inference_config import inference_config

class BatchTagProcessor:
    """Batch processor for tag inference"""
    
    def __init__(self):
        self.inference_engine = EnhancedTagInference()
        self.logger = self._setup_logger()
        self.problem_metadata = self._load_problem_metadata()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for batch processing"""
        logger = logging.getLogger('batch_tag_processor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = os.path.join(inference_config.base_dir, "tag_inference.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_problem_metadata(self) -> Dict:
        """Load problem metadata including difficulty"""
        try:
            metadata_path = os.path.join(inference_config.base_dir, "data", "problem_metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("Problem metadata file not found, difficulty filtering disabled")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in problem metadata: {e}")
            return {}
    
    def load_editorial_mappings(self) -> Dict:
        """Load editorial mappings from crawler database"""
        try:
            with open(inference_config.editorial_mappings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Editorial mappings file not found: {inference_config.editorial_mappings_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in editorial mappings: {e}")
            return {}
    
    def load_existing_tags(self) -> Dict:
        """Load existing problems with tags"""
        try:
            with open(inference_config.problems_with_tags_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("problems", {})
        except FileNotFoundError:
            self.logger.info("No existing problems_with_tags.json found, starting fresh")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in problems_with_tags.json: {e}")
            return {}
    
    def save_results(self, problems_data: Dict, metadata: Dict):
        """Save results to problems_with_tags.json"""
        
        output_data = {
            "problems": problems_data,
            "metadata": {
                **metadata,
                "last_updated": datetime.now().isoformat(),
                "inference_model": inference_config.model_name,
                "inference_method": "enhanced_composite_confidence",
                "total_problems": len(problems_data)
            }
        }
        
        try:
            with open(inference_config.problems_with_tags_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to {inference_config.problems_with_tags_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def filter_problems_for_processing(self, editorial_mappings: Dict, 
                                     existing_tags: Dict,
                                     contest_types: Optional[List[str]] = None,
                                     limit: Optional[int] = None,
                                     skip_existing: bool = True,
                                     min_difficulty: Optional[int] = None,
                                     max_difficulty: Optional[int] = None,
                                     contest_range: Optional[str] = None) -> List[Dict]:
        """Filter problems that need tag inference"""
        
        problems_to_process = []
        mappings = editorial_mappings.get("editorial_mappings", {})
        
        # Parse contest range if specified
        contest_range_filter = None
        if contest_range:
            try:
                start_num, end_num = map(int, contest_range.split('-'))
                contest_range_filter = (start_num, end_num)
                self.logger.info(f"Filtering contests by range: {start_num}-{end_num}")
            except ValueError:
                self.logger.warning(f"Invalid contest range format: {contest_range}, ignoring")
        
        skipped_count = {"already_processed": 0, "contest_type": 0, "contest_range": 0, "difficulty": 0}
        
        for problem_key, problem_data in mappings.items():
            # Skip if already processed and skip_existing is True
            if skip_existing and problem_key in existing_tags:
                skipped_count["already_processed"] += 1
                continue
            
            # Filter by contest type if specified
            contest_id = problem_data.get("contest_id", "")
            if contest_types:
                if not any(contest_id.startswith(ct) for ct in contest_types):
                    skipped_count["contest_type"] += 1
                    continue
            
            # Filter by contest range if specified
            if contest_range_filter:
                start_num, end_num = contest_range_filter
                # Extract contest number (e.g., "abc175" -> 175)
                import re
                match = re.match(r'([a-z]+)(\d+)', contest_id)
                if match:
                    contest_num = int(match.group(2))
                    if not (start_num <= contest_num <= end_num):
                        skipped_count["contest_range"] += 1
                        continue
                else:
                    # If we can't parse the contest number, skip it when range filter is active
                    skipped_count["contest_range"] += 1
                    continue
            
            # Filter by difficulty if specified
            if (min_difficulty is not None or max_difficulty is not None) and self.problem_metadata:
                metadata = self.problem_metadata.get(problem_key, {})
                difficulty = metadata.get("difficulty")
                if difficulty is not None:
                    if min_difficulty is not None and difficulty <= min_difficulty:
                        skipped_count["difficulty"] += 1
                        self.logger.debug(f"Skipping {problem_key}: difficulty {difficulty} <= {min_difficulty}")
                        continue
                    if max_difficulty is not None and difficulty >= max_difficulty:
                        skipped_count["difficulty"] += 1
                        self.logger.debug(f"Skipping {problem_key}: difficulty {difficulty} >= {max_difficulty}")
                        continue
                elif difficulty is None:
                    self.logger.debug(f"No difficulty data for {problem_key}, including in processing")
            
            # Convert to processing format
            processing_data = {
                "problem_id": problem_key,
                "contest_id": problem_data.get("contest_id", ""),
                "problem_index": problem_data.get("problem_index", ""),
                "title": problem_data.get("title", ""),
                "problem_url": problem_data.get("problem_url", ""),
                "editorial_url": problem_data.get("editorial_url", ""),
                "editorial_id": problem_data.get("editorial_id"),
            }
            
            problems_to_process.append(processing_data)
            
            # Apply limit if specified
            if limit and len(problems_to_process) >= limit:
                break
        
        # Log filtering statistics
        total_skipped = sum(skipped_count.values())
        if total_skipped > 0:
            self.logger.info(f"Filtered out {total_skipped} problems:")
            for reason, count in skipped_count.items():
                if count > 0:
                    self.logger.info(f"  {reason}: {count}")
        
        return problems_to_process
    
    def process_single_problem(self, problem_data: Dict) -> tuple:
        """Process a single problem and return (problem_id, result)"""
        problem_id = problem_data.get("problem_id", "unknown")
        try:
            result = self.inference_engine.infer_tags_for_problem(problem_data)
            return problem_id, result
        except Exception as e:
            self.logger.error(f"Error processing {problem_id}: {e}")
            return problem_id, None
    
    def process_problems_parallel(self, problems_to_process: List[Dict], 
                                 max_workers: int = 2) -> Dict[str, Dict]:
        """Process problems in parallel using ThreadPoolExecutor"""
        results = {}
        
        # Limit max workers to avoid API rate limits
        max_workers = min(max_workers, 5)
        
        self.logger.info(f"Processing {len(problems_to_process)} problems with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all problems for processing
            future_to_problem = {
                executor.submit(self.process_single_problem, problem): problem
                for problem in problems_to_process
            }
            
            # Process completed futures as they finish
            for i, future in enumerate(as_completed(future_to_problem), 1):
                problem = future_to_problem[future]
                problem_id = problem.get("problem_id", f"problem_{i}")
                
                try:
                    problem_id, result = future.result()
                    
                    if result:
                        results[problem_id] = result
                        
                        # Log results with confidence details
                        tags_with_conf = [
                            f"{tag}({conf:.2f})" 
                            for tag, conf in zip(result['tags'], result['confidence_scores'])
                        ]
                        conf_status = "[LOW]" if result['low_confidence'] else "[OK]"
                        
                        self.logger.info(f"✓ {problem_id}: {tags_with_conf} "
                                       f"avg={result['avg_confidence']:.2f} {conf_status} "
                                       f"({i}/{len(problems_to_process)})")
                    else:
                        self.logger.warning(f"✗ Failed to process {problem_id} ({i}/{len(problems_to_process)})")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {problem_id}: {e}")
                
                # Add small delay to respect API rate limits
                time.sleep(0.5)
        
        return results
    
    def process_problems(self, 
                        contest_types: Optional[List[str]] = None,
                        limit: Optional[int] = None,
                        skip_existing: bool = True,
                        min_difficulty: Optional[int] = None,
                        max_difficulty: Optional[int] = None,
                        contest_range: Optional[str] = None,
                        parallel_workers: int = 1) -> Dict:
        """
        Main processing function
        
        Args:
            contest_types: List of contest types to process (e.g., ['abc', 'arc'])
            limit: Maximum number of problems to process
            skip_existing: Whether to skip problems that already have tags
            min_difficulty: Minimum difficulty threshold (skip problems <= this value)
            max_difficulty: Maximum difficulty threshold (skip problems >= this value)
            
        Returns:
            Processing statistics
        """
        
        self.logger.info("Starting batch tag processing...")
        
        # Load data
        editorial_mappings = self.load_editorial_mappings()
        if not editorial_mappings:
            return {"error": "No editorial mappings found"}
        
        existing_tags = self.load_existing_tags()
        
        # Filter problems to process
        problems_to_process = self.filter_problems_for_processing(
            editorial_mappings, existing_tags, contest_types, limit, skip_existing, min_difficulty, max_difficulty, contest_range
        )
        
        if not problems_to_process:
            self.logger.info("No problems to process")
            return {"processed": 0, "skipped": len(existing_tags)}
        
        self.logger.info(f"Processing {len(problems_to_process)} problems...")
        
        # Combine existing tags with new results
        all_problems = existing_tags.copy()
        
        # Use parallel processing if requested
        if parallel_workers > 1:
            self.logger.info(f"Using parallel processing with {parallel_workers} workers")
            batch_results = self.process_problems_parallel(problems_to_process, parallel_workers)
        else:
            # Use original batch processing
            batch_results = self.inference_engine.process_batch(problems_to_process)
        
        # Process results
        successful = 0
        failed = 0
        
        for problem in problems_to_process:
            problem_id = problem.get("problem_id", "")
            result = batch_results.get(problem_id)
            
            if result:
                # Merge with existing problem data if available
                existing_data = existing_tags.get(problem_id, {})
                
                updated_data = {
                    **existing_data,
                    "contest_id": problem.get("contest_id", ""),
                    "problem_index": problem.get("problem_index", ""),
                    "title": problem.get("title", ""),
                    "problem_url": problem.get("problem_url", ""),
                    "editorial_url": problem.get("editorial_url", ""),
                    "editorial_id": problem.get("editorial_id"),
                    **result  # Add inference results
                }
                
                all_problems[problem_id] = updated_data
                successful += 1
                
                # For non-parallel mode, log results here
                if parallel_workers <= 1:
                    tags_with_conf = [
                        f"{tag}({conf:.2f})" 
                        for tag, conf in zip(result['tags'], result['confidence_scores'])
                    ]
                    avg_conf = result.get('avg_confidence', 0)
                    conf_status = " [LOW]" if result.get('low_confidence', False) else " [OK]"
                    
                    self.logger.info(f"✓ {problem_id}: {tags_with_conf} avg={avg_conf:.2f}{conf_status}")
            else:
                failed += 1
                if parallel_workers <= 1:
                    self.logger.warning(f"✗ Failed to process {problem_id}")
        
        # Final save
        final_metadata = {
            "processing_completed": True,
            "total_processed_in_session": successful + failed,
            "successful_in_session": successful,
            "failed_in_session": failed,
            "processing_stats": {
                "contest_types": contest_types,
                "limit_applied": limit,
                "skip_existing": skip_existing,
                "min_difficulty": min_difficulty,
                "max_difficulty": max_difficulty
            }
        }
        
        self.save_results(all_problems, final_metadata)
        
        self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return {
            "processed": successful + failed,
            "successful": successful,
            "failed": failed,
            "total_problems": len(all_problems)
        }
    
    def print_statistics(self):
        """Print current database statistics"""
        try:
            with open(inference_config.problems_with_tags_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problems = data.get("problems", {})
            metadata = data.get("metadata", {})
            
            print("=== Tag Inference Database Statistics ===")
            print(f"Total Problems: {len(problems)}")
            
            # Count by contest type
            contest_counts = {}
            tagged_counts = {}
            
            for problem_id, problem_data in problems.items():
                contest_id = problem_data.get("contest_id", "")
                for contest_type in ["abc", "arc", "agc"]:
                    if contest_id.startswith(contest_type):
                        contest_counts[contest_type] = contest_counts.get(contest_type, 0) + 1
                        if "tags" in problem_data and problem_data["tags"]:
                            tagged_counts[contest_type] = tagged_counts.get(contest_type, 0) + 1
                        break
            
            print("\nBy Contest Type:")
            for contest_type in ["abc", "arc", "agc"]:
                total = contest_counts.get(contest_type, 0)
                tagged = tagged_counts.get(contest_type, 0)
                print(f"  {contest_type.upper()}: {tagged}/{total} tagged")
            
            if metadata:
                print(f"\nLast Updated: {metadata.get('last_updated', 'Unknown')}")
                print(f"Inference Model: {metadata.get('inference_model', 'Unknown')}")
                print(f"Inference Method: {metadata.get('inference_method', 'Unknown')}")
            
        except FileNotFoundError:
            print("No tag inference database found")
        except Exception as e:
            print(f"Error reading statistics: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Batch Tag Processor for AtCoder Problems")
    
    parser.add_argument('--contest-types', type=str, default='abc,arc,agc',
                       help='Contest types to process (comma-separated)')
    parser.add_argument('--limit', type=int, help='Limit number of problems to process')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip problems that already have tags')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Reprocess all problems (overrides --skip-existing)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics without processing')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only 3 problems')
    parser.add_argument('--min-difficulty', type=int,
                       help='Skip problems with difficulty <= this value')
    parser.add_argument('--max-difficulty', type=int,
                       help='Skip problems with difficulty >= this value')
    parser.add_argument('--contest-range', type=str,
                       help='Contest number range (e.g., "175-199" for abc175-abc199)')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel workers (default: 1, max: 5)')
    
    args = parser.parse_args()
    
    processor = BatchTagProcessor()
    
    # Show statistics
    if args.stats_only:
        processor.print_statistics()
        return
    
    # Parse contest types
    contest_types = [ct.strip() for ct in args.contest_types.split(',')]
    
    # Set processing parameters
    limit = 3 if args.test else args.limit
    skip_existing = not args.force_reprocess
    parallel_workers = min(max(1, args.parallel), 5)  # Clamp between 1 and 5
    
    if args.test:
        print("🧪 Running in TEST mode (3 problems only)")
    
    print(f"Processing configuration:")
    print(f"  Contest types: {contest_types}")
    print(f"  Limit: {limit}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Parallel workers: {parallel_workers}")
    if args.min_difficulty:
        print(f"  Min difficulty: {args.min_difficulty}")
    if args.max_difficulty:
        print(f"  Max difficulty: {args.max_difficulty}")
    if args.contest_range:
        print(f"  Contest range: {args.contest_range}")
    
    # Run processing
    results = processor.process_problems(
        contest_types=contest_types,
        limit=limit, 
        skip_existing=skip_existing,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        contest_range=args.contest_range,
        parallel_workers=parallel_workers
    )
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return 1
    
    print(f"\n✅ Processing completed:")
    print(f"  Processed: {results['processed']}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total in database: {results['total_problems']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())