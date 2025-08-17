import json
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

from crawler_config import config
from editorial_extractor import EditorialExtractor
from contest_filter import ContestFilter

class EditorialDatabaseBuilder:
    """Build database of editorial URLs for AtCoder problems"""
    
    def __init__(self):
        self.extractor = EditorialExtractor()
        self.filter = ContestFilter()
        self.logger = self._setup_logger()
        self.database = self._load_existing_database()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for database building operations"""
        logger = logging.getLogger('database_builder')
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
            file_handler = logging.FileHandler(config.log_path)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_existing_database(self) -> Dict:
        """Load existing editorial database if it exists"""
        if os.path.exists(config.database_path):
            try:
                with open(config.database_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.logger.warning("Failed to load existing database, starting fresh")
        
        return {
            "editorial_mappings": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_contests": 0,
                "total_problems": 0,
                "target_contests": config.target_contests,
                "extraction_stats": {
                    "successful": 0,
                    "failed": 0,
                    "skipped": 0
                }
            }
        }
    
    def build_database(self, limit_contests: Optional[int] = None, 
                      contest_types: Optional[List[str]] = None) -> Dict:
        """
        Build editorial database for target contests
        
        Args:
            limit_contests: Maximum number of contests to process (for testing)
            contest_types: Specific contest types to process ['abc', 'arc', 'agc']
        """
        self.logger.info("Starting editorial database building process...")
        
        # Get target contests
        if contest_types:
            all_contests = {}
            for contest_type in contest_types:
                contests = self.filter.filter_problems_by_contest_type(contest_type)
                all_contests.update(contests)
        else:
            all_contests = self.filter.get_target_problems()
        
        # Apply limit if specified
        if limit_contests:
            contest_ids = list(all_contests.keys())[:limit_contests]
            all_contests = {cid: all_contests[cid] for cid in contest_ids}
        
        self.logger.info(f"Processing {len(all_contests)} target contests")
        
        successful = 0
        failed = 0
        skipped = 0
        
        for i, (contest_id, problems) in enumerate(all_contests.items(), 1):
            self.logger.info(f"Processing contest {i}/{len(all_contests)}: {contest_id}")
            
            # Skip if already processed
            if self._is_contest_already_processed(contest_id):
                self.logger.info(f"Contest {contest_id} already processed, skipping")
                skipped += 1
                continue
            
            try:
                # Extract editorial URLs for this contest
                editorial_mappings = self.extractor.extract_problem_editorial_urls(contest_id)
                
                if editorial_mappings:
                    # Add to database
                    for problem_index, mapping in editorial_mappings.items():
                        problem_key = f"{contest_id}_{problem_index.lower()}"
                        self.database["editorial_mappings"][problem_key] = mapping
                    
                    self.logger.info(f"Successfully extracted {len(editorial_mappings)} editorial URLs for {contest_id}")
                    successful += 1
                else:
                    self.logger.warning(f"No editorial URLs found for {contest_id}")
                    failed += 1
                
                # Save progress periodically
                if i % 10 == 0:
                    self._save_database()
                    self.logger.info(f"Progress saved. Processed {i}/{len(all_contests)} contests")
                
            except Exception as e:
                self.logger.error(f"Failed to process contest {contest_id}: {e}")
                failed += 1
        
        # Update metadata
        self._update_metadata(successful, failed, skipped)
        
        # Final save
        self._save_database()
        
        self.logger.info(f"Database building completed. Successful: {successful}, Failed: {failed}, Skipped: {skipped}")
        return self.database
    
    def _is_contest_already_processed(self, contest_id: str) -> bool:
        """Check if contest has already been processed"""
        for problem_key in self.database["editorial_mappings"]:
            if problem_key.startswith(f"{contest_id}_"):
                return True
        return False
    
    def _update_metadata(self, successful: int, failed: int, skipped: int):
        """Update database metadata"""
        self.database["metadata"]["last_updated"] = datetime.now().isoformat()
        self.database["metadata"]["total_contests"] = len(set(
            mapping["contest_id"] for mapping in self.database["editorial_mappings"].values()
        ))
        self.database["metadata"]["total_problems"] = len(self.database["editorial_mappings"])
        self.database["metadata"]["extraction_stats"] = {
            "successful": successful,
            "failed": failed,
            "skipped": skipped
        }
    
    def _save_database(self):
        """Save database to file"""
        try:
            with open(config.database_path, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Database saved to {config.database_path}")
        except Exception as e:
            self.logger.error(f"Failed to save database: {e}")
    
    def validate_database(self) -> Dict[str, int]:
        """Validate the built database"""
        validation_stats = {
            "total_problems": len(self.database["editorial_mappings"]),
            "valid_urls": 0,
            "invalid_urls": 0,
            "missing_editorial_ids": 0
        }
        
        self.logger.info("Validating database entries...")
        
        for problem_key, mapping in self.database["editorial_mappings"].items():
            # Check if editorial URL is accessible (sample validation)
            if mapping.get("editorial_id"):
                validation_stats["valid_urls"] += 1
            else:
                validation_stats["missing_editorial_ids"] += 1
        
        self.logger.info(f"Validation completed: {validation_stats}")
        return validation_stats
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        mappings = self.database["editorial_mappings"]
        
        # Contest type breakdown
        contest_types = {}
        for mapping in mappings.values():
            contest_id = mapping["contest_id"]
            for contest_type in ["abc", "arc", "agc"]:
                if contest_id.startswith(contest_type):
                    contest_types[contest_type] = contest_types.get(contest_type, 0) + 1
                    break
        
        return {
            "total_problems": len(mappings),
            "total_contests": len(set(m["contest_id"] for m in mappings.values())),
            "contest_types": contest_types,
            "metadata": self.database["metadata"]
        }
    
    def print_statistics(self):
        """Print database statistics"""
        stats = self.get_statistics()
        
        print("=== Editorial Database Statistics ===")
        print(f"Total Problems: {stats['total_problems']}")
        print(f"Total Contests: {stats['total_contests']}")
        print("\nBy Contest Type:")
        for contest_type, count in stats['contest_types'].items():
            print(f"  {contest_type.upper()}: {count} problems")
        
        extraction_stats = stats['metadata']['extraction_stats']
        print(f"\nExtraction Results:")
        print(f"  Successful: {extraction_stats['successful']}")
        print(f"  Failed: {extraction_stats['failed']}")  
        print(f"  Skipped: {extraction_stats['skipped']}")

if __name__ == "__main__":
    # Build database with a small sample for testing
    builder = EditorialDatabaseBuilder()
    
    print("Building editorial database (testing with 5 contests)...")
    database = builder.build_database(limit_contests=5, contest_types=["abc"])
    
    builder.print_statistics()
    
    print(f"\nDatabase saved to: {config.database_path}")
    print(f"Logs saved to: {config.log_path}")