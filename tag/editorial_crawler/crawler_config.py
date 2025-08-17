import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime

@dataclass
class CrawlerConfig:
    """Editorial crawler configuration"""
    
    # Target contest thresholds
    target_contests: Dict[str, int] = None
    
    # Request settings
    user_agent: str = "AtCoder Editorial Crawler (Educational Purpose)"
    request_delay: float = 1.0  # seconds between requests
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Editorial URL patterns
    editorial_lang: str = "ja"  # Japanese language parameter (?lang=ja)
    
    # Output settings
    output_dir: str = os.path.join(os.path.dirname(__file__), "data")
    database_file: str = "editorial_mappings.json"
    log_file: str = "crawler.log"
    
    def __post_init__(self):
        if self.target_contests is None:
            self.target_contests = {
                "abc": 175,  # ABC175以降
                "arc": 104,  # ARC104以降  
                "agc": 48    # AGC048以降
            }
        
        # Create output directory if not exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def database_path(self) -> str:
        return os.path.join(self.output_dir, self.database_file)
    
    @property
    def log_path(self) -> str:
        return os.path.join(self.output_dir, self.log_file)
    
    def is_target_contest(self, contest_id: str) -> bool:
        """Check if contest is target for crawling"""
        for contest_type, min_number in self.target_contests.items():
            if contest_id.startswith(contest_type):
                try:
                    number = int(contest_id[len(contest_type):])
                    return number >= min_number
                except ValueError:
                    continue
        return False

# Global configuration instance
config = CrawlerConfig()