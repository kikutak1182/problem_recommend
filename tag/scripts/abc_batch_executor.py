#!/usr/bin/env python3
"""
ABC250-400 Batch Executor for 7-Batch Processing

Executes tag inference for ABC250-400 problems in 7 batches of ~100 problems each.
Provides progress monitoring and batch management functionality.
"""

import json
import os
import sys
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.batch_api_processor import BatchAPIProcessor
from scripts.tag_inference_config import inference_config

class ABCBatchExecutor:
    """Executor for ABC250-400 batch processing"""
    
    def __init__(self):
        self.processor = BatchAPIProcessor()
        self.logger = self._setup_logger()
        self.batch_info = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('abc_batch_executor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_target_problems(self) -> List[Dict]:
        """Load ABC250-400 problems data"""
        data_path = os.path.join(inference_config.base_dir, "data", "abc250_400_problems.json")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                problems = json.load(f)
            
            self.logger.info(f"Loaded {len(problems)} target problems")
            return problems
            
        except FileNotFoundError:
            self.logger.error(f"Target problems file not found: {data_path}")
            return []
    
    def split_into_batches(self, problems: List[Dict], batch_size: int = 100) -> List[List[Dict]]:
        """Split problems into batches"""
        batches = []
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Split {len(problems)} problems into {len(batches)} batches")
        return batches
    
    def execute_batch(self, batch_num: int, batch_problems: List[Dict]) -> Optional[str]:
        """Execute a single batch"""
        self.logger.info(f"Starting batch {batch_num}: {len(batch_problems)} problems")
        
        try:
            # Create batch requests
            requests = self.processor.create_batch_requests(batch_problems)
            
            if not requests:
                self.logger.error(f"No valid requests created for batch {batch_num}")
                return None
            
            # Submit batch
            batch_id = self.processor.submit_batch(requests)
            
            # Store batch info
            batch_info = {
                "batch_number": batch_num,
                "batch_id": batch_id,
                "problem_count": len(batch_problems),
                "request_count": len(requests),
                "submitted_at": datetime.now().isoformat(),
                "status": "submitted",
                "problems": [p["problem_id"] for p in batch_problems]
            }
            
            self.batch_info[batch_num] = batch_info
            
            # Save batch tracking info
            tracking_path = os.path.join(
                inference_config.base_dir, 
                "data", 
                f"abc_batch_{batch_num}_tracking.json"
            )
            
            with open(tracking_path, 'w', encoding='utf-8') as f:
                json.dump(batch_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Batch {batch_num} submitted: {batch_id}")
            self.logger.info(f"Problems: {len(batch_problems)}, Requests: {len(requests)}")
            
            return batch_id
            
        except Exception as e:
            self.logger.error(f"Failed to execute batch {batch_num}: {e}")
            return None
    
    def execute_all_batches(self, delay_between_batches: int = 60):
        """Execute all 7 batches with delay between submissions"""
        
        # Load problems
        problems = self.load_target_problems()
        if not problems:
            self.logger.error("No problems to process")
            return
        
        # Split into batches
        batches = self.split_into_batches(problems)
        
        self.logger.info(f"Starting execution of {len(batches)} batches")
        self.logger.info(f"Total problems: {len(problems)}")
        self.logger.info(f"Estimated total cost: ${len(problems) * 0.005:.2f}")
        
        executed_batches = []
        
        for i, batch_problems in enumerate(batches, 1):
            batch_id = self.execute_batch(i, batch_problems)
            
            if batch_id:
                executed_batches.append((i, batch_id))
                self.logger.info(f"✓ Batch {i}/{len(batches)} submitted: {batch_id}")
                
                # Add delay between batches (except for the last one)
                if i < len(batches):
                    self.logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                    time.sleep(delay_between_batches)
            else:
                self.logger.error(f"✗ Failed to submit batch {i}")
        
        # Save summary
        summary = {
            "execution_started": datetime.now().isoformat(),
            "total_problems": len(problems),
            "total_batches": len(batches),
            "executed_batches": len(executed_batches),
            "batch_details": executed_batches,
            "estimated_cost": len(problems) * 0.005,
            "completion_expected": "Within 24 hours per batch"
        }
        
        summary_path = os.path.join(
            inference_config.base_dir,
            "data",
            f"abc_batches_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Execution summary saved: {summary_path}")
        self.logger.info(f"Executed {len(executed_batches)}/{len(batches)} batches successfully")
        
        return executed_batches
    
    def check_all_batch_status(self):
        """Check status of all executed batches"""
        
        # Find all tracking files
        data_dir = os.path.join(inference_config.base_dir, "data")
        tracking_files = [f for f in os.listdir(data_dir) if f.startswith("abc_batch_") and f.endswith("_tracking.json")]
        
        if not tracking_files:
            self.logger.info("No batch tracking files found")
            return
        
        self.logger.info(f"Checking status of {len(tracking_files)} batches")
        
        for tracking_file in sorted(tracking_files):
            tracking_path = os.path.join(data_dir, tracking_file)
            
            with open(tracking_path, 'r', encoding='utf-8') as f:
                batch_info = json.load(f)
            
            batch_num = batch_info["batch_number"]
            batch_id = batch_info["batch_id"]
            
            # Check current status
            status = self.processor.check_batch_status(batch_id)
            
            self.logger.info(f"Batch {batch_num} ({batch_id}): {status.get('status', 'unknown')}")
            if status.get('request_counts'):
                counts = status['request_counts']
                self.logger.info(f"  Progress: {counts.get('completed', 0)}/{counts.get('total', 0)}")
            
            # Update tracking file with latest status
            batch_info["last_checked"] = datetime.now().isoformat()
            batch_info["current_status"] = status.get('status', 'unknown')
            
            with open(tracking_path, 'w', encoding='utf-8') as f:
                json.dump(batch_info, f, indent=2, ensure_ascii=False)
    
    def download_completed_results(self):
        """Download results for all completed batches"""
        
        data_dir = os.path.join(inference_config.base_dir, "data")
        tracking_files = [f for f in os.listdir(data_dir) if f.startswith("abc_batch_") and f.endswith("_tracking.json")]
        
        downloaded_results = []
        
        for tracking_file in sorted(tracking_files):
            tracking_path = os.path.join(data_dir, tracking_file)
            
            with open(tracking_path, 'r', encoding='utf-8') as f:
                batch_info = json.load(f)
            
            batch_num = batch_info["batch_number"]
            batch_id = batch_info["batch_id"]
            
            # Check if completed
            status = self.processor.check_batch_status(batch_id)
            
            if status.get('status') == 'completed':
                self.logger.info(f"Downloading results for batch {batch_num}...")
                
                results_file = self.processor.download_batch_results(batch_id)
                if results_file:
                    self.logger.info(f"✓ Downloaded: {results_file}")
                    downloaded_results.append((batch_num, results_file))
                else:
                    self.logger.error(f"✗ Failed to download batch {batch_num}")
            else:
                self.logger.info(f"Batch {batch_num} not yet completed: {status.get('status')}")
        
        return downloaded_results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ABC250-400 Batch Executor")
    parser.add_argument("--action", choices=["execute", "status", "download"], required=True,
                      help="Action to perform")
    parser.add_argument("--delay", type=int, default=60,
                      help="Delay between batch submissions (seconds)")
    
    args = parser.parse_args()
    
    executor = ABCBatchExecutor()
    
    if args.action == "execute":
        print("Starting ABC250-400 batch execution...")
        executed = executor.execute_all_batches(delay_between_batches=args.delay)
        print(f"Executed {len(executed)} batches")
        
    elif args.action == "status":
        print("Checking batch status...")
        executor.check_all_batch_status()
        
    elif args.action == "download":
        print("Downloading completed results...")
        downloaded = executor.download_completed_results()
        print(f"Downloaded {len(downloaded)} batch results")

if __name__ == "__main__":
    main()