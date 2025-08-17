#!/usr/bin/env python3
"""
Cost-Efficient Batch Tag Processor

Integrates system message caching and Batch API for maximum cost reduction.
Provides both immediate processing (with caching) and delayed batch processing.
"""

import json
import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.batch_tag_processor import BatchTagProcessor
from scripts.batch_api_processor import BatchAPIProcessor
from scripts.tag_inference_config import inference_config

class CostEfficientProcessor:
    """Cost-efficient processor with caching and batch API options"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.regular_processor = BatchTagProcessor()
        self.batch_processor = BatchAPIProcessor()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('cost_efficient_processor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_immediate_with_caching(self, **kwargs) -> Dict:
        """Process immediately with system message caching (30-50% cost reduction)"""
        
        self.logger.info("🚀 Processing with system message caching...")
        self.logger.info("💰 Expected cost reduction: 30-50%")
        
        # Use the regular processor (now with caching enabled)
        return self.regular_processor.process_problems(**kwargs)
    
    def process_batch_api(self, contest_types: Optional[List[str]] = None,
                         limit: Optional[int] = None,
                         min_difficulty: Optional[int] = None,
                         max_difficulty: Optional[int] = None,
                         contest_range: Optional[str] = None) -> str:
        """Process using Batch API for 50% cost reduction (24h delay)"""
        
        self.logger.info("📦 Processing with Batch API...")
        self.logger.info("💰 Expected cost reduction: 50%")
        self.logger.info("⏰ Processing time: 24 hours")
        
        # Load problems to process
        editorial_mappings = self.regular_processor.load_editorial_mappings()
        existing_tags = self.regular_processor.load_existing_tags()
        
        if not editorial_mappings:
            return "No editorial mappings found"
        
        # Filter problems
        problems_to_process = self.regular_processor.filter_problems_for_processing(
            editorial_mappings, existing_tags, contest_types, limit, True, 
            min_difficulty, max_difficulty, contest_range
        )
        
        if not problems_to_process:
            return "No problems to process"
        
        # Create and submit batch
        requests = self.batch_processor.create_batch_requests(problems_to_process)
        
        if not requests:
            return "No valid batch requests created"
        
        batch_id = self.batch_processor.submit_batch(requests)
        
        self.logger.info(f"✅ Batch submitted successfully: {batch_id}")
        self.logger.info(f"📊 Problems in batch: {len(requests)}")
        self.logger.info(f"⏳ Check status with: check_batch_status('{batch_id}')")
        
        return batch_id
    
    def check_batch_status(self, batch_id: str) -> Dict:
        """Check batch processing status"""
        return self.batch_processor.check_batch_status(batch_id)
    
    def complete_batch_processing(self, batch_id: str) -> Dict:
        """Download and integrate batch results"""
        
        self.logger.info(f"📥 Downloading batch results: {batch_id}")
        
        # Download results
        results_filepath = self.batch_processor.download_batch_results(batch_id)
        
        if not results_filepath:
            return {"error": "Failed to download batch results"}
        
        # Process results
        batch_results = self.batch_processor.process_batch_results(results_filepath)
        
        if not batch_results:
            return {"error": "No valid results processed"}
        
        # Integrate with existing database
        existing_tags = self.regular_processor.load_existing_tags()
        
        # Merge results
        all_problems = {**existing_tags, **batch_results}
        
        # Save updated database
        metadata = {
            "total_problems": len(all_problems),
            "last_updated": datetime.now().isoformat(),
            "inference_model": inference_config.model_name,
            "inference_method": "batch_api_processing",
            "batch_id": batch_id,
            "batch_processed_count": len(batch_results)
        }
        
        self.regular_processor.save_results(all_problems, metadata)
        
        self.logger.info(f"✅ Batch processing completed")
        self.logger.info(f"📊 Processed: {len(batch_results)} problems")
        self.logger.info(f"📊 Total database: {len(all_problems)} problems")
        
        return {
            "processed": len(batch_results),
            "total": len(all_problems),
            "batch_id": batch_id
        }

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description='Cost-Efficient Tag Processor')
    
    # Processing mode
    parser.add_argument('--mode', choices=['immediate', 'batch', 'check', 'complete'], 
                       default='immediate', help='Processing mode')
    parser.add_argument('--batch-id', type=str, help='Batch ID for check/complete operations')
    
    # Standard parameters
    parser.add_argument('--contest-types', type=str, default='abc',
                       help='Contest types to process (comma-separated)')
    parser.add_argument('--limit', type=int, help='Limit number of problems to process')
    parser.add_argument('--min-difficulty', type=int,
                       help='Skip problems with difficulty <= this value')
    parser.add_argument('--max-difficulty', type=int,
                       help='Skip problems with difficulty >= this value')
    parser.add_argument('--contest-range', type=str,
                       help='Contest number range (e.g., "247-300")')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel workers (immediate mode only)')
    
    args = parser.parse_args()
    
    processor = CostEfficientProcessor()
    
    if args.mode == 'immediate':
        print("🚀 Starting immediate processing with system message caching...")
        print("💰 Cost reduction: 30-50%")
        print("⏰ Processing time: Real-time")
        print()
        
        contest_types = [ct.strip() for ct in args.contest_types.split(',')]
        
        results = processor.process_immediate_with_caching(
            contest_types=contest_types,
            limit=args.limit,
            min_difficulty=args.min_difficulty,
            max_difficulty=args.max_difficulty,
            contest_range=args.contest_range,
            parallel_workers=args.parallel
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return 1
        else:
            print(f"✅ Processing completed successfully")
            print(f"📊 Results: {results}")
    
    elif args.mode == 'batch':
        print("📦 Starting batch API processing...")
        print("💰 Cost reduction: 50%")
        print("⏰ Processing time: 24 hours")
        print()
        
        contest_types = [ct.strip() for ct in args.contest_types.split(',')]
        
        batch_id = processor.process_batch_api(
            contest_types=contest_types,
            limit=args.limit,
            min_difficulty=args.min_difficulty,
            max_difficulty=args.max_difficulty,
            contest_range=args.contest_range
        )
        
        print(f"📋 Batch ID: {batch_id}")
        print(f"📝 Save this batch ID to check status later!")
        print(f"🔍 Check status: python {__file__} --mode check --batch-id {batch_id}")
    
    elif args.mode == 'check':
        if not args.batch_id:
            print("❌ Error: --batch-id required for check mode")
            return 1
        
        print(f"🔍 Checking batch status: {args.batch_id}")
        status = processor.check_batch_status(args.batch_id)
        
        print(f"📊 Status: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
        if status.get('status') == 'completed':
            print()
            print("✅ Batch completed! Ready to download results.")
            print(f"📥 Download: python {__file__} --mode complete --batch-id {args.batch_id}")
    
    elif args.mode == 'complete':
        if not args.batch_id:
            print("❌ Error: --batch-id required for complete mode")
            return 1
        
        print(f"📥 Completing batch processing: {args.batch_id}")
        results = processor.complete_batch_processing(args.batch_id)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return 1
        else:
            print(f"✅ Batch processing completed successfully")
            print(f"📊 Results: {results}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())