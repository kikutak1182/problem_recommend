#!/usr/bin/env python3
"""
Batch Embedding Generator for Problem Text
Uses OpenAI Batch API for 50% cost reduction on embedding generation.
"""

import json
import os
import sys
import time
import logging
import pickle
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import openai

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config
from scripts.editorial_text_extractor import EditorialTextExtractor

class BatchEmbeddingGenerator:
    """Generate embeddings using Batch API for cost efficiency"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=inference_config.openai_api_key)
        self.text_extractor = EditorialTextExtractor()
        self.logger = self._setup_logger()
        
        # Paths
        self.embeddings_cache_path = os.path.join(
            inference_config.base_dir, "vectors", "problem_embeddings.pkl"
        )
        self.comprehensive_data_path = os.path.join(
            inference_config.base_dir, "editorial_crawler", "data", "comprehensive_problem_data.json"
        )
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('batch_embedding_generator')
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
    
    def load_existing_embeddings(self) -> Dict[str, np.ndarray]:
        """Load existing embeddings cache"""
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                self.logger.info(f"Loaded {len(cache)} existing embeddings")
                return cache
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def prepare_embedding_requests(self, problems: List[Dict], existing_cache: Dict) -> List[Dict]:
        """Prepare batch embedding requests for problems not in cache"""
        
        requests = []
        
        for problem in problems:
            problem_id = problem['problem_id']
            
            # Skip if already in cache
            if problem_id in existing_cache:
                continue
                
            # Skip if no editorial URL
            if not problem.get('editorial_url'):
                self.logger.warning(f"No editorial URL for {problem_id}")
                continue
            
            try:
                # Extract editorial text
                self.logger.info(f"  Extracting editorial for {problem_id}...")
                editorial_text = self.text_extractor.extract_editorial_text(problem['editorial_url'])
                
                if not editorial_text:
                    self.logger.warning(f"Failed to extract editorial for {problem_id}")
                    continue
                
                # Create combined text
                title = problem.get('title', problem_id)
                combined_text = f"問題: {title}\n解説: {editorial_text}"
                
                # Truncate if too long
                max_length = 8000
                if len(combined_text) > max_length:
                    combined_text = combined_text[:max_length] + "..."
                
                # Create batch request
                request = {
                    "custom_id": f"embedding_{problem_id}",
                    "method": "POST", 
                    "url": "/v1/embeddings",
                    "body": {
                        "model": "text-embedding-3-small",
                        "input": combined_text
                    }
                }
                
                requests.append(request)
                
                if len(requests) % 100 == 0:
                    self.logger.info(f"Prepared {len(requests)} embedding requests...")
                
            except Exception as e:
                self.logger.error(f"Failed to prepare request for {problem_id}: {e}")
                continue
        
        return requests
    
    def submit_embedding_batch(self, requests: List[Dict]) -> str:
        """Submit batch embedding requests and return batch ID"""
        
        # Create batch file
        batch_filename = f"embedding_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        batch_filepath = os.path.join(inference_config.base_dir, "data", batch_filename)
        
        # Write requests to JSONL file
        with open(batch_filepath, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created batch file: {batch_filepath}")
        
        # Upload file to OpenAI
        batch_file = self.client.files.create(
            file=open(batch_filepath, 'rb'),
            purpose="batch"
        )
        
        self.logger.info(f"Uploaded batch file: {batch_file.id}")
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={
                "description": "Problem embedding generation",
                "created_at": datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Created embedding batch: {batch.id}")
        self.logger.info(f"Status: {batch.status}")
        self.logger.info(f"Request count: {len(requests)}")
        
        # Save batch info
        batch_info = {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": datetime.now().isoformat(),
            "request_count": len(requests),
            "batch_file": batch_filepath
        }
        
        batch_info_path = os.path.join(
            inference_config.base_dir, 
            "data", 
            f"embedding_batch_{batch.id}_info.json"
        )
        
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        return batch.id
    
    def check_batch_status(self, batch_id: str) -> Dict:
        """Check batch processing status"""
        
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            status_info = {
                "batch_id": batch_id,
                "status": batch.status,
                "request_counts": batch.request_counts.__dict__ if batch.request_counts else {},
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "expires_at": batch.expires_at,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id
            }
            
            self.logger.info(f"Batch {batch_id} status: {batch.status}")
            if batch.request_counts:
                self.logger.info(f"Requests: {batch.request_counts.__dict__}")
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to check batch status: {e}")
            return {}
    
    def process_batch_results(self, batch_id: str) -> Dict[str, np.ndarray]:
        """Process completed batch results and update cache"""
        
        # Check batch status
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            self.logger.error(f"Batch {batch_id} not completed. Status: {batch.status}")
            return {}
        
        if not batch.output_file_id:
            self.logger.error(f"No output file for batch {batch_id}")
            return {}
        
        # Download results
        result_file = self.client.files.content(batch.output_file_id)
        results_content = result_file.content.decode('utf-8')
        
        # Parse results
        embeddings = {}
        success_count = 0
        error_count = 0
        
        for line in results_content.strip().split('\n'):
            try:
                result = json.loads(line)
                custom_id = result['custom_id']
                problem_id = custom_id.replace('embedding_', '')
                
                if result.get('response') and result['response'].get('body'):
                    # Extract embedding
                    embedding_data = result['response']['body']['data'][0]['embedding']
                    embeddings[problem_id] = np.array(embedding_data)
                    success_count += 1
                else:
                    error_count += 1
                    self.logger.warning(f"Failed embedding for {problem_id}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                self.logger.error(f"Failed to parse result line: {e}")
                error_count += 1
                continue
        
        self.logger.info(f"Processed batch results: {success_count} success, {error_count} errors")
        
        # Update cache
        if embeddings:
            existing_cache = self.load_existing_embeddings()
            existing_cache.update(embeddings)
            
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(existing_cache, f)
            
            self.logger.info(f"Updated embedding cache with {len(embeddings)} new embeddings")
            self.logger.info(f"Total embeddings in cache: {len(existing_cache)}")
        
        return embeddings
    
    def generate_abc_embeddings(self, start_contest: int = 175, end_contest: int = 407):
        """Generate embeddings for ABC contests in specified range"""
        
        self.logger.info(f"Generating embeddings for ABC{start_contest}-{end_contest}")
        
        # Load comprehensive data
        problems = self.load_comprehensive_data()
        
        # Filter ABC problems in range
        abc_problems = [
            p for p in problems 
            if p['problem_id'].startswith('abc') and 
               start_contest <= int(p['problem_id'].split('_')[0][3:]) <= end_contest
        ]
        
        self.logger.info(f"Found {len(abc_problems)} ABC problems in range")
        
        # Load existing embeddings
        existing_cache = self.load_existing_embeddings()
        
        # Prepare batch requests
        self.logger.info("Preparing batch requests...")
        requests = self.prepare_embedding_requests(abc_problems, existing_cache)
        
        if not requests:
            self.logger.info("No new embeddings needed - all problems already cached")
            return None
        
        self.logger.info(f"Prepared {len(requests)} embedding requests")
        
        # Submit batch
        batch_id = self.submit_embedding_batch(requests)
        
        return batch_id


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Embedding Generator")
    parser.add_argument('--start', type=int, default=175, help='Start contest number')
    parser.add_argument('--end', type=int, default=407, help='End contest number')
    parser.add_argument('--check-status', type=str, help='Check status of batch ID')
    parser.add_argument('--process-results', type=str, help='Process results of batch ID')
    
    args = parser.parse_args()
    
    generator = BatchEmbeddingGenerator()
    
    if args.check_status:
        generator.check_batch_status(args.check_status)
    elif args.process_results:
        generator.process_batch_results(args.process_results)
    else:
        batch_id = generator.generate_abc_embeddings(args.start, args.end)
        if batch_id:
            print(f"\nBatch submitted successfully!")
            print(f"Batch ID: {batch_id}")
            print(f"\nTo check status:")
            print(f"python3 {__file__} --check-status {batch_id}")
            print(f"\nTo process results (after completion):")
            print(f"python3 {__file__} --process-results {batch_id}")


if __name__ == "__main__":
    main()