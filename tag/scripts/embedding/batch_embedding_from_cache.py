#!/usr/bin/env python3
"""
Batch Embedding Generator from Cache

Uses pre-cached problem + editorial text to generate embeddings via OpenAI Batch API.
Provides 50% cost reduction compared to real-time API calls.
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

class BatchEmbeddingFromCache:
    """Generate embeddings from cached text using Batch API"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=inference_config.openai_api_key)
        self.logger = self._setup_logger()
        
        # Paths
        self.text_cache_path = os.path.join(
            inference_config.base_dir, "data", "problem_combined_text_cache.json"
        )
        self.embeddings_cache_path = os.path.join(
            inference_config.base_dir, "vectors", "problem_embeddings.pkl"
        )
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('batch_embedding_from_cache')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_text_cache(self) -> Dict[str, Dict[str, str]]:
        """Load cached problem + editorial texts"""
        try:
            with open(self.text_cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            self.logger.info(f"Loaded {len(cache)} cached problem texts")
            return cache
        except FileNotFoundError:
            self.logger.error(f"Text cache not found: {self.text_cache_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load text cache: {e}")
            return {}
    
    def load_existing_embeddings(self) -> Dict[str, np.ndarray]:
        """Load existing embeddings cache"""
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                self.logger.info(f"Loaded {len(cache)} existing embeddings")
                return cache
            except Exception as e:
                self.logger.warning(f"Failed to load embeddings cache: {e}")
        return {}
    
    def prepare_batch_requests(self, text_cache: Dict, existing_embeddings: Dict) -> List[Dict]:
        """Prepare batch embedding requests from text cache"""
        
        requests = []
        
        for problem_id, text_data in text_cache.items():
            # Skip if embedding already exists
            if problem_id in existing_embeddings:
                continue
            
            # Get combined text
            combined_text = text_data.get('combined_text')
            if not combined_text:
                self.logger.warning(f"No combined text for {problem_id}")
                continue
            
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
        
        return requests
    
    def submit_batch(self, requests: List[Dict]) -> str:
        """Submit batch embedding requests"""
        
        if not requests:
            self.logger.info("No requests to submit")
            return None
        
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
                "description": "Problem embedding generation from cache",
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
            "batch_file": batch_filepath,
            "type": "embeddings"
        }
        
        batch_info_path = os.path.join(
            inference_config.base_dir, 
            "data", 
            f"embedding_batch_{batch.id}_info.json"
        )
        
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch info saved: {batch_info_path}")
        
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
                counts = batch.request_counts.__dict__
                self.logger.info(f"Requests: {counts}")
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to check batch status: {e}")
            return {}
    
    def process_batch_results(self, batch_id: str) -> Dict[str, np.ndarray]:
        """Process completed batch results and update embeddings cache"""
        
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
        new_embeddings = {}
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
                    new_embeddings[problem_id] = np.array(embedding_data)
                    success_count += 1
                else:
                    error_count += 1
                    error_msg = result.get('error', {}).get('message', 'Unknown error')
                    self.logger.warning(f"Failed embedding for {problem_id}: {error_msg}")
                    
            except Exception as e:
                self.logger.error(f"Failed to parse result line: {e}")
                error_count += 1
                continue
        
        self.logger.info(f"Processed batch results: {success_count} success, {error_count} errors")
        
        # Update embeddings cache
        if new_embeddings:
            existing_embeddings = self.load_existing_embeddings()
            existing_embeddings.update(new_embeddings)
            
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(existing_embeddings, f)
            
            self.logger.info(f"Updated embeddings cache with {len(new_embeddings)} new embeddings")
            self.logger.info(f"Total embeddings in cache: {len(existing_embeddings)}")
        
        return new_embeddings
    
    def generate_embeddings_from_cache(self):
        """Generate embeddings for all cached text data"""
        
        self.logger.info("Starting embedding generation from cache")
        
        # Load text cache
        text_cache = self.load_text_cache()
        if not text_cache:
            self.logger.error("No text cache available")
            return None
        
        # Load existing embeddings
        existing_embeddings = self.load_existing_embeddings()
        
        # Prepare batch requests
        self.logger.info("Preparing batch requests...")
        requests = self.prepare_batch_requests(text_cache, existing_embeddings)
        
        if not requests:
            self.logger.info("All problems already have embeddings")
            return None
        
        self.logger.info(f"Prepared {len(requests)} embedding requests")
        
        # Submit batch
        batch_id = self.submit_batch(requests)
        
        return batch_id
    
    def get_cache_status(self):
        """Show cache status"""
        text_cache = self.load_text_cache()
        embeddings_cache = self.load_existing_embeddings()
        
        self.logger.info(f"Cache Status:")
        self.logger.info(f"  Text cache entries: {len(text_cache)}")
        self.logger.info(f"  Embedding cache entries: {len(embeddings_cache)}")
        
        if text_cache:
            # Check coverage
            text_problems = set(text_cache.keys())
            embedding_problems = set(embeddings_cache.keys())
            
            missing_embeddings = text_problems - embedding_problems
            self.logger.info(f"  Problems with text but no embedding: {len(missing_embeddings)}")
            
            if text_cache and embedding_problems:
                coverage = len(embedding_problems & text_problems) / len(text_cache) * 100
                self.logger.info(f"  Embedding coverage: {coverage:.1f}%")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Embedding from Cache")
    parser.add_argument('--generate', action='store_true', help='Generate embeddings from cache')
    parser.add_argument('--check-status', type=str, help='Check status of batch ID')
    parser.add_argument('--process-results', type=str, help='Process results of batch ID')
    parser.add_argument('--status', action='store_true', help='Show cache status')
    
    args = parser.parse_args()
    
    generator = BatchEmbeddingFromCache()
    
    if args.status:
        generator.get_cache_status()
    elif args.check_status:
        generator.check_batch_status(args.check_status)
    elif args.process_results:
        generator.process_batch_results(args.process_results)
    elif args.generate:
        batch_id = generator.generate_embeddings_from_cache()
        if batch_id:
            print(f"\nBatch submitted successfully!")
            print(f"Batch ID: {batch_id}")
            print(f"\nTo check status:")
            print(f"python3 {__file__} --check-status {batch_id}")
            print(f"\nTo process results (after completion):")
            print(f"python3 {__file__} --process-results {batch_id}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()