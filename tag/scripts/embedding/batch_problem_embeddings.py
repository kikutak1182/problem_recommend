#!/usr/bin/env python3
"""
Batch Problem Embeddings Generator

Pre-computes embeddings for all problems to avoid real-time API calls during inference.
"""

import json
import os
import sys
import pickle
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import openai

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config
from scripts.editorial_text_extractor import EditorialTextExtractor

class ProblemEmbeddingBatcher:
    """Batch generate embeddings for all problems"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=inference_config.openai_api_key)
        self.text_extractor = EditorialTextExtractor()
        self.logger = self._setup_logger()
        
        # Paths
        self.embeddings_cache_path = os.path.join(
            inference_config.base_dir, "vectors", "problem_embeddings.pkl"
        )
        self.mappings_path = os.path.join(
            inference_config.base_dir, "editorial_crawler", "data", "editorial_mappings.json"
        )
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('problem_embedding_batcher')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_existing_cache(self) -> Dict[str, np.ndarray]:
        """Load existing problem embeddings cache"""
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    cache = pickle.load(f)
                self.logger.info(f"Loaded {len(cache)} cached problem embeddings")
                return cache
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        return {}
    
    def load_editorial_mappings(self) -> Dict:
        """Load editorial mappings"""
        with open(self.mappings_path, 'r', encoding='utf-8') as f:
            return json.load(f)['editorial_mappings']
    
    def create_problem_embedding(self, problem_title: str, editorial_text: str) -> Optional[np.ndarray]:
        """Create embedding for problem + editorial"""
        
        # Combine problem and editorial text
        combined_text = f"問題: {problem_title}\n解説: {editorial_text}"
        
        # Truncate if too long
        max_length = 8000
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=[combined_text]
            )
            
            return np.array(response.data[0].embedding)
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {e}")
            return None
    
    def batch_generate_embeddings(self, problem_ids: List[str] = None, force_update: bool = False):
        """Generate embeddings for all problems in batch"""
        
        # Load existing cache
        cache = {} if force_update else self.load_existing_cache()
        
        # Load problem mappings
        mappings = self.load_editorial_mappings()
        
        # Determine which problems to process
        if problem_ids is None:
            # Process all ABC175+ problems
            problem_ids = [pid for pid in mappings.keys() 
                          if pid.startswith(('abc175_', 'abc176_', 'abc177_', 'abc178_', 'abc179_', 'abc180_'))]
        
        # Filter out already cached problems
        if not force_update:
            problem_ids = [pid for pid in problem_ids if pid not in cache]
        
        self.logger.info(f"Processing {len(problem_ids)} problems for embedding generation")
        
        total_processed = 0
        total_failed = 0
        
        for i, problem_id in enumerate(problem_ids, 1):
            if problem_id not in mappings:
                self.logger.warning(f"Problem {problem_id} not found in mappings")
                continue
                
            self.logger.info(f"Processing {i}/{len(problem_ids)}: {problem_id}")
            
            try:
                # Get editorial text
                editorial_url = mappings[problem_id].get('editorial_url', '')
                if not editorial_url:
                    self.logger.warning(f"No editorial URL for {problem_id}")
                    continue
                
                self.logger.info(f"  Extracting editorial text...")
                editorial_text = self.text_extractor.extract_editorial_text(editorial_url)
                
                if not editorial_text:
                    self.logger.warning(f"Failed to extract editorial for {problem_id}")
                    total_failed += 1
                    continue
                
                # Create embedding
                self.logger.info(f"  Creating embedding...")
                embedding = self.create_problem_embedding(problem_id, editorial_text)
                
                if embedding is not None:
                    cache[problem_id] = embedding
                    total_processed += 1
                    self.logger.info(f"  ✓ Success ({len(editorial_text)} chars)")
                    
                    # Save cache periodically
                    if total_processed % 5 == 0:
                        self._save_cache(cache)
                        self.logger.info(f"  💾 Saved {len(cache)} embeddings to cache")
                else:
                    total_failed += 1
                    self.logger.error(f"  ❌ Failed to create embedding")
                    
            except Exception as e:
                self.logger.error(f"Error processing {problem_id}: {e}")
                total_failed += 1
        
        # Final save
        self._save_cache(cache)
        
        self.logger.info(f"Batch processing complete!")
        self.logger.info(f"  Total processed: {total_processed}")
        self.logger.info(f"  Total failed: {total_failed}")
        self.logger.info(f"  Total in cache: {len(cache)}")
        
        return cache
    
    def _save_cache(self, cache: Dict[str, np.ndarray]):
        """Save cache to disk"""
        os.makedirs(os.path.dirname(self.embeddings_cache_path), exist_ok=True)
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache, f)
    
    def verify_cache(self):
        """Verify cached embeddings"""
        cache = self.load_existing_cache()
        
        print("Problem Embeddings Cache Status:")
        print(f"  Total problems: {len(cache)}")
        
        if cache:
            sample_embedding = list(cache.values())[0]
            print(f"  Embedding dimension: {sample_embedding.shape}")
            print(f"  Sample problems: {list(cache.keys())[:5]}")

if __name__ == "__main__":
    batcher = ProblemEmbeddingBatcher()
    
    print("Problem Embedding Batch Generator")
    print("=" * 50)
    
    # Check existing cache
    batcher.verify_cache()
    
    # Generate embeddings for ABC175-180
    target_problems = []
    for contest_num in range(175, 181):
        for problem_char in ['c', 'd', 'e', 'f']:
            target_problems.append(f'abc{contest_num}_{problem_char}')
    
    print(f"\\nGenerating embeddings for {len(target_problems)} problems...")
    cache = batcher.batch_generate_embeddings(target_problems)
    
    print("\\n" + "=" * 50)
    print("✅ Batch processing completed!")
    batcher.verify_cache()