#!/usr/bin/env python3
"""
Embedding-based Tag Filtering System

Uses text-embedding-3-small to filter candidate tags based on similarity
to problem description and editorial content.
"""

import json
import os
import sys
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config

@dataclass
class EmbeddingConfig:
    """Configuration for embedding-based tag filtering"""
    
    # OpenAI Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # Default dimension for text-embedding-3-small
    
    # Tag filtering settings
    top_k_candidates: int = 8  # Number of candidate tags to select
    similarity_threshold: float = 0.3  # Minimum similarity threshold
    
    # File paths
    @property
    def tag_definitions_path(self) -> str:
        return os.path.join(inference_config.base_dir, "config", "tag_definitions.json")
    
    @property
    def tag_embeddings_cache_path(self) -> str:
        return os.path.join(inference_config.base_dir, "vectors", "tag_embeddings.pkl")
    
    @property
    def vectors_dir(self) -> str:
        return os.path.join(inference_config.base_dir, "vectors")

# Global configuration
embedding_config = EmbeddingConfig()

class EmbeddingTagFilter:
    """Tag filtering system using embeddings"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=inference_config.openai_api_key)
        self.logger = self._setup_logger()
        
        # Ensure vectors directory exists
        os.makedirs(embedding_config.vectors_dir, exist_ok=True)
        
        # Load tag definitions
        self.tag_definitions = self._load_tag_definitions()
        
        # Load or compute tag embeddings
        self.tag_embeddings = self._load_or_compute_tag_embeddings()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for embedding operations"""
        logger = logging.getLogger('embedding_tag_filter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_tag_definitions(self) -> Dict:
        """Load tag definitions from JSON file"""
        try:
            with open(embedding_config.tag_definitions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tag definitions not found: {embedding_config.tag_definitions_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tag definitions: {e}")
    
    def _load_or_compute_tag_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached tag embeddings or compute them"""
        
        cache_path = embedding_config.tag_embeddings_cache_path
        
        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache is up to date
                if len(cached_data) == len(self.tag_definitions['tags']):
                    self.logger.info(f"Loaded {len(cached_data)} tag embeddings from cache")
                    return cached_data
                else:
                    self.logger.info("Cache size mismatch, recomputing embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, recomputing embeddings")
        
        # Compute embeddings
        return self._compute_tag_embeddings()
    
    def _compute_tag_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for all tag descriptions"""
        
        self.logger.info("Computing tag embeddings...")
        embeddings = {}
        
        # Prepare texts for embedding
        tag_texts = []
        tag_ids = []
        
        for tag in self.tag_definitions['tags']:
            tag_id = tag['id']
            # Combine name and description for better context
            text = f"{tag['name']}: {tag['description']}"
            tag_texts.append(text)
            tag_ids.append(tag_id)
        
        try:
            # Get embeddings from OpenAI
            response = self.client.embeddings.create(
                model=embedding_config.embedding_model,
                input=tag_texts
            )
            
            # Store embeddings
            for i, embedding_data in enumerate(response.data):
                tag_id = tag_ids[i]
                embedding_vector = np.array(embedding_data.embedding)
                embeddings[tag_id] = embedding_vector
            
            # Cache the results
            with open(embedding_config.tag_embeddings_cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            self.logger.info(f"Computed and cached {len(embeddings)} tag embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to compute embeddings: {e}")
            raise
        
        return embeddings
    
    def get_problem_embedding(self, problem_text: str, editorial_text: str) -> np.ndarray:
        """Get embedding for problem + editorial text"""
        
        # Combine problem and editorial text
        combined_text = f"問題: {problem_text}\n解説: {editorial_text}"
        
        # Truncate if too long (embedding API has token limits)
        max_length = 8000  # Conservative limit for text-embedding-3-small
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "..."
        
        try:
            response = self.client.embeddings.create(
                model=embedding_config.embedding_model,
                input=[combined_text]
            )
            
            return np.array(response.data[0].embedding)
            
        except Exception as e:
            self.logger.error(f"Failed to get problem embedding: {e}")
            raise
    
    def filter_candidate_tags(self, problem_text: str, editorial_text: str) -> List[Dict]:
        """
        Filter candidate tags based on similarity to problem + editorial
        
        Returns:
            List of candidate tag dictionaries with similarity scores
        """
        
        # Get problem embedding
        problem_embedding = self.get_problem_embedding(problem_text, editorial_text)
        
        # Calculate similarities
        similarities = []
        
        for tag in self.tag_definitions['tags']:
            tag_id = tag['id']
            tag_embedding = self.tag_embeddings[tag_id]
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                problem_embedding.reshape(1, -1),
                tag_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                'id': tag_id,
                'name': tag['name'],
                'description': tag['description'],
                'similarity': float(similarity)
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter by threshold and select top K
        candidates = [
            tag for tag in similarities 
            if tag['similarity'] >= embedding_config.similarity_threshold
        ][:embedding_config.top_k_candidates]
        
        self.logger.info(f"Selected {len(candidates)} candidate tags from {len(similarities)} total")
        
        if candidates:
            self.logger.info(f"Top candidate: {candidates[0]['name']} (similarity: {candidates[0]['similarity']:.3f})")
        
        return candidates
    
    def get_tag_info_by_id(self, tag_id: str) -> Optional[Dict]:
        """Get tag information by ID"""
        for tag in self.tag_definitions['tags']:
            if tag['id'] == tag_id:
                return tag
        return None

if __name__ == "__main__":
    # Test the embedding filter
    filter_system = EmbeddingTagFilter()
    
    # Test with sample problem
    sample_problem = "雨が続いている日数を数える問題"
    sample_editorial = "各文字をチェックして連続するRの最大長を求める。文字列を順番に見ていき、Rが続く部分の長さを記録する。"
    
    print("Testing embedding-based tag filtering...")
    candidates = filter_system.filter_candidate_tags(sample_problem, sample_editorial)
    
    print(f"\nTop {len(candidates)} candidate tags:")
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate['name']} (ID: {candidate['id']}) - Similarity: {candidate['similarity']:.3f}")
        print(f"   {candidate['description']}")
        print()