#!/usr/bin/env python3
"""
Cached Embedding-based Tag Filter

Uses pre-computed problem embeddings to avoid real-time API calls during inference.
"""

import json
import os
import sys
import numpy as np
import pickle
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config

@dataclass
class CachedEmbeddingConfig:
    """Configuration for cached embedding filter"""
    
    # Tag filtering settings
    top_k_candidates: int = 12  # Increased from 8 to capture more candidates
    similarity_threshold: float = 0.3
    
    # File paths
    @property
    def tag_definitions_path(self) -> str:
        return os.path.join(inference_config.base_dir, "config", "tag_definitions.json")
    
    @property
    def tag_embeddings_cache_path(self) -> str:
        return os.path.join(inference_config.base_dir, "vectors", "tag_embeddings.pkl")
    
    @property
    def problem_embeddings_cache_path(self) -> str:
        return os.path.join(inference_config.base_dir, "vectors", "problem_embeddings.pkl")

# Global configuration
cached_config = CachedEmbeddingConfig()

class CachedEmbeddingTagFilter:
    """Tag filtering using pre-computed embeddings (no real-time API calls)"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Load tag definitions
        self.tag_definitions = self._load_tag_definitions()
        
        # Load cached embeddings
        self.tag_embeddings = self._load_tag_embeddings()
        self.problem_embeddings = self._load_problem_embeddings()
        
        self.logger.info(f"Initialized with {len(self.tag_embeddings)} tag embeddings "
                        f"and {len(self.problem_embeddings)} problem embeddings")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('cached_embedding_filter')
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
        """Load tag definitions"""
        with open(cached_config.tag_definitions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_tag_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached tag embeddings"""
        try:
            with open(cached_config.tag_embeddings_cache_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tag embeddings cache not found: {cached_config.tag_embeddings_cache_path}")
    
    def _load_problem_embeddings(self) -> Dict[str, np.ndarray]:
        """Load cached problem embeddings"""
        try:
            with open(cached_config.problem_embeddings_cache_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Problem embeddings cache not found: {cached_config.problem_embeddings_cache_path}")
            return {}
    
    def get_problem_embedding(self, problem_id: str) -> Optional[np.ndarray]:
        """Get cached problem embedding by ID"""
        return self.problem_embeddings.get(problem_id)
    
    def filter_candidate_tags_by_id(self, problem_id: str, editorial_text: str = "") -> List[Dict]:
        """
        Filter candidate tags using cached embeddings + rule-based matching
        
        Args:
            problem_id: Problem identifier (e.g., 'abc175_c')
            editorial_text: Editorial text for rule-based matching (optional)
            
        Returns:
            List of candidate tag dictionaries with similarity scores
        """
        
        # Get cached problem embedding
        problem_embedding = self.get_problem_embedding(problem_id)
        
        if problem_embedding is None:
            self.logger.error(f"No cached embedding found for problem {problem_id}")
            return []
        
        return self._calculate_similarities_with_rules(problem_embedding, editorial_text)
    
    def filter_candidate_tags_by_embedding(self, problem_embedding: np.ndarray) -> List[Dict]:
        """
        Filter candidate tags using provided embedding
        
        Args:
            problem_embedding: Pre-computed problem embedding
            
        Returns:
            List of candidate tag dictionaries with similarity scores
        """
        return self._calculate_similarities(problem_embedding)
    
    def _calculate_similarities_with_rules(self, problem_embedding: np.ndarray, editorial_text: str = "") -> List[Dict]:
        """Calculate similarities with rule-based score boost"""
        
        # Get regular embedding-based similarities
        similarities = self._calculate_similarities_raw(problem_embedding)
        
        # If editorial text is provided, identify rule-matched tags
        rule_matched_tags = set()
        if editorial_text:
            rule_matched_tags = self._get_rule_matched_tags(editorial_text)
        
        # Apply +1.0 score boost to rule-matched tags
        for tag in similarities:
            if tag['id'] in rule_matched_tags:
                tag['similarity'] += 1.0
                self.logger.info(f"Rule boost applied to {tag['name']}: {tag['similarity']:.3f}")
        
        # Re-sort by updated similarity scores (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top K candidates
        final_candidates = similarities[:cached_config.top_k_candidates]
        
        # No need for duplicate removal or re-sorting since we already sorted after boost
        unique_candidates = final_candidates
        
        # Apply threshold filter
        filtered_candidates = [
            tag for tag in unique_candidates
            if tag['similarity'] >= cached_config.similarity_threshold
        ]
        
        if rule_matched_tags:
            rule_count = sum(1 for c in filtered_candidates if (c['similarity'] > 1.0))  # Check for boosted scores
            self.logger.info(f"Selected {len(filtered_candidates)} candidates ({rule_count} rule-boosted)")
        else:
            self.logger.info(f"Selected {len(filtered_candidates)} candidates from {len(similarities)} total")
        
        if filtered_candidates:
            top_candidate = filtered_candidates[0]
            marker = "[RULE]" if top_candidate['similarity'] > 1.0 else "[EMBED]"
            self.logger.info(f"Top candidate: {top_candidate['name']} {marker} (similarity: {top_candidate['similarity']:.3f})")
        
        return filtered_candidates
    
    def _calculate_similarities_raw(self, problem_embedding: np.ndarray) -> List[Dict]:
        """Calculate raw similarities without filtering"""
        
        similarities = []
        
        for tag in self.tag_definitions['tags']:
            tag_id = tag['id']
            
            if tag_id not in self.tag_embeddings:
                self.logger.warning(f"No embedding found for tag {tag_id}")
                continue
            
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
        
        return similarities
    
    def _get_rule_matched_tags(self, editorial_text: str) -> set:
        """Get tag IDs that match rule-based keywords"""
        try:
            # Import keyword matcher
            from scripts.filtering.keyword_matcher import KeywordMatcher
            
            matcher = KeywordMatcher()
            matches = matcher.find_keyword_matches(editorial_text)
            
            matched_tag_ids = set()
            for match in matches:
                matched_tag_ids.add(match.tag_id)
            
            if matched_tag_ids:
                self.logger.info(f"Rule-based matches found for tags: {list(matched_tag_ids)}")
            
            return matched_tag_ids
            
        except Exception as e:
            self.logger.warning(f"Failed to get rule matches: {e}")
            return set()
    
    def _calculate_similarities(self, problem_embedding: np.ndarray) -> List[Dict]:
        """Calculate similarities between problem and all tags"""
        
        similarities = []
        
        for tag in self.tag_definitions['tags']:
            tag_id = tag['id']
            
            if tag_id not in self.tag_embeddings:
                self.logger.warning(f"No embedding found for tag {tag_id}")
                continue
            
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
            if tag['similarity'] >= cached_config.similarity_threshold
        ][:cached_config.top_k_candidates]
        
        self.logger.info(f"Selected {len(candidates)} candidate tags from {len(similarities)} total")
        
        if candidates:
            self.logger.info(f"Top candidate: {candidates[0]['name']} (similarity: {candidates[0]['similarity']:.3f})")
        
        return candidates
    
    def get_embedding_similarity(self, problem_id: str, tag_id: str) -> float:
        """Get embedding similarity between problem and specific tag"""
        
        problem_embedding = self.get_problem_embedding(problem_id)
        if problem_embedding is None:
            return 0.0
        
        tag_embedding = self.tag_embeddings.get(tag_id)
        if tag_embedding is None:
            return 0.0
        
        similarity = cosine_similarity(
            problem_embedding.reshape(1, -1),
            tag_embedding.reshape(1, -1)
        )[0][0]
        
        # Normalize to 0-1 range
        return float((similarity + 1) / 2)
    
    def get_tag_info_by_id(self, tag_id: str) -> Optional[Dict]:
        """Get tag information by ID"""
        for tag in self.tag_definitions['tags']:
            if tag['id'] == tag_id:
                return tag
        return None
    
    def check_cache_status(self):
        """Check status of all caches"""
        print("Cache Status:")
        print(f"  Tag embeddings: {len(self.tag_embeddings)} cached")
        print(f"  Problem embeddings: {len(self.problem_embeddings)} cached")
        
        if self.problem_embeddings:
            print("  Sample problems:")
            for i, problem_id in enumerate(list(self.problem_embeddings.keys())[:5]):
                print(f"    {i+1}. {problem_id}")

if __name__ == "__main__":
    # Test the cached filter
    filter_system = CachedEmbeddingTagFilter()
    filter_system.check_cache_status()
    
    # Test with a sample problem if available
    if filter_system.problem_embeddings:
        test_problem_id = list(filter_system.problem_embeddings.keys())[0]
        print(f"\\nTesting with {test_problem_id}...")
        
        candidates = filter_system.filter_candidate_tags_by_id(test_problem_id)
        
        print(f"\\nTop {len(candidates)} candidates:")
        for i, candidate in enumerate(candidates, 1):
            print(f"{i}. {candidate['name']} (similarity: {candidate['similarity']:.3f})")
    else:
        print("\\nNo problem embeddings available. Run batch_problem_embeddings.py first.")