#!/usr/bin/env python3
"""
Enhanced Tag Inference with Composite Confidence System

Combines embedding-based filtering with multi-component confidence scoring.
"""

import json
import os
import sys
import logging
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

# Import configurations and systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.confidence_system import ConfidenceSystem
from scripts.editorial_text_extractor import EditorialTextExtractor
from scripts.tag_inference_config import inference_config

class EnhancedTagInference:
    """Enhanced tag inference using composite confidence system"""
    
    def __init__(self):
        self.confidence_system = ConfidenceSystem()
        self.text_extractor = EditorialTextExtractor()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for enhanced inference"""
        logger = logging.getLogger('enhanced_tag_inference')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def infer_tags_for_problem(self, problem_data: Dict) -> Optional[Dict]:
        """
        Infer tags for a single problem using enhanced confidence system
        
        Args:
            problem_data: Problem data containing editorial_url, title, etc.
            
        Returns:
            Dict with inferred tags and detailed confidence metrics
        """
        try:
            # Extract editorial text
            editorial_url = problem_data.get("editorial_url", "")
            if not editorial_url:
                self.logger.warning(f"No editorial URL for problem {problem_data.get('problem_index', 'unknown')}")
                return None
            
            editorial_text = self.text_extractor.extract_editorial_text(editorial_url)
            if not editorial_text:
                self.logger.warning(f"Failed to extract text from {editorial_url}")
                return None
            
            # Get problem info
            problem_title = problem_data.get("title", "")
            
            # Filter candidate tags using embeddings
            candidate_tags = self.confidence_system.embedding_filter.filter_candidate_tags(
                problem_title, editorial_text
            )
            
            if len(candidate_tags) < 3:
                self.logger.warning(f"Only {len(candidate_tags)} candidate tags found, need at least 3")
                return None
            
            self.logger.info(f"Using {len(candidate_tags)} candidate tags for confidence analysis")
            
            # Calculate composite confidence for top candidates
            confidence_results = self.confidence_system.calculate_composite_confidence(
                problem_title, editorial_text, candidate_tags[:8]  # Top 8 for analysis
            )
            
            if not confidence_results:
                self.logger.warning("Failed to calculate confidence scores")
                return None
            
            # Sort by composite confidence and select top 3
            sorted_tags = sorted(
                confidence_results.items(),
                key=lambda x: x[1]['composite_confidence'],
                reverse=True
            )[:3]
            
            # Extract results
            selected_tags = []
            selected_tag_ids = []
            confidence_scores = []
            detailed_confidence = {}
            
            for tag_id, confidence_data in sorted_tags:
                selected_tags.append(confidence_data['tag_name'])
                selected_tag_ids.append(tag_id)
                confidence_scores.append(confidence_data['composite_confidence'])
                detailed_confidence[tag_id] = confidence_data
            
            # Calculate overall confidence metrics
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            min_confidence = min(confidence_scores)
            confidence_std = float(np.std(confidence_scores)) if len(confidence_scores) > 1 else 0.0
            
            # Determine if confidence is low
            low_confidence = (
                avg_confidence < 0.6 or  # Low average
                min_confidence < 0.4 or  # Any very low score
                confidence_std > 0.3     # High variance
            )
            
            return {
                "tags": selected_tags,
                "confidence_scores": confidence_scores,
                "avg_confidence": float(avg_confidence),
                "low_confidence": low_confidence,
                "method": "enhanced_composite_confidence",
                "model": inference_config.model_name,
                "inferred_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error inferring tags for problem: {e}")
            return None
    
    def process_batch(self, problems: List[Dict]) -> Dict[str, Dict]:
        """Process a batch of problems with enhanced confidence"""
        
        results = {}
        
        for i, problem in enumerate(problems, 1):
            problem_id = problem.get("problem_id", f"problem_{i}")
            self.logger.info(f"Processing {i}/{len(problems)}: {problem_id}")
            
            result = self.infer_tags_for_problem(problem)
            if result:
                results[problem_id] = result
                
                # Log results with confidence details
                tags_with_conf = [
                    f"{tag}({conf:.2f})" 
                    for tag, conf in zip(result['tags'], result['confidence_scores'])
                ]
                conf_status = "[LOW]" if result['low_confidence'] else "[OK]"
                
                self.logger.info(f"✓ {problem_id}: {tags_with_conf} "
                               f"avg={result['avg_confidence']:.2f} {conf_status}")
            else:
                self.logger.warning(f"✗ Failed to process {problem_id}")
            
        return results

if __name__ == "__main__":
    
    # Test the enhanced inference system
    inference = EnhancedTagInference()
    
    # Test with sample problem data
    test_problem = {
        "problem_id": "abc175_a",
        "title": "Rainy Season",
        "editorial_url": "https://atcoder.jp/contests/abc175/editorial/51"
    }
    
    print("Testing enhanced tag inference with composite confidence...")
    result = inference.infer_tags_for_problem(test_problem)
    
    if result:
        print("Successfully inferred tags with simplified output:")
        print(f"Tags: {result['tags']}")
        print(f"Confidence scores: {result['confidence_scores']}")
        print(f"Average confidence: {result['avg_confidence']:.3f}")
        print(f"Low confidence: {result['low_confidence']}")
        print(f"Method: {result['method']}")
        print(f"Model: {result['model']}")
    else:
        print("Failed to infer tags")