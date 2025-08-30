#!/usr/bin/env python3
"""
Keyword-based Rule System for Tag Inference

Analyzes editorial text for keyword matches with tag aliases and calculates rule-based confidence scores.
"""

import json
import re
import os
import sys
from typing import Dict, List, Set, Tuple
from collections import Counter
import logging
from dataclasses import dataclass

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config

@dataclass
class KeywordMatch:
    """Represents a keyword match with details"""
    keyword: str
    tag_id: str
    tag_name: str
    match_type: str  # "exact", "partial", "case_insensitive"
    positions: List[int]  # Character positions where found
    contexts: List[str]   # Surrounding text context

class KeywordMatcher:
    """Rule-based keyword matching system for tag inference"""
    
    def __init__(self, definitions_path: str = None):
        """Initialize with tag definitions (defaults to standard tag_definitions.json)"""
        self.logger = self._setup_logger()
        
        # Use provided path, otherwise use standard tag_definitions.json
        if definitions_path:
            self.definitions_path = definitions_path
        else:
            self.definitions_path = os.path.join(
                inference_config.base_dir, "config", "tag_definitions.json"
            )
        
        self.tag_definitions = self._load_enhanced_definitions()
        self.keyword_to_tags = self._build_keyword_index()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for keyword matching"""
        logger = logging.getLogger('keyword_matcher')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_enhanced_definitions(self) -> Dict:
        """Load enhanced tag definitions with aliases"""
        try:
            with open(self.definitions_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Enhanced definitions not found: {self.definitions_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in enhanced definitions: {e}")
            raise
    
    def _build_keyword_index(self) -> Dict[str, List[Dict]]:
        """Build index mapping keywords to tags"""
        keyword_index = {}
        
        for tag in self.tag_definitions['tags']:
            tag_id = tag['id']
            tag_name = tag['name']
            aliases = tag.get('aliases', [])
            
            # Add all aliases to index
            for alias in aliases:
                # Normalize keyword (lower case for case-insensitive matching)
                normalized_alias = alias.lower()
                
                if normalized_alias not in keyword_index:
                    keyword_index[normalized_alias] = []
                
                keyword_index[normalized_alias].append({
                    'tag_id': tag_id,
                    'tag_name': tag_name,
                    'original_alias': alias,
                    'priority': self._get_alias_priority(alias, tag_name)
                })
        
        self.logger.info(f"Built keyword index with {len(keyword_index)} unique keywords")
        return keyword_index
    
    def _get_alias_priority(self, alias: str, tag_name: str) -> int:
        """Assign priority to aliases (higher = more important)"""
        # Exact match with tag name gets highest priority
        if alias == tag_name:
            return 10
        
        # Common abbreviations get high priority
        if len(alias) <= 3 and alias.isupper():
            return 8
        
        # English terms get medium priority  
        if alias.encode('utf-8').isascii():
            return 6
        
        # Japanese terms get medium priority
        return 5
    
    def find_keyword_matches(self, editorial_text: str) -> List[KeywordMatch]:
        """Find all keyword matches in editorial text"""
        matches = []
        text_lower = editorial_text.lower()
        
        for normalized_keyword, tag_infos in self.keyword_to_tags.items():
            # Find all occurrences of this keyword
            positions = self._find_all_positions(text_lower, normalized_keyword)
            
            if positions:
                for tag_info in tag_infos:
                    # Get surrounding context for each match
                    contexts = [
                        self._get_context(editorial_text, pos, len(normalized_keyword))
                        for pos in positions
                    ]
                    
                    # Determine match type
                    original_alias = tag_info['original_alias']
                    match_type = self._determine_match_type(
                        editorial_text, original_alias, normalized_keyword
                    )
                    
                    match = KeywordMatch(
                        keyword=original_alias,
                        tag_id=tag_info['tag_id'],
                        tag_name=tag_info['tag_name'],
                        match_type=match_type,
                        positions=positions,
                        contexts=contexts
                    )
                    matches.append(match)
        
        return matches
    
    def _find_all_positions(self, text: str, keyword: str) -> List[int]:
        """Find all positions where keyword appears in text"""
        positions = []
        start = 0
        
        while True:
            pos = text.find(keyword, start)
            if pos == -1:
                break
            
            # Check word boundaries to avoid partial matches within other words
            if self._is_word_boundary_match(text, pos, len(keyword)):
                positions.append(pos)
            
            start = pos + 1
        
        return positions
    
    def _is_word_boundary_match(self, text: str, pos: int, length: int) -> bool:
        """Check if match is at word boundaries"""
        # For very short keywords (1-2 chars), require exact boundaries
        if length <= 2:
            before_ok = (pos == 0 or not text[pos-1].isalnum())
            after_ok = (pos + length >= len(text) or not text[pos + length].isalnum())
            return before_ok and after_ok
        
        # For longer keywords, allow partial word matches
        return True
    
    def _get_context(self, text: str, position: int, keyword_length: int, 
                    context_size: int = 50) -> str:
        """Get surrounding context for a keyword match"""
        start = max(0, position - context_size)
        end = min(len(text), position + keyword_length + context_size)
        
        context = text[start:end]
        
        # Highlight the matched keyword
        relative_pos = position - start
        highlighted = (
            context[:relative_pos] + 
            f"**{context[relative_pos:relative_pos + keyword_length]}**" +
            context[relative_pos + keyword_length:]
        )
        
        return highlighted
    
    def _determine_match_type(self, original_text: str, original_alias: str, 
                            normalized_keyword: str) -> str:
        """Determine the type of match"""
        # Check for exact match first
        if original_alias in original_text:
            return "exact"
        
        # For mixed Japanese/English text, be more flexible
        # Check if alias is purely ASCII (English/numbers)
        is_ascii_alias = original_alias.encode('utf-8').isascii()
        
        if is_ascii_alias:
            # For ASCII aliases, check case-insensitive match
            if original_alias.lower() in original_text.lower():
                return "case_insensitive"
        else:
            # For Japanese aliases, check direct substring match
            # (Japanese doesn't have case sensitivity concept)
            if original_alias in original_text:
                return "exact"  # Already checked above, but for clarity
        
        return "partial"
    
    def calculate_rule_based_scores(self, editorial_text: str, 
                                  candidate_tags: List[Dict]) -> Dict[str, float]:
        """Calculate rule-based confidence scores for candidate tags"""
        
        # Find all keyword matches
        matches = self.find_keyword_matches(editorial_text)
        
        # Group matches by tag
        tag_matches = {}
        for match in matches:
            tag_id = match.tag_id
            if tag_id not in tag_matches:
                tag_matches[tag_id] = []
            tag_matches[tag_id].append(match)
        
        # Calculate scores for candidate tags only
        candidate_ids = {tag['id'] for tag in candidate_tags}
        scores = {}
        
        for tag_id in candidate_ids:
            if tag_id in tag_matches:
                score = self._calculate_tag_score(tag_matches[tag_id])
                scores[tag_id] = min(1.0, score)  # Cap at 1.0
                
                # Log details for debugging
                self._log_match_details(tag_id, tag_matches[tag_id], score)
            else:
                scores[tag_id] = 0.0
        
        return scores
    
    def _calculate_tag_score(self, matches: List[KeywordMatch]) -> float:
        """Calculate score for a single tag based on its matches"""
        if not matches:
            return 0.0
        
        total_score = 0.0
        
        # Base score for having any matches (INCREASED)
        base_score = 0.5  # Increased from 0.3
        
        # Bonus for match quality (INCREASED)
        for match in matches:
            # Score based on match type (ENHANCED)
            type_scores = {
                "exact": 0.6,        # Increased from 0.4
                "case_insensitive": 0.5,  # Increased from 0.3
                "partial": 0.3       # Increased from 0.2
            }
            total_score += type_scores.get(match.match_type, 0.1)
            
            # Bonus for multiple occurrences (ENHANCED)
            occurrence_bonus = min(0.3, len(match.positions) * 0.1)  # Doubled bonus
            total_score += occurrence_bonus
        
        # Bonus for multiple different keywords matching (ENHANCED)
        unique_keywords = len(set(match.keyword for match in matches))
        keyword_variety_bonus = min(0.5, (unique_keywords - 1) * 0.2)  # Doubled bonus
        total_score += keyword_variety_bonus
        
        final_score = base_score + total_score
        return min(1.0, final_score)
    
    def _log_match_details(self, tag_id: str, matches: List[KeywordMatch], score: float):
        """Log detailed match information"""
        tag_name = matches[0].tag_name if matches else "Unknown"
        keywords = [match.keyword for match in matches]
        
        self.logger.info(f"Rule score for {tag_name} ({tag_id}): {score:.3f}")
        self.logger.info(f"  Matched keywords: {keywords}")
        
        for match in matches[:3]:  # Show first 3 contexts
            self.logger.debug(f"  Context: {match.contexts[0] if match.contexts else 'N/A'}")

if __name__ == "__main__":
    # Test the keyword matcher
    matcher = KeywordMatcher()
    
    # Test with sample text
    test_text = """
    この問題は動的計画法（DP）を使って解くことができます。
    状態遷移を考えて、各状態での最適解をメモ化していきます。
    セグメント木を使って区間クエリを効率化することも可能です。
    """
    
    # Mock candidate tags
    candidates = [
        {"id": "DP", "name": "動的計画法"},
        {"id": "SEG", "name": "セグメント木"},
        {"id": "BFS", "name": "幅優先探索"}
    ]
    
    scores = matcher.calculate_rule_based_scores(test_text, candidates)
    
    print("Rule-based scores:")
    for tag_id, score in scores.items():
        tag_info = next(tag for tag in candidates if tag['id'] == tag_id)
        print(f"  {tag_info['name']}: {score:.3f}")