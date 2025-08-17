#!/usr/bin/env python3
"""
Test difficulty filtering functionality without requiring OpenAI API
"""

import json
import os
import sys

# Import the batch processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config

class MockBatchTagProcessor:
    """Mock processor for testing difficulty filtering without OpenAI"""
    
    def __init__(self):
        self.problem_metadata = self._load_problem_metadata()
        
    def _load_problem_metadata(self):
        """Load problem metadata including difficulty"""
        try:
            metadata_path = os.path.join(inference_config.base_dir, "data", "problem_metadata.json")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Problem metadata file not found, difficulty filtering disabled")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in problem metadata: {e}")
            return {}
    
    def load_editorial_mappings(self):
        """Load editorial mappings from crawler database"""
        try:
            with open(inference_config.editorial_mappings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Editorial mappings file not found: {inference_config.editorial_mappings_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in editorial mappings: {e}")
            return {}
    
    def load_existing_tags(self):
        """Load existing problems with tags"""
        try:
            with open(inference_config.problems_with_tags_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("problems", {})
        except FileNotFoundError:
            print("ℹ️ No existing problems_with_tags.json found, starting fresh")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in problems_with_tags.json: {e}")
            return {}
    
    def filter_problems_for_processing(self, editorial_mappings, existing_tags,
                                     contest_types=None, limit=None, skip_existing=True, 
                                     min_difficulty=None):
        """Filter problems that need tag inference - copy of original logic"""
        
        problems_to_process = []
        mappings = editorial_mappings.get("editorial_mappings", {})
        
        skipped_count = {"already_processed": 0, "contest_type": 0, "difficulty": 0}
        
        for problem_key, problem_data in mappings.items():
            # Skip if already processed and skip_existing is True
            if skip_existing and problem_key in existing_tags:
                skipped_count["already_processed"] += 1
                continue
            
            # Filter by contest type if specified
            if contest_types:
                contest_id = problem_data.get("contest_id", "")
                if not any(contest_id.startswith(ct) for ct in contest_types):
                    skipped_count["contest_type"] += 1
                    continue
            
            # Filter by difficulty if specified
            if min_difficulty is not None and self.problem_metadata:
                metadata = self.problem_metadata.get(problem_key, {})
                difficulty = metadata.get("difficulty")
                if difficulty is not None and difficulty <= min_difficulty:
                    skipped_count["difficulty"] += 1
                    continue
                elif difficulty is None:
                    pass  # Include problems without difficulty data
            
            # Convert to processing format
            processing_data = {
                "problem_id": problem_key,
                "contest_id": problem_data.get("contest_id", ""),
                "problem_index": problem_data.get("problem_index", ""),
                "title": problem_data.get("title", ""),
                "problem_url": problem_data.get("problem_url", ""),
                "editorial_url": problem_data.get("editorial_url", ""),
                "editorial_id": problem_data.get("editorial_id"),
            }
            
            problems_to_process.append(processing_data)
            
            # Apply limit if specified
            if limit and len(problems_to_process) >= limit:
                break
        
        # Log filtering statistics
        total_skipped = sum(skipped_count.values())
        if total_skipped > 0:
            print(f"   Filtered out {total_skipped} problems:")
            for reason, count in skipped_count.items():
                if count > 0:
                    print(f"     {reason}: {count}")
        
        return problems_to_process

def test_difficulty_filtering():
    """Test the difficulty filtering feature"""
    
    print("=== Testing Difficulty Filtering ===")
    
    # Create mock processor
    processor = MockBatchTagProcessor()
    
    # Load data manually
    editorial_mappings = processor.load_editorial_mappings()
    existing_tags = processor.load_existing_tags()
    
    if not editorial_mappings:
        print("❌ No editorial mappings found")
        return
    
    print(f"📊 Total problems in editorial database: {len(editorial_mappings.get('editorial_mappings', {}))}")
    print(f"📊 Existing problems with tags: {len(existing_tags)}")
    
    # Test different difficulty thresholds
    thresholds = [None, 0, 200, 500, 1000]
    
    for threshold in thresholds:
        print(f"\n🧪 Testing with min_difficulty = {threshold}")
        
        # Filter problems
        filtered_problems = processor.filter_problems_for_processing(
            editorial_mappings=editorial_mappings,
            existing_tags=existing_tags,
            contest_types=['abc'],  # Focus on ABC
            limit=None,
            skip_existing=True,
            min_difficulty=threshold
        )
        
        print(f"   Found {len(filtered_problems)} problems to process")
        
        # Show some examples
        if len(filtered_problems) > 0:
            print(f"   First 3 examples:")
            for i, problem in enumerate(filtered_problems[:3]):
                problem_id = problem['problem_id']
                
                # Get difficulty from metadata
                if processor.problem_metadata:
                    metadata = processor.problem_metadata.get(problem_id, {})
                    difficulty = metadata.get('difficulty', 'N/A')
                    print(f"     {problem_id}: difficulty = {difficulty}")
                else:
                    print(f"     {problem_id}: no metadata available")

def analyze_difficulty_distribution():
    """Analyze the distribution of difficulties"""
    
    print("\n=== Difficulty Distribution Analysis ===")
    
    processor = MockBatchTagProcessor()
    
    if not processor.problem_metadata:
        print("❌ No problem metadata found")
        return
    
    # Collect difficulties
    difficulties = []
    contest_difficulties = {'abc': [], 'arc': [], 'agc': []}
    
    for problem_id, metadata in processor.problem_metadata.items():
        difficulty = metadata.get('difficulty')
        if difficulty is not None:
            difficulties.append(difficulty)
            
            # Categorize by contest type
            for contest_type in ['abc', 'arc', 'agc']:
                if problem_id.startswith(contest_type):
                    contest_difficulties[contest_type].append(difficulty)
                    break
    
    difficulties.sort()
    
    print(f"📊 Total problems with difficulty data: {len(difficulties)}")
    print(f"📊 Difficulty range: {min(difficulties)} to {max(difficulties)}")
    
    # Show percentiles
    percentiles = [10, 25, 50, 75, 90, 95]
    for p in percentiles:
        idx = int(len(difficulties) * p / 100)
        if idx < len(difficulties):
            print(f"📊 {p}th percentile: {difficulties[idx]}")
    
    # Count problems by threshold
    thresholds = [0, 200, 400, 800, 1200, 1600, 2000]
    print(f"\n📊 Problems above each threshold:")
    for threshold in thresholds:
        count = sum(1 for d in difficulties if d > threshold)
        print(f"   > {threshold}: {count} problems")
    
    # By contest type
    print(f"\n📊 By contest type:")
    for contest_type, contest_diffs in contest_difficulties.items():
        if contest_diffs:
            avg_diff = sum(contest_diffs) / len(contest_diffs)
            print(f"   {contest_type.upper()}: {len(contest_diffs)} problems, avg difficulty = {avg_diff:.0f}")

if __name__ == "__main__":
    test_difficulty_filtering()
    analyze_difficulty_distribution()