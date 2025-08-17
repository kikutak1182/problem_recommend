#!/usr/bin/env python3
"""
Test specific problems with enhanced confidence system
"""

import json
import sys
import os
from datetime import datetime

# Import our systems
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.enhanced_tag_inference import EnhancedTagInference
from scripts.tag_inference_config import inference_config

def load_editorial_mappings():
    """Load editorial mappings database"""
    mappings_path = os.path.join(
        inference_config.base_dir, 
        "editorial_crawler", "data", "editorial_mappings.json"
    )
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['editorial_mappings']

def test_specific_problems(problem_ids):
    """Test specific problems by their IDs"""
    
    print(f"=== Testing {len(problem_ids)} specific problems ===")
    
    # Load mappings
    mappings = load_editorial_mappings()
    
    # Create inference engine
    inference_engine = EnhancedTagInference()
    
    results = {}
    total_start_time = datetime.now()
    
    for i, problem_id in enumerate(problem_ids, 1):
        if problem_id not in mappings:
            print(f"❌ {problem_id}: NOT FOUND in database")
            continue
        
        problem_data = mappings[problem_id]
        problem_data['problem_id'] = problem_id
        
        print(f"\\n🔍 Processing {i}/{len(problem_ids)}: {problem_id}")
        print(f"   Title: {problem_data.get('title', 'Unknown')}")
        print(f"   Editorial: {problem_data.get('editorial_url', '')}")
        
        start_time = datetime.now()
        result = inference_engine.infer_tags_for_problem(problem_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if result:
            results[problem_id] = result
            
            # Display results
            print(f"✅ Success ({processing_time:.1f}s)")
            print(f"   Tags: {result['tags']}")
            print(f"   Confidence scores: {[f'{c:.2f}' for c in result['confidence_scores']]}")
            print(f"   Average confidence: {result['avg_confidence']:.3f}")
            print(f"   Status: {'[LOW]' if result['low_confidence'] else '[OK]'}")
            
            # Show detailed confidence breakdown
            print("   Detailed breakdown:")
            for tag_id, details in result['detailed_confidence'].items():
                comp = details['components']
                print(f"     {details['tag_name']}: {details['composite_confidence']:.3f}")
                print(f"       Self: {comp['self_confidence']:.3f}, "
                      f"Verifier: {comp['verifier_score']:.3f}, "
                      f"Embed: {comp['embedding_similarity']:.3f}")
                print(f"       Reasoning: {details['reasoning']}")
        else:
            print(f"❌ Failed ({processing_time:.1f}s)")
    
    total_time = (datetime.now() - total_start_time).total_seconds()
    successful = len(results)
    
    print(f"\\n=== Summary ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per problem: {total_time/len(problem_ids):.1f}s")
    print(f"Successful: {successful}/{len(problem_ids)}")
    
    # Save results to main database
    if results:
        save_results_to_database(results)
        print(f"✅ Results saved to problems_with_tags.json")
    
    return results

def save_results_to_database(new_results):
    """Save test results to the main database"""
    
    # Load existing data
    problems_with_tags_path = os.path.join(
        inference_config.base_dir, "data", "problems_with_tags.json"
    )
    
    try:
        with open(problems_with_tags_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"problems": {}, "metadata": {}}
    
    # Update with new results
    existing_problems = data.get("problems", {})
    
    for problem_id, result in new_results.items():
        # Convert to storage format
        storage_data = {
            "tags": result["tags"],
            "tag_ids": result.get("tag_ids", []),
            "confidence_scores": result.get("confidence_scores", []),
            "avg_confidence": result["avg_confidence"],
            "min_confidence": result.get("min_confidence", 0),
            "confidence_std": result.get("confidence_std", 0),
            "low_confidence": result["low_confidence"],
            "detailed_confidence": result.get("detailed_confidence", {}),
            "candidate_tags_count": result.get("candidate_tags_count", 0),
            "editorial_text_length": result.get("editorial_text_length", 0),
            "inferred_at": result["inferred_at"],
            "model": result["model"],
            "method": result["method"]
        }
        
        existing_problems[problem_id] = storage_data
    
    # Update metadata
    data["problems"] = existing_problems
    data["metadata"] = {
        **data.get("metadata", {}),
        "last_updated": datetime.now().isoformat(),
        "inference_method": "enhanced_composite_confidence",
        "inference_model": inference_config.model_name,
        "total_problems": len(existing_problems)
    }
    
    # Save back to file
    with open(problems_with_tags_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Test ABC176 C,D,E,F problems
    target_problems = ['abc176_c', 'abc176_d', 'abc176_e', 'abc176_f']
    
    print("Testing ABC176 C,D,E,F problems with enhanced confidence system...")
    results = test_specific_problems(target_problems)
    
    print("\\n🎉 Testing completed!")