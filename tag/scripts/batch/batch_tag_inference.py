#!/usr/bin/env python3
"""
Batch Tag Inference for ABC175-179

Runs tag inference on all cached problem embeddings with real editorial text.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fast_inference_system import FastTagInference
from scripts.editorial_text_extractor import EditorialTextExtractor
from scripts.tag_inference_config import inference_config

def load_editorial_mappings() -> Dict:
    """Load editorial URL mappings"""
    mappings_path = os.path.join(
        inference_config.base_dir, "editorial_crawler", "data", "editorial_mappings.json"
    )
    with open(mappings_path, 'r', encoding='utf-8') as f:
        return json.load(f)['editorial_mappings']

def get_cached_problem_list() -> List[str]:
    """Get list of problems with cached embeddings"""
    import pickle
    
    embeddings_path = os.path.join(
        inference_config.base_dir, "vectors", "problem_embeddings.pkl"
    )
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        return sorted(embeddings.keys())
    except FileNotFoundError:
        print("❌ No problem embeddings found. Run batch_problem_embeddings.py first.")
        return []

def batch_inference_abc175_179():
    """Run batch inference on ABC175-179 problems"""
    
    print("🚀 ABC175-179 Batch Tag Inference")
    print("=" * 60)
    
    # Get cached problems
    cached_problems = get_cached_problem_list()
    abc175_179 = [p for p in cached_problems if any(p.startswith(f'abc{i}_') for i in range(175, 180))]
    
    print(f"📋 Found {len(abc175_179)} cached problems:")
    for i, problem_id in enumerate(abc175_179, 1):
        print(f"  {i:2d}. {problem_id}")
    
    if not abc175_179:
        print("❌ No ABC175-179 problems found in cache")
        return
    
    # Initialize systems
    print("\n🔧 Initializing inference system...")
    inference_system = FastTagInference()
    text_extractor = EditorialTextExtractor()
    mappings = load_editorial_mappings()
    
    # Results tracking
    results = {}
    successful = 0
    failed = 0
    total_time = 0
    
    start_time = datetime.now()
    
    print(f"\n🎯 Processing {len(abc175_179)} problems...")
    print("-" * 60)
    
    for i, problem_id in enumerate(abc175_179, 1):
        print(f"\n[{i:2d}/{len(abc175_179)}] Processing {problem_id}...")
        
        try:
            # Get editorial text
            if problem_id not in mappings:
                print(f"  ❌ No mapping found for {problem_id}")
                failed += 1
                continue
            
            editorial_url = mappings[problem_id].get('editorial_url', '')
            if not editorial_url:
                print(f"  ❌ No editorial URL for {problem_id}")
                failed += 1
                continue
            
            print(f"  📄 Extracting editorial text...")
            editorial_text = text_extractor.extract_editorial_text(editorial_url)
            
            if not editorial_text:
                print(f"  ❌ Failed to extract editorial text")
                failed += 1
                continue
            
            print(f"  ✓ Extracted {len(editorial_text)} characters")
            
            # Run inference
            print(f"  🤖 Running tag inference...")
            problem_start = datetime.now()
            
            result = inference_system.infer_tags_for_problem_id(
                problem_id=problem_id,
                problem_title=f"Problem {problem_id}",
                editorial_text=editorial_text
            )
            
            problem_end = datetime.now()
            processing_time = (problem_end - problem_start).total_seconds()
            total_time += processing_time
            
            if result:
                results[problem_id] = result
                successful += 1
                
                print(f"  ✅ Success ({processing_time:.1f}s)")
                print(f"     Tags: {result['tags']}")
                print(f"     Confidence: {[f'{c:.3f}' for c in result['confidence_scores']]}")
                print(f"     Average: {result['avg_confidence']:.3f}")
                print(f"     Status: {'[LOW]' if result['low_confidence'] else '[OK]'}")
                
                # Show top component scores
                detailed = result['detailed_scores']
                if detailed:
                    top_tag_id = result['tag_ids'][0]
                    components = detailed[top_tag_id]
                    print(f"     Top tag breakdown: self={components['self_confidence']:.3f}, "
                          f"verif={components['verifier_score']:.3f}, "
                          f"embed={components['embedding_similarity']:.3f}, "
                          f"rule={components['rule_based_score']:.3f}")
                
            else:
                print(f"  ❌ Inference failed ({processing_time:.1f}s)")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BATCH INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total problems processed: {len(abc175_179)}")
    print(f"Successful inferences: {successful}")
    print(f"Failed inferences: {failed}")
    print(f"Success rate: {successful/len(abc175_179)*100:.1f}%")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Average time per problem: {total_time/max(successful,1):.1f}s")
    print(f"Total API processing time: {total_time:.1f}s")
    
    if results:
        # Analyze results by contest
        contests = {}
        for problem_id, result in results.items():
            contest = problem_id.split('_')[0].upper()
            if contest not in contests:
                contests[contest] = []
            contests[contest].append((problem_id, result))
        
        print(f"\n📋 Results by Contest:")
        for contest in sorted(contests.keys()):
            contest_results = contests[contest]
            avg_conf = sum(r[1]['avg_confidence'] for r in contest_results) / len(contest_results)
            print(f"  {contest}: {len(contest_results)} problems, avg confidence: {avg_conf:.3f}")
        
        # Show confidence distribution
        all_confidences = [r['avg_confidence'] for r in results.values()]
        high_conf = sum(1 for c in all_confidences if c >= 0.7)
        med_conf = sum(1 for c in all_confidences if 0.5 <= c < 0.7)
        low_conf = sum(1 for c in all_confidences if c < 0.5)
        
        print(f"\n📈 Confidence Distribution:")
        print(f"  High (≥0.7): {high_conf} problems")
        print(f"  Medium (0.5-0.7): {med_conf} problems")  
        print(f"  Low (<0.5): {low_conf} problems")
        
        # Show some examples
        print(f"\n🎯 Top Results:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_confidence'], reverse=True)
        for problem_id, result in sorted_results[:5]:
            print(f"  {problem_id}: {result['tags']} ({result['avg_confidence']:.3f})")
    
    return results

if __name__ == "__main__":
    results = batch_inference_abc175_179()
    print(f"\n🎉 Batch processing completed!")
    if results:
        print(f"Results have been automatically saved to the database.")
    print(f"Use test_results_viewer.py to analyze the results in detail.")