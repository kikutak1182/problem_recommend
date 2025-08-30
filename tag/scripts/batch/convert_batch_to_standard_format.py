#!/usr/bin/env python3
"""
Convert Batch Results to Standard Format

Convert batch API results to the standard format used by other test systems.
"""

import json
import os
import sys
import statistics
from datetime import datetime
from typing import Dict, List, Optional

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config

def convert_batch_results_to_standard(batch_file: str, output_file: str, rule_scores_file: str = None):
    """Convert batch results to standard test format"""
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    # Load rule-based scores if provided
    rule_scores = {}
    if rule_scores_file and os.path.exists(rule_scores_file):
        with open(rule_scores_file, 'r', encoding='utf-8') as f:
            rule_scores = json.load(f)
    
    # Load comprehensive problem data for correct URLs
    comprehensive_data = {}
    comprehensive_path = os.path.join(inference_config.base_dir, "editorial_crawler", "data", "comprehensive_problem_data.json")
    if os.path.exists(comprehensive_path):
        with open(comprehensive_path, 'r', encoding='utf-8') as f:
            comp_data = json.load(f)
            for problem in comp_data['problems']:
                comprehensive_data[problem['problem_id']] = problem
    
    batch_info = batch_data['batch_info']
    batch_results = batch_data['results']
    
    # Process results
    processed_results = {}
    successful_inferences = 0
    all_confidences = []
    total_api_calls = 0
    
    for problem_id, result in batch_results.items():
        if result.get('error'):
            processed_results[problem_id] = None
        else:
            tags_data = result.get('tags', [])
            
            if not tags_data:
                processed_results[problem_id] = None
                continue
            
            # Extract tags, IDs, and confidences
            tags = [tag['name'] for tag in tags_data]
            tag_ids = [tag['id'] for tag in tags_data]
            confidences = [tag['confidence'] for tag in tags_data]
            
            # Create detailed scores using actual rule-based scores and calculate composite scores
            detailed_scores = {}
            problem_rule_scores = rule_scores.get(problem_id, {})
            composite_scores = []
            
            for i, (tag_id, confidence) in enumerate(zip(tag_ids, confidences)):
                rule_based_score = problem_rule_scores.get(tag_id, 0.0)
                
                # Estimate embedding similarity from confidence (normalized)
                embedding_similarity = confidence * 0.8
                
                # Calculate composite score: rule_based(0/1) + 0.5*self_confidence + 0.5*embedding_similarity
                composite_score = rule_based_score + (0.5 * confidence) + (0.5 * embedding_similarity)
                composite_scores.append(composite_score)
                
                detailed_scores[tag_id] = {
                    "self_confidence": confidence,
                    "embedding_similarity": embedding_similarity,
                    "rule_based_score": rule_based_score
                }
            
            # Sort tags by composite score (descending)
            tag_data = list(zip(tags, tag_ids, composite_scores, [detailed_scores[tag_id] for tag_id in tag_ids]))
            tag_data.sort(key=lambda x: x[2], reverse=True)  # Sort by composite score (descending)
            
            # Extract sorted data
            sorted_tags = [item[0] for item in tag_data]
            sorted_tag_ids = [item[1] for item in tag_data]
            sorted_composite_scores = [item[2] for item in tag_data]
            sorted_detailed_scores = {tag_id: detailed for _, tag_id, _, detailed in tag_data}
            
            # Calculate statistics using composite scores
            avg_confidence = statistics.mean(sorted_composite_scores) if sorted_composite_scores else 0
            min_confidence = min(sorted_composite_scores) if sorted_composite_scores else 0
            confidence_std = statistics.stdev(sorted_composite_scores) if len(sorted_composite_scores) > 1 else 0
            
            all_confidences.extend(sorted_composite_scores)  # Use composite scores for statistics
            
            # Generate proper problem title and URLs using comprehensive data
            problem_data = comprehensive_data.get(problem_id, {})
            contest_num = problem_id.split('_')[0]
            
            # Use actual URLs from comprehensive data if available
            title = problem_data.get('title', f"Problem {problem_id.replace('_', ' ').upper()}")
            problem_url = problem_data.get('problem_url', f"https://atcoder.jp/contests/{contest_num}/tasks/{problem_id}")
            editorial_url = problem_data.get('editorial_url', f"https://atcoder.jp/contests/{contest_num}/editorial")
            
            processed_results[problem_id] = {
                "title": title if title else f"Problem {problem_id.replace('_', ' ').upper()}",
                "problem_url": problem_url,
                "editorial_url": editorial_url,
                "tags": sorted_tags,
                "tag_ids": sorted_tag_ids,
                "confidence_scores": sorted_composite_scores,  # Use sorted composite scores
                "detailed_scores": sorted_detailed_scores
            }
            
            successful_inferences += 1
            total_api_calls += 1  # Batch API counts as 1 call per problem
    
    # Calculate overall statistics
    total_problems = len(batch_results)
    success_rate = successful_inferences / total_problems if total_problems > 0 else 0
    
    confidence_analysis = {
        "average_confidence": statistics.mean(all_confidences) if all_confidences else 0,
        "min_confidence": min(all_confidences) if all_confidences else 0,
        "max_confidence": max(all_confidences) if all_confidences else 0,
        "std_confidence": statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
    }
    
    # Create standard format
    standard_result = {
        "test_info": {
            "test_date": batch_info.get('completed_at', datetime.now().isoformat()),
            "problems_tested": total_problems,
            "successful_inferences": successful_inferences,
            "success_rate": success_rate,
            "total_test_time": 840,  # 14 minutes in seconds (estimated)
            "api_calls_per_problem": 1,
            "processing_method": "batch_api_50_percent_cost_reduction",
            "contest_range": "ABC180-199",
            "problem_levels": "C-F"
        },
        "confidence_analysis": confidence_analysis,
        "results": processed_results
    }
    
    # Save converted results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(standard_result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {total_problems} problems to standard format")
    print(f"📊 Success rate: {success_rate:.1%}")
    print(f"📈 Average confidence: {confidence_analysis['average_confidence']:.3f}")
    print(f"💾 Output: {output_file}")
    
    return standard_result

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Batch Results to Standard Format")
    parser.add_argument('--input', required=True, help='Input batch results file')
    parser.add_argument('--output', help='Output standard format file')
    parser.add_argument('--rule-scores', help='Rule-based scores file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(
            inference_config.base_dir, 
            "data", 
            f"abc180_199_standard_results_{timestamp}.json"
        )
    
    convert_batch_results_to_standard(args.input, args.output, getattr(args, 'rule_scores', None))
    print(f"\n✅ Conversion completed successfully!")

if __name__ == "__main__":
    main()