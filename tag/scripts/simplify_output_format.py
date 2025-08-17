#!/usr/bin/env python3
"""
Convert existing problems_with_tags.json to simplified format
Removes verbose reasoning and detailed confidence breakdowns
"""

import json
import os
import sys
from datetime import datetime

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config

def simplify_problem_data(problem_data: dict) -> dict:
    """Convert a problem entry to simplified format"""
    
    # Keep only essential fields
    simplified = {}
    
    # Keep problem metadata if present
    for field in ["contest_id", "problem_index", "title", "problem_url", "editorial_url", "editorial_id"]:
        if field in problem_data:
            simplified[field] = problem_data[field]
    
    # Keep only essential inference fields
    if "tags" in problem_data:
        simplified["tags"] = problem_data["tags"]
    
    if "confidence_scores" in problem_data:
        simplified["confidence_scores"] = problem_data["confidence_scores"]
    
    if "avg_confidence" in problem_data:
        simplified["avg_confidence"] = problem_data["avg_confidence"]
    
    if "low_confidence" in problem_data:
        simplified["low_confidence"] = problem_data["low_confidence"]
    
    # Set method and model with defaults
    simplified["method"] = problem_data.get("method", "enhanced_composite_confidence")
    simplified["model"] = problem_data.get("model", "o4-mini")
    simplified["inferred_at"] = problem_data.get("inferred_at", datetime.now().isoformat())
    
    return simplified

def main():
    """Main conversion function"""
    
    # Load existing data
    try:
        with open(inference_config.problems_with_tags_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {inference_config.problems_with_tags_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return 1
    
    problems = data.get("problems", {})
    metadata = data.get("metadata", {})
    
    print(f"Converting {len(problems)} problems to simplified format...")
    
    # Convert all problems
    simplified_problems = {}
    for problem_id, problem_data in problems.items():
        simplified_problems[problem_id] = simplify_problem_data(problem_data)
    
    # Update metadata
    metadata.update({
        "format_simplified_at": datetime.now().isoformat(),
        "simplified_reason": "Removed verbose reasoning and detailed confidence to reduce token usage",
        "last_updated": datetime.now().isoformat()
    })
    
    # Create output data
    output_data = {
        "problems": simplified_problems,
        "metadata": metadata
    }
    
    # Create backup of original
    backup_path = inference_config.problems_with_tags_path + ".backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Original backed up to: {backup_path}")
    
    # Save simplified version
    with open(inference_config.problems_with_tags_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Simplified format saved to: {inference_config.problems_with_tags_path}")
    print(f"✅ Format conversion completed for {len(simplified_problems)} problems")
    
    # Show size reduction
    try:
        original_size = os.path.getsize(backup_path)
        new_size = os.path.getsize(inference_config.problems_with_tags_path)
        reduction = (original_size - new_size) / original_size * 100
        print(f"✅ File size reduced by {reduction:.1f}% ({original_size:,} → {new_size:,} bytes)")
    except:
        pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())