#!/usr/bin/env python3
"""
Create Batch API Requests for ABC180-199 Tag Inference

Generate JSONL file for OpenAI Batch API processing with 50% cost reduction.
"""

import json
import os
import sys
import logging
from typing import Dict, List
from datetime import datetime

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config
from scripts.cache.editorial_text_extractor import EditorialTextExtractor
from scripts.filtering.cached_embedding_filter import CachedEmbeddingTagFilter

class BatchRequestCreator:
    """Create batch requests for tag inference"""
    
    def __init__(self):
        self.extractor = EditorialTextExtractor()
        self.embedding_filter = CachedEmbeddingTagFilter()
        self.logger = self._setup_logger()
        
        # Load comprehensive data
        self.comprehensive_data_path = os.path.join(
            inference_config.base_dir, "editorial_crawler", "data", "comprehensive_problem_data.json"
        )
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger('batch_request_creator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_comprehensive_data(self) -> List[Dict]:
        """Load comprehensive problem data"""
        with open(self.comprehensive_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['problems']
    
    def get_problems_by_range(self, start_contest: int = None, end_contest: int = None, 
                             target_problems: List[str] = None) -> List[Dict]:
        """Get problems by contest range and difficulty"""
        problems = self.load_comprehensive_data()
        
        # Use config defaults if not specified
        if start_contest is None:
            start_contest = inference_config.default_start_contest
        if end_contest is None:
            end_contest = inference_config.default_end_contest
        if target_problems is None:
            target_problems = inference_config.target_problems
        
        self.logger.info(f"Filtering problems: ABC{start_contest}-{end_contest}, problems: {target_problems}")
        
        filtered_problems = []
        for problem in problems:
            problem_id = problem['problem_id']
            if problem_id.startswith('abc'):
                try:
                    contest_num = int(problem_id.split('_')[0][3:])
                    problem_level = problem_id.split('_')[1]
                    
                    # Filter by contest range and difficulty threshold
                    if (start_contest <= contest_num <= end_contest and 
                        problem.get('difficulty') is not None and problem.get('difficulty') >= inference_config.difficulty_threshold):
                        filtered_problems.append(problem)
                except:
                    continue
        
        # Sort by contest and problem level
        filtered_problems.sort(key=lambda x: (int(x['problem_id'].split('_')[0][3:]), x['problem_id'].split('_')[1]))
        
        return filtered_problems
    
    def create_tag_inference_prompt(self, problem_text: str, candidate_tags: List[Dict]) -> str:
        """Create prompt for tag inference"""
        
        tag_list = []
        for tag in candidate_tags:
            tag_list.append(f"- {tag['name']} ({tag['id']}): {tag['description']}")
        
        tags_text = "\n".join(tag_list)
        
        prompt = f"""あなたは競技プログラミング問題のタグ推定の専門家です。

以下の問題文・解説を分析し、適切なタグとその信頼度（0.0-1.0）を推定してください。

【問題文・解説】
{problem_text}

【候補タグ一覧】
{tags_text}

【指示】
1. この問題に最も適切なタグを3つ選択してください
2. 各タグの信頼度を0.0-1.0で評価してください
3. 必ずJSON形式で回答してください

【回答形式】
{{
  "tags": [
    {{"id": "TAG_ID", "name": "タグ名", "confidence": 0.85}},
    {{"id": "TAG_ID", "name": "タグ名", "confidence": 0.75}},
    {{"id": "TAG_ID", "name": "タグ名", "confidence": 0.65}}
  ]
}}"""
        
        return prompt
    
    def create_batch_requests(self, output_file: str, start_contest: int = None, 
                             end_contest: int = None, target_problems: List[str] = None):
        """Create batch requests file"""
        
        problems = self.get_problems_by_range(start_contest, end_contest, target_problems)
        self.logger.info(f"Creating batch requests for {len(problems)} problems")
        
        batch_requests = []
        rule_based_scores = {}  # Store rule-based scores for later use
        success_count = 0
        error_count = 0
        
        for i, problem in enumerate(problems, 1):
            problem_id = problem['problem_id']
            editorial_url = problem.get('editorial_url')
            
            if not editorial_url:
                self.logger.warning(f"[{i}/{len(problems)}] {problem_id}: No editorial URL")
                error_count += 1
                continue
            
            try:
                self.logger.info(f"[{i}/{len(problems)}] Processing {problem_id}...")
                
                # Load combined text from cache
                cache_path = os.path.join(inference_config.base_dir, "data", "problem_combined_text_cache.json")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                
                if problem_id not in cache:
                    self.logger.warning(f"  {problem_id} not found in text cache")
                    error_count += 1
                    continue
                
                combined_text = cache[problem_id]['combined_text']
                
                if not combined_text:
                    self.logger.warning(f"  Failed to extract text for {problem_id}")
                    error_count += 1
                    continue
                
                # Get candidate tags using embedding filter
                candidate_tags = self.embedding_filter.filter_candidate_tags_by_id(
                    problem_id, 
                    combined_text
                )
                
                if not candidate_tags:
                    self.logger.warning(f"  No candidate tags found for {problem_id}")
                    error_count += 1
                    continue
                
                # Calculate and store rule-based scores for this problem
                from scripts.filtering.keyword_matcher import KeywordMatcher
                keyword_matcher = KeywordMatcher()
                rule_scores = keyword_matcher.calculate_rule_based_scores(combined_text, candidate_tags)
                rule_based_scores[problem_id] = rule_scores
                
                # Create prompt
                prompt = self.create_tag_inference_prompt(combined_text, candidate_tags)
                
                # Create batch request
                batch_request = {
                    "custom_id": problem_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                }
                
                batch_requests.append(batch_request)
                success_count += 1
                
                self.logger.info(f"  ✓ {problem_id}: {len(candidate_tags)} candidates, {len(combined_text)} chars")
                
                # Small delay to be respectful
                import time
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"  ✗ {problem_id}: {e}")
                error_count += 1
        
        # Save batch requests
        with open(output_file, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        # Save rule-based scores
        timestamp = output_file.split('_')[-1].replace('.jsonl', '')
        start = start_contest or inference_config.default_start_contest  
        end = end_contest or inference_config.default_end_contest
        rule_scores_file = os.path.join(
            inference_config.base_dir, "data", 
            f"abc{start}_{end}_rule_based_scores_{timestamp}.json"
        )
        
        with open(rule_scores_file, 'w', encoding='utf-8') as f:
            json.dump(rule_based_scores, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch request creation completed:")
        self.logger.info(f"  Success: {success_count}")
        self.logger.info(f"  Errors: {error_count}")
        self.logger.info(f"  Output file: {output_file}")
        self.logger.info(f"  Rule-based scores saved: {rule_scores_file}")
        
        return success_count, error_count

def main():
    """Main execution"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Create Batch API Requests for Tag Inference")
    parser.add_argument('--output', default=None, 
                       help='Output JSONL file for batch requests (default: auto-generated with timestamp)')
    parser.add_argument('--start', type=int, default=None, 
                       help=f'Start contest number (default: {inference_config.default_start_contest})')
    parser.add_argument('--end', type=int, default=None, 
                       help=f'End contest number (default: {inference_config.default_end_contest})')
    parser.add_argument('--target-problems', nargs='+', default=None,
                       help=f'Target problem types (default: {inference_config.target_problems})')
    
    args = parser.parse_args()
    
    # Generate default output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start = args.start or inference_config.default_start_contest
        end = args.end or inference_config.default_end_contest
        args.output = f"data/abc{start}_{end}_batch_requests_{timestamp}.jsonl"
    
    creator = BatchRequestCreator()
    success, errors = creator.create_batch_requests(
        output_file=args.output,
        start_contest=args.start,
        end_contest=args.end,
        target_problems=args.target_problems
    )
    
    if success > 0:
        print(f"\n✅ Batch requests file created: {args.output}")
        print(f"📊 Requests: {success} successful, {errors} errors")
        print(f"\n🚀 Next step: Submit batch job with:")
        print(f"python3 scripts/submit_batch_job.py --file {args.output}")
    else:
        print(f"\n❌ Failed to create batch requests")

if __name__ == "__main__":
    main()