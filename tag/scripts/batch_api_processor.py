#!/usr/bin/env python3
"""
Batch API Processor for Cost-Efficient Tag Inference

Uses OpenAI Batch API for 50% cost reduction on large-scale processing.
Processes requests asynchronously with 24-hour completion window.
"""

import json
import os
import sys
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import openai
from dataclasses import dataclass

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.tag_inference_config import inference_config
from scripts.enhanced_tag_inference import EnhancedTagInference

@dataclass
class BatchRequest:
    """Single batch request format"""
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: Dict = None

class BatchAPIProcessor:
    """Batch API processor for cost-efficient tag inference"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=inference_config.openai_api_key)
        self.logger = self._setup_logger()
        self.inference_engine = EnhancedTagInference()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('batch_api_processor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_batch_requests(self, problems_data: List[Dict]) -> List[Dict]:
        """Create batch API requests from problems"""
        
        requests = []
        
        for i, problem_data in enumerate(problems_data):
            try:
                # Get editorial text and candidate tags (cached)
                editorial_text = self.inference_engine.text_extractor.extract_editorial_text(
                    problem_data['editorial_url']
                )
                
                if not editorial_text:
                    self.logger.warning(f"No editorial text for {problem_data['problem_id']}")
                    continue
                
                # Get candidate tags using embedding filter
                candidate_tags = self.inference_engine.confidence_system.embedding_filter.filter_candidate_tags(
                    problem_data.get('title', ''), editorial_text
                )
                
                if not candidate_tags:
                    self.logger.warning(f"No candidate tags for {problem_data['problem_id']}")
                    continue
                
                # Create system message with caching
                system_msg = """あなたは競技プログラミングのタグ付け専門家です。
問題と解説を読んで、候補タグから3つ選択し、各タグの確信度(0-1)と根拠を必ず示してください。"""
                
                # Create user message
                tag_list = [f"{tag['id']}: {tag['name']}" for tag in candidate_tags]
                tags_info = "\\n".join(tag_list)
                
                user_msg = f"""[Problem]
{problem_data.get('title', problem_data['problem_id'])}

[Editorial]
{editorial_text[:1000]}

[Candidate Tags]
{tags_info}

上記から3つのタグを選び、各タグの確信度(0-1)と根拠を示してください。"""
                
                # Create JSON schema
                allowed_ids = [tag['id'] for tag in candidate_tags]
                schema = {
                    "type": "object",
                    "properties": {
                        "selected_tags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tag_id": {"type": "string", "enum": allowed_ids},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    "reasoning": {"type": "string"}
                                },
                                "required": ["tag_id", "confidence", "reasoning"]
                            },
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "required": ["selected_tags"]
                }
                
                # Create batch request
                request_body = {
                    "model": inference_config.model_name,
                    "messages": [
                        {
                            "role": "system", 
                            "content": system_msg,
                            "cache_control": {"type": "ephemeral"}
                        },
                        {"role": "user", "content": user_msg}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "tag_confidence",
                            "schema": schema
                        }
                    }
                }
                
                request = {
                    "custom_id": f"tag_inference_{problem_data['problem_id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body
                }
                
                requests.append(request)
                
            except Exception as e:
                self.logger.error(f"Failed to create request for {problem_data['problem_id']}: {e}")
                continue
        
        self.logger.info(f"Created {len(requests)} batch requests")
        return requests
    
    def submit_batch(self, requests: List[Dict]) -> str:
        """Submit batch requests and return batch ID"""
        
        # Create batch file
        batch_filename = f"batch_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        batch_filepath = os.path.join(inference_config.base_dir, "data", batch_filename)
        
        # Write requests to JSONL file
        with open(batch_filepath, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created batch file: {batch_filepath}")
        
        # Upload file to OpenAI
        batch_file = self.client.files.create(
            file=open(batch_filepath, 'rb'),
            purpose="batch"
        )
        
        self.logger.info(f"Uploaded batch file: {batch_file.id}")
        
        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Tag inference batch processing",
                "created_at": datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Created batch: {batch.id}")
        self.logger.info(f"Status: {batch.status}")
        self.logger.info(f"Request count: {batch.request_counts}")
        
        # Save batch info
        batch_info = {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": datetime.now().isoformat(),
            "request_count": len(requests),
            "file_id": batch_file.id,
            "local_file": batch_filepath
        }
        
        batch_info_path = os.path.join(inference_config.base_dir, "data", f"batch_info_{batch.id}.json")
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        return batch.id
    
    def check_batch_status(self, batch_id: str) -> Dict:
        """Check batch processing status"""
        
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            status_info = {
                "batch_id": batch_id,
                "status": batch.status,
                "request_counts": batch.request_counts.__dict__ if batch.request_counts else {},
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "expires_at": batch.expires_at,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id
            }
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to check batch status: {e}")
            return {"error": str(e)}
    
    def download_batch_results(self, batch_id: str) -> Optional[str]:
        """Download and save batch results"""
        
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            if batch.status != "completed":
                self.logger.warning(f"Batch {batch_id} not completed yet. Status: {batch.status}")
                return None
            
            if not batch.output_file_id:
                self.logger.error(f"No output file for batch {batch_id}")
                return None
            
            # Download results
            output_content = self.client.files.content(batch.output_file_id)
            
            # Save results
            results_filename = f"batch_results_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            results_filepath = os.path.join(inference_config.base_dir, "data", results_filename)
            
            with open(results_filepath, 'wb') as f:
                f.write(output_content.content)
            
            self.logger.info(f"Downloaded batch results: {results_filepath}")
            
            # Download errors if any
            if batch.error_file_id:
                error_content = self.client.files.content(batch.error_file_id)
                error_filename = f"batch_errors_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                error_filepath = os.path.join(inference_config.base_dir, "data", error_filename)
                
                with open(error_filepath, 'wb') as f:
                    f.write(error_content.content)
                
                self.logger.warning(f"Downloaded batch errors: {error_filepath}")
            
            return results_filepath
            
        except Exception as e:
            self.logger.error(f"Failed to download batch results: {e}")
            return None
    
    def process_batch_results(self, results_filepath: str) -> Dict[str, Dict]:
        """Process batch results and convert to tag inference format"""
        
        results = {}
        
        try:
            with open(results_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line.strip())
                        
                        custom_id = result.get('custom_id', '')
                        if not custom_id.startswith('tag_inference_'):
                            continue
                        
                        problem_id = custom_id.replace('tag_inference_', '')
                        
                        # Extract response
                        if result.get('response') and result['response'].get('body'):
                            response_body = result['response']['body']
                            if response_body.get('choices'):
                                content = response_body['choices'][0]['message']['content']
                                
                                try:
                                    parsed_content = json.loads(content)
                                    selected_tags = parsed_content.get('selected_tags', [])
                                    
                                    # Convert to expected format
                                    tags = []
                                    confidence_scores = []
                                    
                                    for tag_info in selected_tags:
                                        tag_id = tag_info.get('tag_id')
                                        confidence = tag_info.get('confidence', 0.0)
                                        
                                        # Convert tag_id to tag name
                                        tag_name = self._get_tag_name_by_id(tag_id)
                                        if tag_name:
                                            tags.append(tag_name)
                                            confidence_scores.append(confidence)
                                    
                                    if tags:
                                        avg_confidence = sum(confidence_scores) / len(confidence_scores)
                                        low_confidence = avg_confidence < 0.6
                                        
                                        results[problem_id] = {
                                            "tags": tags,
                                            "confidence_scores": confidence_scores,
                                            "avg_confidence": avg_confidence,
                                            "low_confidence": low_confidence,
                                            "method": "batch_api_processing",
                                            "model": inference_config.model_name,
                                            "inferred_at": datetime.now().isoformat()
                                        }
                                    
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"Failed to parse content for {problem_id}: {e}")
                                    continue
                        
                        # Handle errors
                        if result.get('error'):
                            self.logger.error(f"Error for {problem_id}: {result['error']}")
                            continue
            
            self.logger.info(f"Processed {len(results)} batch results")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process batch results: {e}")
            return {}
    
    def _get_tag_name_by_id(self, tag_id: str) -> Optional[str]:
        """Get tag name by ID from tag definitions"""
        try:
            tag_defs = self.inference_engine.confidence_system.embedding_filter.tag_definitions
            for tag in tag_defs.get('tags', []):
                if tag.get('id') == tag_id:
                    return tag.get('name')
            return None
        except:
            return None

def main():
    """Example usage of batch API processor"""
    
    processor = BatchAPIProcessor()
    
    # Example: Create batch from sample problems
    sample_problems = [
        {
            "problem_id": "abc300_d",
            "title": "Sample Problem",
            "editorial_url": "https://atcoder.jp/contests/abc300/editorial/6500"
        }
    ]
    
    print("Creating batch requests...")
    requests = processor.create_batch_requests(sample_problems)
    
    if requests:
        print("Submitting batch...")
        batch_id = processor.submit_batch(requests)
        print(f"Batch submitted: {batch_id}")
        print("Batch processing will complete within 24 hours.")
        print(f"Check status with: processor.check_batch_status('{batch_id}')")
    else:
        print("No valid requests created")

if __name__ == "__main__":
    main()