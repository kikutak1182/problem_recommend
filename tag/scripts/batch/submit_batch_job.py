#!/usr/bin/env python3
"""
Submit Batch Job for Tag Inference

Submit JSONL file to OpenAI Batch API for processing with 50% cost reduction.
"""

import json
import os
import sys
from datetime import datetime
from openai import OpenAI

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config

def submit_batch_job(jsonl_file: str, api_key: str):
    """Submit batch job to OpenAI"""
    
    if not os.path.exists(jsonl_file):
        print(f"❌ Batch file not found: {jsonl_file}")
        return None
    
    client = OpenAI(api_key=api_key)
    
    print(f"📤 Uploading batch file: {jsonl_file}")
    
    try:
        # Upload the batch file
        with open(jsonl_file, 'rb') as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        
        print(f"✅ File uploaded: {batch_input_file.id}")
        
        # Create batch job
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "ABC180-199 tag inference with 50% cost reduction",
                "created_by": "claude_code_batch_processor"
            }
        )
        
        print(f"🚀 Batch job created: {batch_job.id}")
        print(f"📊 Status: {batch_job.status}")
        print(f"⏰ Created: {datetime.fromtimestamp(batch_job.created_at)}")
        
        # Save batch info
        batch_info = {
            "batch_id": batch_job.id,
            "input_file_id": batch_input_file.id,
            "status": batch_job.status,
            "created_at": datetime.now().isoformat(),
            "jsonl_file": jsonl_file,
            "type": "tag_inference"
        }
        
        info_file = os.path.join(
            inference_config.base_dir, 
            "data", 
            f"batch_tag_inference_{batch_job.id}_info.json"
        )
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Batch info saved: {info_file}")
        
        print(f"\n📋 Monitor progress with:")
        print(f"python3 scripts/check_batch_status.py --batch-id {batch_job.id}")
        
        print(f"\n💰 Cost savings: ~50% vs real-time API")
        print(f"⏱️  Processing time: Up to 24 hours")
        
        return batch_job
        
    except Exception as e:
        print(f"❌ Error submitting batch job: {e}")
        return None

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Submit Batch Job for Tag Inference")
    parser.add_argument('--file', required=True, help='JSONL file with batch requests')
    parser.add_argument('--api-key', help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key required. Use --api-key or set OPENAI_API_KEY environment variable")
        return
    
    batch_job = submit_batch_job(args.file, api_key)
    
    if batch_job:
        print(f"\n✅ Batch job successfully submitted!")
    else:
        print(f"\n❌ Failed to submit batch job")

if __name__ == "__main__":
    main()