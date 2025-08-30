#!/usr/bin/env python3
"""
Check Batch Job Status

Monitor OpenAI Batch API job progress and download results when ready.
"""

import json
import os
import sys
from datetime import datetime
from openai import OpenAI

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.tag_inference_config import inference_config

def check_batch_status(batch_id: str, api_key: str, download_results: bool = False):
    """Check batch job status and optionally download results"""
    
    client = OpenAI(api_key=api_key)
    
    try:
        # Get batch job status
        batch_job = client.batches.retrieve(batch_id)
        
        print(f"🔍 Batch Job Status: {batch_id}")
        print(f"📊 Status: {batch_job.status}")
        print(f"⏰ Created: {datetime.fromtimestamp(batch_job.created_at)}")
        
        if hasattr(batch_job, 'in_progress_at') and batch_job.in_progress_at:
            print(f"🚀 Started: {datetime.fromtimestamp(batch_job.in_progress_at)}")
        
        if hasattr(batch_job, 'completed_at') and batch_job.completed_at:
            print(f"✅ Completed: {datetime.fromtimestamp(batch_job.completed_at)}")
        
        if hasattr(batch_job, 'request_counts'):
            counts = batch_job.request_counts
            print(f"📈 Requests:")
            print(f"   Total: {counts.total}")
            print(f"   Completed: {counts.completed}")
            print(f"   Failed: {counts.failed}")
        
        # Download results if completed and requested
        if batch_job.status == "completed" and download_results:
            if hasattr(batch_job, 'output_file_id') and batch_job.output_file_id:
                print(f"\n📥 Downloading results...")
                
                # Download output file
                result_content = client.files.content(batch_job.output_file_id)
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = os.path.join(
                    inference_config.base_dir,
                    "data",
                    f"batch_tag_results_{timestamp}.jsonl"
                )
                
                with open(output_file, 'wb') as f:
                    f.write(result_content.content)
                
                print(f"💾 Results saved: {output_file}")
                
                # Process results into readable format
                process_results_file = os.path.join(
                    inference_config.base_dir,
                    "data", 
                    f"abc180_199_batch_results_{timestamp}.json"
                )
                
                results = {}
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            problem_id = result['custom_id']
                            
                            if 'response' in result and result['response']:
                                try:
                                    content = result['response']['body']['choices'][0]['message']['content']
                                    
                                    # Clean markdown code blocks if present
                                    cleaned_content = content.strip()
                                    if cleaned_content.startswith('```json') and cleaned_content.endswith('```'):
                                        cleaned_content = cleaned_content[7:-3].strip()
                                    elif cleaned_content.startswith('```') and cleaned_content.endswith('```'):
                                        cleaned_content = cleaned_content[3:-3].strip()
                                    
                                    parsed_content = json.loads(cleaned_content)
                                    results[problem_id] = {
                                        'tags': parsed_content.get('tags', []),
                                        'raw_response': content
                                    }
                                except Exception as e:
                                    results[problem_id] = {
                                        'error': f'Failed to parse response: {str(e)}',
                                        'raw_response': content if 'content' in locals() else 'No content'
                                    }
                            else:
                                results[problem_id] = {
                                    'error': result.get('error', 'Unknown error')
                                }
                
                # Save processed results
                final_results = {
                    "batch_info": {
                        "batch_id": batch_id,
                        "completed_at": datetime.now().isoformat(),
                        "total_problems": len(results),
                        "cost_savings": "~50% vs real-time API"
                    },
                    "results": results
                }
                
                with open(process_results_file, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                
                print(f"📊 Processed results: {process_results_file}")
                
                # Show summary
                successful = len([r for r in results.values() if not r.get('error')])
                failed = len([r for r in results.values() if r.get('error')])
                
                print(f"\n📈 Results Summary:")
                print(f"   Successful: {successful}")
                print(f"   Failed: {failed}")
                print(f"   Success rate: {successful/(successful+failed):.1%}")
                
        elif batch_job.status == "failed":
            print(f"❌ Batch job failed")
            if hasattr(batch_job, 'errors') and batch_job.errors:
                for error in batch_job.errors:
                    print(f"   Error: {error}")
        
        elif batch_job.status in ["validating", "in_progress"]:
            print(f"⏳ Job is still processing...")
            print(f"🔄 Check again in a few minutes")
        
        return batch_job
        
    except Exception as e:
        print(f"❌ Error checking batch status: {e}")
        return None

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Batch Job Status")
    parser.add_argument('--batch-id', required=True, help='Batch job ID')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--download', action='store_true', help='Download results if completed')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key required. Use --api-key or set OPENAI_API_KEY environment variable")
        return
    
    batch_job = check_batch_status(args.batch_id, api_key, args.download)
    
    if not batch_job:
        print(f"\n❌ Failed to check batch status")

if __name__ == "__main__":
    main()