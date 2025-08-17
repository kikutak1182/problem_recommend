#!/usr/bin/env python3
"""
Batch Results Aggregator for ABC250-400

Processes and aggregates results from all 7 batches into the main problems_with_tags.json file.
Provides statistics and quality metrics for the tag inference results.
"""

import json
import os
import sys
import glob
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter

# Import configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.batch_api_processor import BatchAPIProcessor
from scripts.tag_inference_config import inference_config

class BatchResultsAggregator:
    """Aggregates and processes batch results"""
    
    def __init__(self):
        self.processor = BatchAPIProcessor()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('batch_results_aggregator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def find_batch_result_files(self) -> List[str]:
        """Find all batch result files"""
        data_dir = os.path.join(inference_config.base_dir, "data")
        pattern = os.path.join(data_dir, "batch_results_batch_*.jsonl")
        files = glob.glob(pattern)
        
        self.logger.info(f"Found {len(files)} batch result files")
        return sorted(files)
    
    def process_all_batch_results(self) -> Dict[str, Dict]:
        """Process all batch results into unified format"""
        
        all_results = {}
        batch_files = self.find_batch_result_files()
        
        for i, result_file in enumerate(batch_files, 1):
            self.logger.info(f"Processing batch {i}: {os.path.basename(result_file)}")
            
            batch_results = self.processor.process_batch_results(result_file)
            
            if batch_results:
                all_results.update(batch_results)
                self.logger.info(f"Added {len(batch_results)} results from batch {i}")
            else:
                self.logger.warning(f"No results from batch {i}")
        
        self.logger.info(f"Total processed results: {len(all_results)}")
        return all_results
    
    def generate_statistics(self, results: Dict[str, Dict]) -> Dict:
        """Generate statistics from batch results"""
        
        stats = {
            "total_problems": len(results),
            "average_confidence": 0.0,
            "tag_distribution": Counter(),
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "method_distribution": Counter(),
            "processing_summary": {
                "processed_at": datetime.now().isoformat(),
                "model_used": inference_config.model_name,
                "batch_count": 7
            }
        }
        
        confidence_scores = []
        
        for problem_id, result in results.items():
            # Confidence analysis
            avg_conf = result.get("avg_confidence", 0.0)
            confidence_scores.append(avg_conf)
            
            if avg_conf >= 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif avg_conf >= 0.6:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
            
            # Tag distribution
            tags = result.get("tags", [])
            for tag in tags:
                stats["tag_distribution"][tag] += 1
            
            # Method distribution
            method = result.get("method", "unknown")
            stats["method_distribution"][method] += 1
        
        # Calculate average confidence
        if confidence_scores:
            stats["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        # Convert Counter to dict for JSON serialization
        stats["tag_distribution"] = dict(stats["tag_distribution"])
        stats["method_distribution"] = dict(stats["method_distribution"])
        
        return stats
    
    def update_problems_with_tags(self, new_results: Dict[str, Dict]) -> Dict:
        """Update problems_with_tags.json with new results"""
        
        problems_file = inference_config.problems_with_tags_path
        
        # Load existing data
        try:
            with open(problems_file, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        except FileNotFoundError:
            current_data = {"problems": {}}
        
        # Backup existing file
        backup_path = f"{problems_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created backup: {backup_path}")
        
        # Update with new results
        updated_count = 0
        new_count = 0
        
        for problem_id, result in new_results.items():
            if problem_id in current_data["problems"]:
                # Update existing problem
                current_data["problems"][problem_id].update({
                    "tags": result["tags"],
                    "confidence_scores": result.get("confidence_scores", []),
                    "avg_confidence": result.get("avg_confidence", 0.0),
                    "low_confidence": result.get("low_confidence", False),
                    "method": result.get("method", "batch_api_processing"),
                    "model": result.get("model", inference_config.model_name),
                    "inferred_at": result.get("inferred_at", datetime.now().isoformat())
                })
                updated_count += 1
            else:
                # Add new problem (shouldn't happen for ABC250-400, but just in case)
                current_data["problems"][problem_id] = {
                    "title": problem_id,  # Will be updated from external data
                    "tags": result["tags"],
                    "confidence_scores": result.get("confidence_scores", []),
                    "avg_confidence": result.get("avg_confidence", 0.0),
                    "low_confidence": result.get("low_confidence", False),
                    "method": result.get("method", "batch_api_processing"),
                    "model": result.get("model", inference_config.model_name),
                    "inferred_at": result.get("inferred_at", datetime.now().isoformat())
                }
                new_count += 1
        
        # Save updated data
        with open(problems_file, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Updated {updated_count} problems, added {new_count} new problems")
        self.logger.info(f"Updated file: {problems_file}")
        
        return current_data
    
    def save_statistics_report(self, stats: Dict) -> str:
        """Save statistics report to file"""
        
        report_path = os.path.join(
            inference_config.base_dir,
            "data",
            f"abc250_400_statistics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Statistics report saved: {report_path}")
        return report_path
    
    def generate_human_readable_report(self, stats: Dict) -> str:
        """Generate human-readable report"""
        
        report = []
        report.append("=" * 60)
        report.append("ABC250-400 タグ推定 - 最終レポート")
        report.append("=" * 60)
        report.append(f"処理日時: {stats['processing_summary']['processed_at']}")
        report.append(f"使用モデル: {stats['processing_summary']['model_used']}")
        report.append(f"バッチ数: {stats['processing_summary']['batch_count']}")
        report.append("")
        
        # Summary statistics
        report.append("【処理サマリー】")
        report.append(f"・処理問題数: {stats['total_problems']:,}問題")
        report.append(f"・平均確信度: {stats['average_confidence']:.3f}")
        report.append("")
        
        # Confidence distribution
        report.append("【確信度分布】")
        conf_dist = stats['confidence_distribution']
        total = sum(conf_dist.values())
        for level, count in conf_dist.items():
            percentage = (count / total * 100) if total > 0 else 0
            report.append(f"・{level.upper()}: {count:,}問題 ({percentage:.1f}%)")
        report.append("")
        
        # Top tags
        report.append("【頻出タグ TOP 10】")
        top_tags = sorted(stats['tag_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (tag, count) in enumerate(top_tags, 1):
            report.append(f"{i:2d}. {tag}: {count:,}問題")
        report.append("")
        
        # Method distribution
        report.append("【処理方法分布】")
        for method, count in stats['method_distribution'].items():
            percentage = (count / stats['total_problems'] * 100) if stats['total_problems'] > 0 else 0
            report.append(f"・{method}: {count:,}問題 ({percentage:.1f}%)")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def aggregate_all_results(self) -> Dict:
        """Main aggregation function"""
        
        self.logger.info("Starting batch results aggregation...")
        
        # Process all batch results
        all_results = self.process_all_batch_results()
        
        if not all_results:
            self.logger.error("No results to aggregate")
            return {}
        
        # Generate statistics
        stats = self.generate_statistics(all_results)
        
        # Update main problems file
        updated_data = self.update_problems_with_tags(all_results)
        
        # Save statistics report
        report_path = self.save_statistics_report(stats)
        
        # Generate and print human-readable report
        human_report = self.generate_human_readable_report(stats)
        print(human_report)
        
        # Save human-readable report
        human_report_path = os.path.join(
            inference_config.base_dir,
            "data",
            f"abc250_400_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        with open(human_report_path, 'w', encoding='utf-8') as f:
            f.write(human_report)
        
        self.logger.info(f"Human-readable report saved: {human_report_path}")
        self.logger.info("Aggregation completed successfully!")
        
        return {
            "statistics": stats,
            "results": all_results,
            "report_path": report_path,
            "human_report_path": human_report_path
        }

def main():
    """Main function"""
    aggregator = BatchResultsAggregator()
    result = aggregator.aggregate_all_results()
    
    if result:
        print(f"\n✅ 集計完了!")
        print(f"📊 統計レポート: {result['report_path']}")
        print(f"📄 詳細レポート: {result['human_report_path']}")
        print(f"🎯 処理問題数: {len(result['results'])}問題")

if __name__ == "__main__":
    main()