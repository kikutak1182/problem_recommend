import json
import re
import sys
import os
from typing import List, Dict, Set
from datetime import datetime

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.tag_config import config as tag_config
from crawler_config import config as crawler_config

class ContestFilter:
    """Filter contests and problems based on target criteria"""
    
    def __init__(self):
        self.problems_data = self._load_problems_data()
    
    def _load_problems_data(self) -> List[Dict]:
        """Load problems data from the main tag system"""
        try:
            with open(tag_config.problems_data_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Problems data file not found: {tag_config.problems_data_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in problems data: {e}")
            return []
    
    def get_target_contests(self) -> Set[str]:
        """Get all target contest IDs that meet the criteria"""
        target_contests = set()
        
        for problem in self.problems_data:
            contest_id = problem.get('contest_id', '')
            
            if crawler_config.is_target_contest(contest_id):
                target_contests.add(contest_id)
        
        return target_contests
    
    def get_target_problems(self) -> Dict[str, List[Dict]]:
        """Get problems grouped by contest ID for target contests"""
        problems_by_contest = {}
        
        for problem in self.problems_data:
            contest_id = problem.get('contest_id', '')
            
            if crawler_config.is_target_contest(contest_id):
                if contest_id not in problems_by_contest:
                    problems_by_contest[contest_id] = []
                
                problems_by_contest[contest_id].append({
                    'problem_id': problem.get('id', ''),
                    'problem_index': problem.get('problem_index', ''),
                    'title': problem.get('title', ''),
                    'contest_id': contest_id
                })
        
        return problems_by_contest
    
    def filter_problems_by_contest_type(self, contest_type: str) -> Dict[str, List[Dict]]:
        """Filter problems by specific contest type (abc, arc, agc)"""
        all_problems = self.get_target_problems()
        filtered = {}
        
        for contest_id, problems in all_problems.items():
            if contest_id.startswith(contest_type):
                filtered[contest_id] = problems
        
        return filtered
    
    def get_contest_statistics(self) -> Dict[str, Dict]:
        """Get statistics about target contests"""
        target_problems = self.get_target_problems()
        stats = {}
        
        for contest_type, min_number in crawler_config.target_contests.items():
            contest_count = 0
            problem_count = 0
            contest_numbers = []
            
            for contest_id, problems in target_problems.items():
                if contest_id.startswith(contest_type):
                    try:
                        number = int(contest_id[len(contest_type):])
                        contest_numbers.append(number)
                        contest_count += 1
                        problem_count += len(problems)
                    except ValueError:
                        continue
            
            if contest_numbers:
                stats[contest_type] = {
                    'contest_count': contest_count,
                    'problem_count': problem_count,
                    'min_number': min(contest_numbers),
                    'max_number': max(contest_numbers),
                    'threshold': min_number
                }
        
        return stats
    
    def validate_contest_criteria(self) -> bool:
        """Validate that contest filtering criteria are working correctly"""
        stats = self.get_contest_statistics()
        
        for contest_type, stat in stats.items():
            threshold = crawler_config.target_contests[contest_type]
            if stat['min_number'] < threshold:
                print(f"Warning: Found {contest_type} contests below threshold {threshold}")
                return False
        
        return True
    
    def print_statistics(self):
        """Print contest filtering statistics"""
        stats = self.get_contest_statistics()
        
        print("=== Contest Filter Statistics ===")
        for contest_type, stat in stats.items():
            print(f"{contest_type.upper()}:")
            print(f"  Threshold: {stat['threshold']}+ ")
            print(f"  Contests: {stat['contest_count']} ({stat['min_number']}-{stat['max_number']})")
            print(f"  Problems: {stat['problem_count']}")
        
        total_contests = sum(stat['contest_count'] for stat in stats.values())
        total_problems = sum(stat['problem_count'] for stat in stats.values())
        
        print(f"\nTotal Target Contests: {total_contests}")
        print(f"Total Target Problems: {total_problems}")

if __name__ == "__main__":
    filter = ContestFilter()
    filter.print_statistics()
    
    # Validation
    if filter.validate_contest_criteria():
        print("\n✅ Contest filtering criteria validated successfully")
    else:
        print("\n❌ Contest filtering validation failed")