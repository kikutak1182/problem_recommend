import json
import requests
import time
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from urllib.parse import urlparse

from crawler_config import config
from editorial_extractor import EditorialExtractor

class EditorialValidator:
    """Validate editorial URLs and database integrity"""
    
    def __init__(self):
        self.extractor = EditorialExtractor()
        self.logger = self._setup_logger()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent
        })
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for validation operations"""
        logger = logging.getLogger('editorial_validator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(config.log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_database(self, database_path: str = None) -> Dict[str, any]:
        """
        Validate the entire editorial database
        
        Returns:
            Validation report with statistics and issues
        """
        if database_path is None:
            database_path = config.database_path
        
        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                database = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {"error": f"Failed to load database: {e}"}
        
        self.logger.info("Starting database validation...")
        
        mappings = database.get("editorial_mappings", {})
        validation_report = {
            "total_problems": len(mappings),
            "validation_results": {
                "valid_urls": 0,
                "invalid_urls": 0,
                "accessible_urls": 0,
                "inaccessible_urls": 0,
                "missing_editorial_ids": 0,
                "invalid_problem_urls": 0
            },
            "issues": [],
            "validated_at": datetime.now().isoformat()
        }
        
        for i, (problem_key, mapping) in enumerate(mappings.items(), 1):
            if i % 50 == 0:
                self.logger.info(f"Validated {i}/{len(mappings)} entries")
            
            issues = self._validate_mapping(problem_key, mapping)
            if issues:
                validation_report["issues"].extend(issues)
            
            # Update counters
            self._update_validation_counters(mapping, issues, validation_report["validation_results"])
            
            # Rate limiting
            time.sleep(0.1)
        
        self.logger.info(f"Validation completed: {validation_report['validation_results']}")
        return validation_report
    
    def _validate_mapping(self, problem_key: str, mapping: Dict) -> List[Dict]:
        """Validate a single editorial mapping entry"""
        issues = []
        
        # Check required fields
        required_fields = ["contest_id", "problem_index", "problem_url", "editorial_url"]
        for field in required_fields:
            if field not in mapping or not mapping[field]:
                issues.append({
                    "problem_key": problem_key,
                    "type": "missing_field",
                    "field": field,
                    "severity": "error"
                })
        
        # Validate URLs format
        if "editorial_url" in mapping:
            if not self._is_valid_url_format(mapping["editorial_url"]):
                issues.append({
                    "problem_key": problem_key,
                    "type": "invalid_url_format",
                    "url": mapping["editorial_url"],
                    "severity": "error"
                })
        
        if "problem_url" in mapping:
            if not self._is_valid_url_format(mapping["problem_url"]):
                issues.append({
                    "problem_key": problem_key,
                    "type": "invalid_problem_url_format", 
                    "url": mapping["problem_url"],
                    "severity": "error"
                })
        
        # Validate editorial ID
        if "editorial_id" not in mapping or mapping["editorial_id"] is None:
            issues.append({
                "problem_key": problem_key,
                "type": "missing_editorial_id",
                "severity": "warning"
            })
        
        # Validate contest ID consistency
        if "contest_id" in mapping and "_" in problem_key:
            expected_contest = problem_key.split("_")[0]
            if mapping["contest_id"] != expected_contest:
                issues.append({
                    "problem_key": problem_key,
                    "type": "contest_id_mismatch",
                    "expected": expected_contest,
                    "actual": mapping["contest_id"],
                    "severity": "error"
                })
        
        return issues
    
    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _update_validation_counters(self, mapping: Dict, issues: List[Dict], counters: Dict):
        """Update validation result counters"""
        # Check URL validity
        has_url_format_issues = any(issue["type"] == "invalid_url_format" for issue in issues)
        has_problem_url_issues = any(issue["type"] == "invalid_problem_url_format" for issue in issues)
        
        if not has_url_format_issues:
            counters["valid_urls"] += 1
        else:
            counters["invalid_urls"] += 1
        
        if has_problem_url_issues:
            counters["invalid_problem_urls"] += 1
        
        # Check editorial ID
        if mapping.get("editorial_id") is None:
            counters["missing_editorial_ids"] += 1
    
    def validate_url_accessibility(self, urls: List[str], sample_size: int = 10) -> Dict[str, int]:
        """
        Validate URL accessibility for a sample of URLs
        
        Args:
            urls: List of URLs to validate
            sample_size: Number of URLs to check (for performance)
        """
        self.logger.info(f"Checking accessibility of {min(sample_size, len(urls))} URLs...")
        
        results = {"accessible": 0, "inaccessible": 0, "errors": 0}
        
        # Take sample
        sample_urls = urls[:sample_size] if sample_size < len(urls) else urls
        
        for i, url in enumerate(sample_urls, 1):
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    results["accessible"] += 1
                else:
                    results["inaccessible"] += 1
                    self.logger.warning(f"URL inaccessible ({response.status_code}): {url}")
                
            except requests.RequestException as e:
                results["errors"] += 1
                self.logger.error(f"Error checking URL {url}: {e}")
            
            # Rate limiting
            time.sleep(config.request_delay)
            
            if i % 5 == 0:
                self.logger.info(f"Checked {i}/{len(sample_urls)} URLs")
        
        return results
    
    def generate_validation_report(self, database_path: str = None) -> str:
        """Generate a comprehensive validation report"""
        validation_results = self.validate_database(database_path)
        
        if "error" in validation_results:
            return f"Validation Error: {validation_results['error']}"
        
        report = f"""
=== Editorial Database Validation Report ===
Generated: {validation_results['validated_at']}

Total Problems: {validation_results['total_problems']}

Validation Results:
  ✓ Valid URLs: {validation_results['validation_results']['valid_urls']}
  ✗ Invalid URLs: {validation_results['validation_results']['invalid_urls']}
  ✗ Invalid Problem URLs: {validation_results['validation_results']['invalid_problem_urls']}
  ⚠ Missing Editorial IDs: {validation_results['validation_results']['missing_editorial_ids']}

Issues Found: {len(validation_results['issues'])}
"""
        
        if validation_results['issues']:
            report += "\nDetailed Issues:\n"
            for issue in validation_results['issues'][:10]:  # Show first 10 issues
                report += f"  - {issue['problem_key']}: {issue['type']}"
                if 'severity' in issue:
                    report += f" ({issue['severity']})"
                report += "\n"
            
            if len(validation_results['issues']) > 10:
                report += f"  ... and {len(validation_results['issues']) - 10} more issues\n"
        
        return report
    
    def fix_common_issues(self, database_path: str = None) -> Dict[str, int]:
        """Attempt to fix common issues in the database"""
        if database_path is None:
            database_path = config.database_path
        
        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                database = json.load(f)
        except Exception as e:
            return {"error": f"Failed to load database: {e}"}
        
        fixes_applied = {"missing_editorial_ids": 0, "url_format_fixes": 0}
        
        mappings = database.get("editorial_mappings", {})
        
        for problem_key, mapping in mappings.items():
            # Fix missing editorial IDs by re-extracting from URL
            if mapping.get("editorial_id") is None and mapping.get("editorial_url"):
                editorial_id = self.extractor._extract_editorial_id(mapping["editorial_url"])
                if editorial_id:
                    mapping["editorial_id"] = editorial_id
                    fixes_applied["missing_editorial_ids"] += 1
        
        # Save fixed database
        try:
            with open(database_path, 'w', encoding='utf-8') as f:
                json.dump(database, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Applied fixes: {fixes_applied}")
        except Exception as e:
            return {"error": f"Failed to save fixes: {e}"}
        
        return fixes_applied

if __name__ == "__main__":
    validator = EditorialValidator()
    
    print("Validating editorial database...")
    report = validator.generate_validation_report()
    print(report)
    
    print("\nAttempting to fix common issues...")
    fixes = validator.fix_common_issues()
    if "error" not in fixes:
        print(f"Fixes applied: {fixes}")
    else:
        print(f"Fix failed: {fixes['error']}")