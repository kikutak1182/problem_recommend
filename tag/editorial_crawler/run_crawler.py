#!/usr/bin/env python3
"""
AtCoder Editorial Crawler - Main execution script

This script crawls AtCoder contest pages to build a database mapping
problem URLs to their official editorial URLs.

Usage:
    python run_crawler.py [options]

Options:
    --test: Run with limited contests for testing
    --contest-types: Specify contest types (abc,arc,agc)
    --limit: Limit number of contests to process
    --validate-only: Only run validation on existing database
    --fix-issues: Attempt to fix common issues in database
"""

import argparse
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_builder import EditorialDatabaseBuilder
from validator import EditorialValidator
from contest_filter import ContestFilter
from crawler_config import config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AtCoder Editorial Crawler - Build problem-editorial URL mapping database"
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run in test mode (process only 5 contests)'
    )
    
    parser.add_argument(
        '--contest-types',
        type=str,
        default='abc,arc,agc',
        help='Contest types to process (comma-separated: abc,arc,agc)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of contests to process'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing database without crawling'
    )
    
    parser.add_argument(
        '--fix-issues',
        action='store_true',
        help='Attempt to fix common issues in existing database'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show contest filtering statistics'
    )
    
    return parser.parse_args()

def show_banner():
    """Display application banner"""
    print("=" * 60)
    print("    AtCoder Editorial Crawler")
    print("    Problem-Editorial URL Mapping Database Builder")
    print("=" * 60)
    print(f"Target Contests: {config.target_contests}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Database File: {config.database_file}")
    print("-" * 60)

def run_crawler(contest_types: list, limit: int = None):
    """Run the editorial crawler"""
    print("\n🚀 Starting Editorial Crawler...")
    
    builder = EditorialDatabaseBuilder()
    
    start_time = datetime.now()
    
    try:
        database = builder.build_database(
            limit_contests=limit,
            contest_types=contest_types
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ Crawling completed in {duration}")
        builder.print_statistics()
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Crawling interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Crawling failed: {e}")
        return False

def run_validation():
    """Run database validation"""
    print("\n🔍 Validating Editorial Database...")
    
    validator = EditorialValidator()
    
    if not os.path.exists(config.database_path):
        print(f"❌ Database file not found: {config.database_path}")
        return False
    
    try:
        report = validator.generate_validation_report()
        print(report)
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def fix_database_issues():
    """Attempt to fix common database issues"""
    print("\n🔧 Fixing Database Issues...")
    
    validator = EditorialValidator()
    
    if not os.path.exists(config.database_path):
        print(f"❌ Database file not found: {config.database_path}")
        return False
    
    try:
        fixes = validator.fix_common_issues()
        
        if "error" in fixes:
            print(f"❌ Fix failed: {fixes['error']}")
            return False
        else:
            print(f"✅ Fixes applied: {fixes}")
            return True
            
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

def show_statistics():
    """Show contest filtering statistics"""
    print("\n📊 Contest Filtering Statistics...")
    
    try:
        filter = ContestFilter()
        filter.print_statistics()
        return True
        
    except Exception as e:
        print(f"❌ Failed to show statistics: {e}")
        return False

def main():
    """Main execution function"""
    args = parse_arguments()
    
    show_banner()
    
    # Show statistics only
    if args.stats_only:
        return 0 if show_statistics() else 1
    
    # Validate only
    if args.validate_only:
        return 0 if run_validation() else 1
    
    # Fix issues only
    if args.fix_issues:
        success = fix_database_issues()
        if success:
            # Run validation after fixes
            run_validation()
        return 0 if success else 1
    
    # Parse contest types
    contest_types = [ct.strip() for ct in args.contest_types.split(',')]
    
    # Set limit
    limit = None
    if args.test:
        limit = 5
        print("🧪 Running in TEST mode (5 contests only)")
    elif args.limit:
        limit = args.limit
        print(f"📏 Processing limited to {limit} contests")
    
    # Show target statistics
    show_statistics()
    
    # Confirm before full crawl
    if not args.test and limit is None:
        print(f"\n⚠️  This will crawl ALL target contests ({config.target_contests})")
        confirm = input("Do you want to continue? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Aborted by user")
            return 0
    
    # Run crawler
    success = run_crawler(contest_types, limit)
    
    if success:
        # Run validation after successful crawl
        print("\n" + "="*60)
        run_validation()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())