import json
import os
import sys
from typing import Dict, List, Tuple, Any

# 設定ファイルをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.tag_config import config

def validate_problems_data() -> Tuple[bool, List[str]]:
    """問題データの妥当性を検証"""
    errors = []
    
    try:
        with open(config.problems_with_tags_path, encoding="utf-8") as f:
            data = json.load(f)
        
        # データ構造の確認
        if "problems" not in data:
            errors.append("'problems' key not found in data")
            return False, errors
        
        if "metadata" not in data:
            errors.append("'metadata' key not found in data")
        
        problems = data["problems"]
        
        # 各問題データの確認
        for problem_id, problem_data in problems.items():
            required_fields = ["title", "url", "tags"]
            for field in required_fields:
                if field not in problem_data:
                    errors.append(f"Problem {problem_id}: missing field '{field}'")
            
            # タグが配列かどうか確認
            if "tags" in problem_data:
                if not isinstance(problem_data["tags"], list):
                    errors.append(f"Problem {problem_id}: 'tags' should be a list, got {type(problem_data['tags'])}")
                else:
                    # 空のタグリストは警告
                    if len(problem_data["tags"]) == 0:
                        errors.append(f"Problem {problem_id}: empty tags list")
        
        print(f"✓ Problems data validation completed. Found {len(problems)} problems.")
        
    except FileNotFoundError:
        errors.append(f"Problems file not found: {config.problems_with_tags_path}")
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in problems file: {e}")
    
    return len(errors) == 0, errors

def validate_tag_vectors() -> Tuple[bool, List[str]]:
    """タグベクトルデータの妥当性を検証"""
    errors = []
    
    try:
        import pickle
        with open(config.tag_vectors_path, "rb") as f:
            data = pickle.load(f)
        
        # データ構造の確認
        if "tags" not in data:
            errors.append("'tags' key not found in vector data")
        if "vectors" not in data:
            errors.append("'vectors' key not found in vector data")
        
        if "tags" in data and "vectors" in data:
            tags = data["tags"]
            vectors = data["vectors"]
            
            if len(tags) != len(vectors):
                errors.append(f"Tag count mismatch: {len(tags)} tags vs {len(vectors)} vectors")
            
            # タグリストファイルとの整合性確認
            try:
                with open(config.tag_list_path, encoding="utf-8") as f:
                    file_tags = [line.strip() for line in f if line.strip()]
                
                if set(tags) != set(file_tags):
                    errors.append("Tags in vector file don't match tags in tag_list.txt")
                    missing_in_vector = set(file_tags) - set(tags)
                    missing_in_file = set(tags) - set(file_tags)
                    if missing_in_vector:
                        errors.append(f"Missing in vector: {missing_in_vector}")
                    if missing_in_file:
                        errors.append(f"Missing in file: {missing_in_file}")
            
            except FileNotFoundError:
                errors.append(f"Tag list file not found: {config.tag_list_path}")
        
        print(f"✓ Tag vectors validation completed. Found {len(data.get('tags', []))} tag vectors.")
        
    except FileNotFoundError:
        errors.append(f"Vector file not found: {config.tag_vectors_path}")
    except Exception as e:
        errors.append(f"Error loading vector file: {e}")
    
    return len(errors) == 0, errors

def validate_tag_definitions() -> Tuple[bool, List[str]]:
    """タグ定義データの妥当性を検証"""
    errors = []
    
    tag_definitions_path = os.path.join(config.data_dir, "tag_definitions.json")
    
    try:
        with open(tag_definitions_path, encoding="utf-8") as f:
            data = json.load(f)
        
        if "tags" not in data:
            errors.append("'tags' key not found in tag definitions")
            return False, errors
        
        # tag_list.txtとの整合性確認
        try:
            with open(config.tag_list_path, encoding="utf-8") as f:
                file_tags = [line.strip() for line in f if line.strip()]
            
            defined_tags = set(data["tags"].keys())
            file_tags_set = set(file_tags)
            
            if defined_tags != file_tags_set:
                missing_definitions = file_tags_set - defined_tags
                extra_definitions = defined_tags - file_tags_set
                
                if missing_definitions:
                    errors.append(f"Missing tag definitions: {missing_definitions}")
                if extra_definitions:
                    errors.append(f"Extra tag definitions: {extra_definitions}")
        
        except FileNotFoundError:
            errors.append(f"Tag list file not found: {config.tag_list_path}")
        
        print(f"✓ Tag definitions validation completed. Found {len(data.get('tags', {}))} tag definitions.")
        
    except FileNotFoundError:
        errors.append(f"Tag definitions file not found: {tag_definitions_path}")
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in tag definitions file: {e}")
    
    return len(errors) == 0, errors

def run_all_validations():
    """すべての検証を実行"""
    print("=== Tag System Data Validation ===\n")
    
    all_valid = True
    all_errors = []
    
    # 問題データの検証
    print("1. Validating problems data...")
    valid, errors = validate_problems_data()
    if not valid:
        all_valid = False
        all_errors.extend([f"Problems: {err}" for err in errors])
    else:
        print("✓ Problems data is valid")
    
    print()
    
    # タグベクトルの検証
    print("2. Validating tag vectors...")
    valid, errors = validate_tag_vectors()
    if not valid:
        all_valid = False
        all_errors.extend([f"Vectors: {err}" for err in errors])
    else:
        print("✓ Tag vectors are valid")
    
    print()
    
    # タグ定義の検証
    print("3. Validating tag definitions...")
    valid, errors = validate_tag_definitions()
    if not valid:
        all_valid = False
        all_errors.extend([f"Definitions: {err}" for err in errors])
    else:
        print("✓ Tag definitions are valid")
    
    print("\n=== Validation Results ===")
    if all_valid:
        print("✅ All validations passed!")
    else:
        print("❌ Validation errors found:")
        for error in all_errors:
            print(f"  - {error}")
    
    return all_valid, all_errors

if __name__ == "__main__":
    run_all_validations()