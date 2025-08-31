import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TagInferenceConfig:
    """Configuration for tag inference using o4-mini"""
    
    # OpenAI API settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    model_name: str = "o4-mini"  # Using o4-mini model
    
    # Inference settings
    max_tokens: int = 500
    temperature: float = 0.3  # Lower temperature for more consistent results
    
    # Tag settings
    min_tags_per_problem: int = 3
    max_tags_per_problem: int = 5
    
    # Problem range settings for cache building
    default_start_contest: int = 175  # ABC175から開始
    default_end_contest: int = 420    # ABC200まで（テスト用）
    difficulty_threshold: int = 400   # 難易度400以上の問題のみ対象
    target_problems: List[str] = None  # 【廃止予定】['c', 'd', 'e', 'f'] など、固定問題指定（現在は難易度フィルタを使用）
    
    # Batch processing settings
    batch_size: int = 5  # Process 5 problems at once for efficiency
    request_delay: float = 2.0  # Delay between API calls
    concurrent_workers: int = 3  # 同時実行ワーカー数（cache building用）
    use_concurrent: bool = False  # 並行処理を使用するか
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # File paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.target_problems is None:
            # 【廃止予定】現在は使用されない。難易度フィルタで動的に問題を取得
            self.target_problems = ['c', 'd', 'e', 'f']
    
    @property
    def tag_list_path(self) -> str:
        return os.path.join(self.base_dir, "config", "tag_list.txt")
    
    @property
    def problems_with_tags_path(self) -> str:
        return os.path.join(self.base_dir, "data", "problems_with_tags.json")
    
    @property
    def editorial_mappings_path(self) -> str:
        return os.path.join(self.base_dir, "editorial_crawler", "data", "editorial_mappings.json")
    
    def load_available_tags(self) -> List[str]:
        """Load available tags from tag_list.txt"""
        try:
            with open(self.tag_list_path, encoding="utf-8") as f:
                tags = [line.strip() for line in f if line.strip()]
            return tags
        except FileNotFoundError:
            print(f"Warning: Tag list file not found at {self.tag_list_path}")
            return []
    

# Global configuration instance
inference_config = TagInferenceConfig()