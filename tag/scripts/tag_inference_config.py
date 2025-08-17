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
    
    # Batch processing settings
    batch_size: int = 5  # Process 5 problems at once for efficiency
    request_delay: float = 2.0  # Delay between API calls
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 5.0
    
    # File paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
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
    
    def create_tag_synonyms_mapping(self) -> Dict[str, List[str]]:
        """Create mapping of tags to their synonyms for deduplication"""
        synonyms = {
            "DP": ["動的計画法", "動的プログラミング", "Dynamic Programming"],
            "BFS": ["幅優先探索", "breadth-first search"],
            "DFS": ["深さ優先探索", "depth-first search"],
            "グラフ": ["Graph", "graph"],
            "木": ["Tree", "tree"],
            "数学": ["数学的考察", "Math", "mathematics"],
            "貪欲法": ["greedy", "Greedy", "グリーディー"],
            "二分探索": ["binary search", "Binary Search"],
            "ソート": ["sort", "Sort", "sorting"],
            "全探索": ["brute force", "Brute Force", "全列挙"],
            "Union Find": ["UnionFind", "素集合", "DSU"],
            "セグメント木": ["Segment Tree", "SegTree"],
            "Binary Indexed Tree": ["BIT", "Fenwick Tree"],
            "最大流": ["Max Flow", "maximum flow"],
            "最小費用流": ["Min Cost Flow", "minimum cost flow"],
            "LCA": ["Lowest Common Ancestor"],
            "SCC": ["Strongly Connected Components", "強連結成分"],
            "トポロジカルソート": ["Topological Sort", "topological sort"],
            "ダイクストラ": ["Dijkstra", "dijkstra"],
            "ワーシャルフロイド": ["Warshall Floyd", "Floyd Warshall"],
            "ベルマンフォード": ["Bellman Ford", "bellman ford"],
            "FFT": ["NTT", "Fast Fourier Transform"],
            "確率": ["期待値", "probability", "expectation"],
            "組み合わせ": ["combinatorics", "Combinatorics", "コンビネーション"],
            "整数論": ["number theory", "Number Theory", "数論"],
            "幾何": ["geometry", "Geometry", "計算幾何"],
            "文字列": ["string", "String", "文字列処理"],
            "ビット演算": ["bit manipulation", "Bit Manipulation", "bitwise"],
            "累積和": ["cumulative sum", "prefix sum"],
            "尺取り法": ["two pointers", "Two Pointers"],
            "平方分割": ["square root decomposition", "sqrt decomposition"],
            "ダブリング": ["doubling", "Doubling"],
            "半分全列挙": ["meet in the middle", "Meet in the Middle"]
        }
        return synonyms

# Global configuration instance
inference_config = TagInferenceConfig()