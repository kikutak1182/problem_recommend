from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import uvicorn
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

import sys
import re
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG_DIR = os.path.abspath(os.path.join(BASE_DIR, "tag"))
sys.path.append(TAG_DIR)

"""
起動時設定
- 環境変数で使用データ JSON を切り替え可能:
  - DATA_PATH: 絶対/相対パス（.json まで指定）
  - DATA_BASENAME: `app/data/` 配下のベース名（拡張子不要）
    例: DATA_BASENAME=abc175_420_final_results_20250831_164430
  どちらも未指定なら既定ファイルを使用。
"""

# Define paths directly since config is not available
tag_vectors_path = os.path.join(TAG_DIR, "vectors/tag_vectors.pkl")

# Resolve data path from env or fallback
default_data_path = os.path.join(
    BASE_DIR, "app/data/problems_data.json"
)
env_data_path = os.environ.get("DATA_PATH") or os.environ.get("RESULTS_PATH")
env_basename = os.environ.get("DATA_BASENAME") or os.environ.get("RESULTS_BASENAME")

if env_data_path:
    # Normalize to absolute if relative path given
    unified_results_path = (
        env_data_path
        if os.path.isabs(env_data_path)
        else os.path.join(BASE_DIR, env_data_path)
    )
elif env_basename:
    basename = env_basename if env_basename.endswith(".json") else env_basename + ".json"
    unified_results_path = os.path.join(BASE_DIR, "app", "data", basename)
else:
    unified_results_path = default_data_path

sentence_transformer_model = "all-MiniLM-L6-v2"
top_k_tags = 10

# Load tag vectors and data
try:
    with open(tag_vectors_path, "rb") as f:
        data = pickle.load(f)
    tags = data["tags"]
    tag_vectors = np.array(data["vectors"])
except FileNotFoundError:
    raise FileNotFoundError(
        f"Tag vectors not found: {tag_vectors_path}. Ensure 'tag/vectors/tag_vectors.pkl' exists."
    )

# Load sentence transformer model
model = SentenceTransformer(sentence_transformer_model)

# Load problem data (supports both unified and standard formats)
try:
    with open(unified_results_path, encoding="utf-8") as f:
        loaded = json.load(f)
        if isinstance(loaded, dict) and "results" in loaded and isinstance(loaded["results"], dict):
            # Standard format: { test_info, confidence_analysis, results: { pid -> data } }
            tag_problems = loaded["results"]
        else:
            # Unified format: { pid -> data }
            tag_problems = loaded
except FileNotFoundError:
    raise FileNotFoundError(
        f"Data file not found: {unified_results_path}. Set DATA_PATH or DATA_BASENAME correctly."
    )

def format_problem_title(problem_id: str, original_title: str) -> str:
    """Format problem title to 'ABC000 C. Problem Name' format"""
    # Extract contest and problem from problem_id (e.g., abc322_f -> ABC322 F)
    match = re.match(r'([a-z]+)(\d+)_([a-z0-9]+)', problem_id)
    if match:
        contest_type, contest_num, problem_letter = match.groups()
        # Format contest type to uppercase
        formatted_contest = f"{contest_type.upper()}{contest_num}"
        # Handle both letter (a-z) and number (1-9) formats
        if problem_letter.isdigit():
            # Convert number to corresponding letter (1->A, 2->B, etc.)
            letter_index = int(problem_letter) - 1
            if 0 <= letter_index < 26:
                formatted_letter = chr(ord('A') + letter_index)
            else:
                formatted_letter = problem_letter
        else:
            formatted_letter = problem_letter.upper()
        
        # Clean the original title - remove existing letter prefix if present
        cleaned_title = original_title
        if original_title:
            # Remove patterns like "A. ", "B. ", etc. from the beginning
            cleaned_title = re.sub(r'^[A-Z]\.\s*', '', original_title)
        
        return f"{formatted_contest} {formatted_letter}. {cleaned_title}"
    else:
        # Fallback for non-standard format
        return f"{problem_id}. {original_title}"

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def recommend(request: Request, username: str = Form(""), queries: str = Form("")):
    # レート取得
    current_rate = None
    if username.strip():
        # ユーザー名が入力されている場合のみレート取得
        user_info_url = f'https://atcoder.jp/users/{username}/history/json'
        try:
            user_info = requests.get(user_info_url).json()
            if user_info:
                current_rate = user_info[-1]['NewRating']
            else:
                return templates.TemplateResponse("index.html", {"request": request, "error": "ユーザー情報が取得できませんでした。", "result": None, "username": username, "queries": queries})
        except Exception as e:
            return templates.TemplateResponse("index.html", {"request": request, "error": "ユーザー情報の取得中にエラーが発生しました。", "result": None, "username": username, "queries": queries})
    else:
        # ユーザー名が未入力の場合はレート制限なし
        current_rate = None

    # Check if queries are provided for tag-based recommendation
    if queries.strip():
        # Tag-based recommendation
        # 文字化け対応：UTF-8でデコード
        try:
            if isinstance(queries, bytes):
                queries = queries.decode('utf-8')
            else:
                # 既に文字列の場合、Latin-1でエンコードしてUTF-8でデコード
                queries = queries.encode('latin-1').decode('utf-8')
        except:
            pass  # デコードできない場合はそのまま
            
        query_list = [q.strip() for q in queries.splitlines() if q.strip()]
        if not query_list:
            return templates.TemplateResponse("index.html", {"request": request, "error": "単語・フレーズを入力してください", "result": None, "username": username, "queries": queries})

        # タグエイリアス辞書（表記揺れ対応）
        # BEGIN_TAG_ALIASES (auto-generated; do not edit by hand)
        tag_aliases = {
            "動的計画法": ["DP", "動的計画", "Dynamic Programming", "メモ化", "dp"],
            "桁DP": ["桁DP", "桁dp", "digit DP", "digit dp"],
            "最大流": ["最大流", "max flow", "maximum flow", "マックスフロー", "maxflow"],
            "グラフ": ["グラフ", "graph"],
            "木構造": ["木構造"],
            "幾何": ["幾何", "計算幾何"],
            "Union-Find": ["Union-Find", "UnionFind", "DSU", "Disjoint Set Union", "素集合データ構造", "union find"],
            "セグメント木": ["セグメント木", "セグ木", "Segment Tree", "SegTree", "segtree", "セグメントツリー"],
            "Fenwick Tree": ["Fenwick Tree", "BIT", "Binary Indexed Tree", "Fenwick", "fenwick tree", "フェンウィック木", "フェニック木"],
            "素数": ["素数", "prime", "素数判定", "エラトステネスの篩", "素因数分解"],
            "期待値": ["期待値", "確率", "probability", "expected value", "期待値計算"],
            "累積和": ["累積和", "cumulative sum"],
            "尺取り法": ["尺取り", "sliding window", "しゃくとり"],
            "遅延セグ木": ["遅延セグ木", "lazy segment tree", "遅延評価セグメント木", "lazy propagation", "遅延セグメントツリー", "Lazy Segment"],
            "平衡二分木": ["平衡二分木", "balanced binary tree", "AVL木", "赤黒木", "平衡二分探索木"],
            "深さ優先探索": ["DFS", "深さ優先探索", "depth-first search", "dfs", "深さ優先"],
            "幅優先探索": ["BFS", "幅優先探索", "breadth-first search", "bfs", "幅優先"],
            "ダイクストラ法": ["ダイクストラ", "Dijkstra", "dijkstra", "単一始点最短路"],
            "ワーシャルフロイド法": ["ワーシャルフロイド", "Warshall-Floyd", "Floyd-Warshall", "全点対間最短路"],
            "ベルマンフォード法": ["ベルマンフォード", "Bellman-Ford", "bellman ford"],
            "最小共通祖先": ["LCA", "Lowest Common Ancestor", "最小共通祖先"],
            "木DP": ["木DP", "木dp", "tree DP", "木動的計画法", "木上DP"],
            "全方位木DP": ["全方位木DP", "全方位DP"],
            "強連結成分": ["SCC", "Strongly Connected Components", "強連結成分", "強連結成分分解"],
            "最小費用流": ["最小費用流", "min cost flow", "minimum cost flow", "MCF", "最小コスト流"],
            "高速フーリエ変換": ["FFT", "NTT", "高速フーリエ変換", "Fast Fourier Transform"],
            "形式的冪級数": ["FPS", "形式的冪級数", "Formal Power Series"],
            "確率・期待値DP": ["確率DP", "確率dp", "期待値DP", "期待値dp", "確率/期待値DP", "probability DP", "expected DP"],
            "主客転倒": ["主客転倒"],
            "ダブリング": ["ダブリング", "doubling", "繰り返し二乗法"],
            "二分探索": ["二分探索", "binary search", "答えで二分探索", "バイナリサーチ"],
            "半分全列挙": ["半分全列挙", "半分列挙"],
            "平方分割": ["平方分割"],
            "貪欲法": ["貪欲法", "貪欲に", "greedy", "greedy algorithm", "グリーディー", "グリーディ"],
            "クエリ先読み": ["クエリ先読み", "query ahead", "クエリ逆順", "クエリを逆"],
            "ビット全探索": ["ビット全探索", "bit全探索"],
            "bitDP": ["bitDP", "bitdp", "bit DP", "bit dp", "ビットDP", "bitdp", "集合DP"],
            "接尾辞配列": ["suffix array", "接尾辞配列", "サフィックス配列"],
            "Trie木": ["Trie", "トライ木", "trie", "プレフィックス木", "接頭辞木"],
            "最長共通接頭辞": ["LCP", "Longest Common Prefix"],
            "ローリングハッシュ": ["ローリングハッシュ", "rolling hash"],
            "Z-algorithm": ["Z-algorithm", "z-algorithm", "Zアルゴリズム", "Z algorithm"],
            "写像12相": ["写像12相", "stirling numbers", "スターリング数", "写像の数え上げ"],
            "Grundy数": ["Grundy数", "grundy数", "grundy number", "nim値"],
            "ゲーム理論": ["ゲーム", "ゲーム理論", "2人ゲーム", "勝敗判定"],
            "座標回転": ["座標回転"],
            "牛ゲー": ["牛ゲー"],
            "調和級数": ["調和級数", "harmonic series"],
            "マンハッタン距離": ["マンハッタン距離", "manhattan distance"],
            "構築問題": ["構築"],
            "括弧列": ["括弧列", "括弧対応"],
            "Bitset": ["Bitset", "bitset"],
            "最大公約数": ["GCD", "greatest common divisor", "gcd", "最大公約数", "ユークリッド互除法"],
            "Heavy-Light分解": ["HLD", "Heavy-Light分解", "heavy light decomposition"],
            "XOR演算": ["XOR", "xor", "排他的論理和", "exclusive or"],
            "DAG": ["DAG", "Directed Acyclic Graph", "有向非循環グラフ", "トポロジカルソート"],
            "カタラン数": ["カタラン数", "Catalan number", "catalan"],
            "シミュレーション": ["シミュレーション"],
            "座標圧縮": ["座標圧縮"],
            "トポロジカルソート": ["トポロジカルソート", "topological sort"],
            "ハミルトン路": ["ハミルトン路", "hamilton path", "hamilton cycle"],
            "バックトラック": ["バックトラック", "backtrack"],
            "ビット演算": ["ビット演算", "bit operation", "bit manipulation", "AND", "OR", "XOR"],
            "メモ化再帰": ["メモ化再帰"],
            "ユークリッドの互除法": ["ユークリッドの互除法", "Euclidean algorithm", "互除法", "拡張ユークリッド"],
            "三分探索": ["三分探索", "ternary search"],
            "中国剰余定理": ["中国剰余定理"],
            "優先度付きキュー": ["優先度付きキュー", "priority queue", "heap", "ヒープ"],
            "全列挙": ["全列挙", "enumeration"],
            "全探索": ["全探索", "brute force", "ブルートフォース", "exhaustive search"],
            "再帰関数": ["再帰関数", "recursive function"],
            "凸包": ["凸包", "convex hull"],
            "分割統治法": ["分割統治法", "divide and conquer", "分割統治"],
            "区間スケジューリング": ["区間スケジューリング", "interval scheduling"],
            "回文": ["回文判定", "palindrome", "回文", "Manacher"],
            "数学": ["数学的考察", "数論", "整数問題"],
            "最小シュタイナー木": ["最小シュタイナー木", "Steiner tree", "シュタイナー木"],
            "最小公倍数": ["最小公倍数", "least common multiple", "lcm", "LCM"],
            "最適化問題": ["最適化", "optimization", "最適化問題", "optimization problem"],
            "素因数分解": ["素因数分解", "prime factorization", "factorization"],
            "組み合わせ論": ["組み合わせ", "combinatorics", "combination", "permutation", "順列", "重複組合せ"],
            "編集距離": ["編集距離", "edit distance", "Levenshtein distance", "レーベンシュタイン距離"],
            "行列": ["行列演算", "行列の演算", "行列の性質"],
            "辞書順": ["辞書順", "lexicographic order", "辞書式順序"],
            "鳩の巣原理": ["鳩の巣原理", "鳩の巣", "pigeonhole principle"],
            "Manacher法": ["Manacher", "manacher"],
            "0-1 BFS": ["0-1 BFS", "01BFS"],
            "二部マッチング": ["二部マッチング", "二部グラフマッチング", "二部グラフの最大マッチング"],
            "二重連結成分": ["二重連結成分", "2-connected components", "block-cut tree"],
            "オイラーツアー": ["オイラーツアー", "Euler tour"],
            "フロー問題": ["フロー", "flow", "ネットワークフロー", "最大流"],
            "最小カット": ["最小カット", "min cut", "minimum cut", "s-t cut"],
            "Monge性": ["Monge", "monge", "Monge性", "四角不等式"],
            "ゼータ変換": ["ゼータ変換", "zeta transform"],
            "Alien DP": ["Alien DP", "slope trick", "傾きトリック", "Lagrange緩和"],
            "包除原理": ["包除原理", "inclusion-exclusion principle"],
            "約数列挙": ["約数列挙"],
            "Wavelet Matrix": ["Wavelet Matrix", "ウェーブレット行列", "多次元範囲クエリ"],
            "永続セグメント木": ["永続セグ木", "persistent segment tree", "永続データ構造"],
            "Mo's algorithm": ["Mo's algorithm", "Mo のアルゴリズム", "Moのアルゴリズム"],
            "いもす法": ["いもす法", "Imos", "imos", "IMOS"],
            "最長増加部分列": ["LIS", "Longest Increasing Subsequence", "最長増加部分列"],
            "巡回セールスマン問題": ["巡回セールスマン問題", "Traveling Salesman Problem", "TSP"],
            "OEIS参照": ["OEIS", "オンライン整数列大辞典", "The On-Line Encyclopedia of Integer Sequences", "数列データベース"],
            "双方向リスト": ["双方向リスト", "doubly linked list", "二重連結リスト"],
            "ハッシュ": ["ハッシュ", "Zobrist"],
            "2-SAT": ["2-SAT", "2SAT", "2-充足可能性問題"],
        }
        # END_TAG_ALIASES

        # 外部ファイルからの読込は行わず、別スクリプトで上記辞書を自動更新します。
        # 逆引き（エイリアス→正規名）も含めた展開
        alias_to_keys = {}
        for key, vals in tag_aliases.items():
            for v in vals:
                alias_to_keys.setdefault(v, set()).add(key)
        
        # エイリアスを考慮したクエリ拡張
        expanded_queries = []
        for q in query_list:
            expanded_queries.append(q)
            # 正引き（キーに一致）
            if q in tag_aliases:
                expanded_queries.extend(tag_aliases[q])
            # 逆引き（別名に一致）
            if q in alias_to_keys:
                expanded_queries.extend(list(alias_to_keys[q]))
        
        # 重複除去
        expanded_queries = list(dict.fromkeys(expanded_queries))
        
        # クエリベクトル化＆類似度計算（拡張されたクエリを使用）
        query_vecs = model.encode(expanded_queries)
        interest_vec = np.mean(query_vecs, axis=0)
        sims = tag_vectors @ interest_vec / (np.linalg.norm(tag_vectors, axis=1) * np.linalg.norm(interest_vec) + 1e-8)
        
        # エイリアス完全一致したタグに高いスコアを付与
        for query in query_list:
            # 正引きエイリアス
            if query in tag_aliases:
                for alias_tag in tag_aliases[query]:
                    if alias_tag in tags:
                        idx = tags.index(alias_tag)
                        sims[idx] = max(sims[idx], 1.0)
            
            # 逆引きエイリアス
            if query in alias_to_keys:
                for canonical_tag in alias_to_keys[query]:
                    if canonical_tag in tags:
                        idx = tags.index(canonical_tag)
                        sims[idx] = max(sims[idx], 1.0)
        
        top_idx = np.argsort(sims)[::-1][:top_k_tags]
        top_tags = [tags[i] for i in top_idx]

        # 問題抽出（改良版タグマッチング）
        recommend = []
        
        # Get external problem metadata for difficulty (handle network errors gracefully)
        try:
            problems = requests.get('https://kenkoooo.com/atcoder/resources/problems.json', timeout=5).json()
            problem_models = requests.get('https://kenkoooo.com/atcoder/resources/problem-models.json', timeout=5).json()
        except:
            # Fallback: use mock difficulty data
            problems = []
            problem_models = {}
            print("Warning: Could not fetch external data, using fallback")
        
        # Pre-calculate query-to-all-tags similarity for efficiency
        # Get all unique tags from problems
        all_problem_tags = set()
        for v in tag_problems.values():
            prob_tags = v.get("tags", [])
            if isinstance(prob_tags, list):
                all_problem_tags.update(prob_tags)
        
        all_problem_tags = list(all_problem_tags)
        
        # Encode all problem tags at once
        if all_problem_tags:
            all_problem_tag_vecs = model.encode(all_problem_tags)
            
            # Calculate similarity matrix: queries vs all problem tags
            query_tag_similarities = {}
            for i, prob_tag in enumerate(all_problem_tags):
                max_sim = 0.0
                
                # Check direct similarity with expanded queries
                for query_vec in query_vecs:
                    sim = np.dot(all_problem_tag_vecs[i], query_vec) / (
                        np.linalg.norm(all_problem_tag_vecs[i]) * np.linalg.norm(query_vec) + 1e-8
                    )
                    max_sim = max(max_sim, sim)
                
                # Check exact matches with aliases (boost score for exact matches)
                for original_query in query_list:
                    if prob_tag == original_query:
                        max_sim = max(max_sim, 1.0)  # Exact match gets highest score
                    # Forward alias: query is canonical name, prob_tag is its alias
                    elif original_query in tag_aliases and prob_tag in tag_aliases[original_query]:
                        max_sim = max(max_sim, 0.95)  # Alias match gets very high score
                    # Reverse alias: query is an alias of some canonical names, and prob_tag matches one of them
                    elif original_query in alias_to_keys and prob_tag in alias_to_keys[original_query]:
                        max_sim = max(max_sim, 0.95)
                
                query_tag_similarities[prob_tag] = max_sim
        else:
            query_tag_similarities = {}
        
        # Enhanced tag matching with confidence scores
        for pid, v in tag_problems.items():
            prob_tags = v.get("tags", [])
            confidence_scores = v.get("confidence_scores", [])
            if isinstance(prob_tags, list) and prob_tags and confidence_scores:
                # Use average confidence score as base relevance
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                
                prob_tags_set = set(prob_tags)
                
                # Method 1: Direct tag matching from top similar tags (based on tag vectors)
                direct_matched_tags = prob_tags_set & set(top_tags)
                direct_count = len(direct_matched_tags)
                
                # Method 2: Direct query-to-problem-tag similarity (handles notation variations)
                semantic_matched_tags = []
                semantic_score = 0.0
                
                for prob_tag in prob_tags:
                    query_sim = query_tag_similarities.get(prob_tag, 0.0)
                    
                    # If similarity is high enough, consider it a semantic match
                    if query_sim > 0.35:  # Threshold for direct query-tag similarity
                        semantic_score += query_sim
                        if prob_tag not in direct_matched_tags:
                            semantic_matched_tags.append(prob_tag)
                
                # Calculate cosine similarity between input queries and problem tags
                if prob_tags:
                    prob_tag_vecs = model.encode(prob_tags)
                    query_vec = np.mean(query_vecs, axis=0)  # Average of all query vectors
                    
                    # Calculate cosine similarities for all tags
                    similarities = []
                    for tag_vec in prob_tag_vecs:
                        sim = np.dot(tag_vec, query_vec) / (
                            np.linalg.norm(tag_vec) * np.linalg.norm(query_vec) + 1e-8
                        )
                        similarities.append(max(0, sim))  # Only keep positive similarities
                    
                    # Use maximum similarity as relevance score
                    relevance_score = max(similarities) if similarities else 0.0
                else:
                    relevance_score = 0.0

                # Alias/semantic match score from precomputed query_tag_similarities (includes exact/alias boosts)
                alias_match = 0.0
                for prob_tag in prob_tags:
                    alias_match = max(alias_match, query_tag_similarities.get(prob_tag, 0.0))

                # Blend cosine and alias/semantic match
                final_relevance = 0.7 * relevance_score + 0.3 * alias_match
                
                all_matched_tags = list(direct_matched_tags) + semantic_matched_tags
                
                # 直接一致がある場合は閾値未満でも採用。閾値は合成関連度で判定。
                if final_relevance > 0.5 or direct_count > 0:
                    # Get difficulty from external API or use default
                    diff = problem_models.get(pid, {}).get('difficulty')
                    if diff is None:
                        continue  # Skip problems without difficulty data
                    
                    # ユーザー名が入力されている場合はレート制限、未入力の場合は全難易度
                    if current_rate is None or diff > current_rate:
                        # Always get title and URL from external API for accuracy
                        problem_info = next((p for p in problems if p['id'] == pid), None)
                        if problem_info:
                            original_title = problem_info['title']
                            contest_id = problem_info['contest_id'] 
                            url = f"https://atcoder.jp/contests/{contest_id}/tasks/{pid}"
                        else:
                            # Fallback to internal data from unified results
                            original_title = v.get("title", "")
                            if not original_title or original_title == pid:
                                original_title = pid  # Use problem ID as last resort
                            url = v.get("problem_url", "")
                        
                        # Format title to unified format (ABC322 F. Problem Name)
                        title = format_problem_title(pid, original_title)
                        
                        # Create tag info with confidence scores (truncated to 1 decimal)
                        tag_info = []
                        for i, tag in enumerate(prob_tags):
                            conf = confidence_scores[i] if i < len(confidence_scores) else 0.0
                            tag_info.append({"tag": tag, "confidence": int(conf * 10) / 10.0})  # Truncate to 1 decimal
                        
                        # Collect ranking keys: prioritize direct tag matches, then ascending difficulty, then higher final relevance
                        recommend.append((direct_count, diff, final_relevance, title, url, tag_info))
        # Rank: 1) direct matches desc, 2) difficulty asc, 3) relevance desc
        recommend.sort(key=lambda x: (-x[0], x[1], -x[2]))
        result = [
            {
                "title": title,
                "url": url,
                "tags": tag_info,
                "diff": diff,
                "relevance_score": round(relevance, 3)
            }
            for direct_count, diff, relevance, title, url, tag_info in recommend[:10]
        ]
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "username": username, "rate": current_rate, "top_tags": top_tags, "queries": queries})
    
    else:
        # Rate-based recommendation (fallback)
        problems = requests.get('https://kenkoooo.com/atcoder/resources/problems.json').json()
        problem_models = requests.get('https://kenkoooo.com/atcoder/resources/problem-models.json').json()

        # レートとdiffが近い問題を抽出
        recommend = []
        for p in problems:
            pid = p['id']
            title = p['title']
            contest_id = p['contest_id']
            diff = problem_models.get(pid, {}).get('difficulty')
            if diff is not None:
                if current_rate is not None:
                    diff_gap = abs(diff - current_rate)
                    recommend.append((diff_gap, contest_id, title, diff, pid))
                else:
                    # ユーザー名未入力の場合はランダムソート
                    import random
                    diff_gap = random.random()
                    recommend.append((diff_gap, contest_id, title, diff, pid))
        recommend.sort()
        result = [
            {
                "title": format_problem_title(pid, title),
                "url": f"https://atcoder.jp/contests/{contest_id}/tasks/{pid}",
                "tags": [],
                "diff": diff
            }
            for _, contest_id, title, diff, pid in recommend[:10]
        ]
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "username": username, "rate": current_rate, "queries": queries})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True) 
