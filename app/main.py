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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../tag"))
sys.path.append(TAG_DIR)

# Define paths directly since config is not available
tag_vectors_path = os.path.join(TAG_DIR, "vectors/tag_vectors.pkl")
problems_with_tags_path = os.path.join(TAG_DIR, "data/problems_with_tags.json")
sentence_transformer_model = "all-MiniLM-L6-v2"
top_k_tags = 10

# Load tag vectors and data
with open(tag_vectors_path, "rb") as f:
    data = pickle.load(f)
tags = data["tags"]
tag_vectors = np.array(data["vectors"])

# Load sentence transformer model
model = SentenceTransformer(sentence_transformer_model)

# Load problem data
with open(problems_with_tags_path, encoding="utf-8") as f:
    problems_data = json.load(f)
    tag_problems = problems_data.get("problems", {})

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
def recommend(request: Request, username: str = Form(...), queries: str = Form("")):
    # レート取得
    user_info_url = f'https://atcoder.jp/users/{username}/history/json'
    user_info = requests.get(user_info_url).json()
    if user_info:
        current_rate = user_info[-1]['NewRating']
    else:
        return templates.TemplateResponse("index.html", {"request": request, "error": "ユーザー情報が取得できませんでした。", "result": None, "username": username, "queries": queries})

    # Check if queries are provided for tag-based recommendation
    if queries.strip():
        # Tag-based recommendation
        query_list = [q.strip() for q in queries.splitlines() if q.strip()]
        if not query_list:
            return templates.TemplateResponse("index.html", {"request": request, "error": "単語・フレーズを入力してください", "result": None, "username": username, "queries": queries})

        # タグエイリアス辞書（表記揺れ対応）
        tag_aliases = {
            '幅優先探索': ['BFS', '幅優先'],
            '深さ優先探索': ['DFS', '深さ優先'],
            'ダイナミックプログラミング': ['DP', '動的計画法'],
            '動的計画法': ['DP', 'ダイナミックプログラミング'],
            'グラフ理論': ['グラフ'],
            'ベルマンフォード': ['ベルマンフォード法'],
            'ダイクストラ': ['ダイクストラ法'],
            'ワーシャルフロイド': ['フロイドワーシャル', 'ワーシャルフロイド法'],
            '最短経路': ['ダイクストラ', 'ベルマンフォード', 'ワーシャルフロイド'],
            '木構造': ['木', 'Tree'],
            '二分探索': ['binary search', 'BS'],
            '尺取り法': ['two pointer', '二つのポインタ'],
            '貪欲法': ['greedy', 'グリーディ'],
            '全探索': ['brute force', '総当たり'],
            '組み合わせ': ['combination', 'コンビネーション'],
            'セグメント木': ['segment tree', 'セグ木'],
            'Union Find': ['UnionFind', '素集合', 'DSU'],
            '最大流': ['max flow', 'フロー'],
            '最小費用流': ['min cost flow', 'MCF'],
            '素数': ['prime', '素数判定'],
            '素因数分解': ['prime factorization', 'factorization'],
            '数論': ['number theory'],
            '幾何': ['geometry', '計算幾何'],
            '文字列': ['string', 'ストリング'],
            'ハッシュ': ['hash', 'hashing'],
            'ローリングハッシュ': ['rolling hash'],
        }
        
        # エイリアスを考慮したクエリ拡張
        expanded_queries = query_list.copy()
        for query in query_list:
            if query in tag_aliases:
                expanded_queries.extend(tag_aliases[query])
        
        # クエリベクトル化＆類似度計算（拡張されたクエリを使用）
        query_vecs = model.encode(expanded_queries)
        interest_vec = np.mean(query_vecs, axis=0)
        sims = tag_vectors @ interest_vec / (np.linalg.norm(tag_vectors, axis=1) * np.linalg.norm(interest_vec) + 1e-8)
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
                    elif original_query in tag_aliases and prob_tag in tag_aliases[original_query]:
                        max_sim = max(max_sim, 0.95)  # Alias match gets very high score
                
                query_tag_similarities[prob_tag] = max_sim
        else:
            query_tag_similarities = {}
        
        # Enhanced tag matching with pre-calculated similarities
        for pid, v in tag_problems.items():
            prob_tags = v.get("tags", [])
            if isinstance(prob_tags, list) and prob_tags:
                prob_tags_set = set(prob_tags)
                
                # Method 1: Direct tag matching from top similar tags (based on tag vectors)
                direct_matched_tags = prob_tags_set & set(top_tags)
                direct_score = len(direct_matched_tags) * 1.0
                
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
                
                # Calculate total relevance score
                all_matched_tags = list(direct_matched_tags) + semantic_matched_tags
                relevance_score = direct_score + semantic_score
                
                if relevance_score > 0.1:  # At least some relevance
                    # Get difficulty from external API or use default
                    diff = problem_models.get(pid, {}).get('difficulty')
                    if diff is None:
                        diff = current_rate  # Default to user's current rate
                    
                    diff_gap = abs(diff - current_rate)
                    
                    # Get title from external API or use problem_url if title is empty
                    original_title = v.get("title", "")
                    if not original_title:
                        # Find title from external problems data
                        problem_info = next((p for p in problems if p['id'] == pid), None)
                        original_title = problem_info['title'] if problem_info else pid
                    
                    # Format title to unified format (abc322 F. Problem Name)
                    title = format_problem_title(pid, original_title)
                    
                    url = v.get("problem_url", v.get("url", ""))
                    
                    # Sort by relevance first, then by difficulty gap
                    recommend.append((relevance_score, diff_gap, title, url, all_matched_tags, diff))
        # Sort by relevance score (descending), then by difficulty gap (ascending)
        recommend.sort(key=lambda x: (-x[0], x[1]))
        result = [
            {
                "title": title,
                "url": url,
                "tags": matched_tags,
                "diff": diff,
                "relevance_score": round(relevance_score, 3)
            }
            for relevance_score, _, title, url, matched_tags, diff in recommend[:10]
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
                diff_gap = abs(diff - current_rate)
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
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True) 