from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import uvicorn
import os
import pickle
import numpy as np
import json
import re
import logging

app = FastAPI()
templates = Jinja2Templates(directory="/app/app/templates")

import sys
import re

"""
起動時設定 - すべてのパスをルートディレクトリ基準で指定
- 環境変数で使用データ JSON を切り替え可能:
  - DATA_PATH: 絶対パス（.json まで指定）
  - DATA_BASENAME: `/app/app/data/` 配下のベース名（拡張子不要）
    例: DATA_BASENAME=abc175_420_final_results_20250831_164430
  どちらも未指定なら既定ファイルを使用。
"""

# Define paths from root directory (absolute paths)
tag_vectors_path = "/app/app/data/tag_vectors.pkl"
default_data_path = "/app/app/data/problems_data.json"

# Resolve data path from environment variables
env_data_path = os.environ.get("DATA_PATH") or os.environ.get("RESULTS_PATH")
env_basename = os.environ.get("DATA_BASENAME") or os.environ.get("RESULTS_BASENAME")

if env_data_path:
    # Use absolute path directly
    unified_results_path = env_data_path
elif env_basename:
    basename = env_basename if env_basename.endswith(".json") else env_basename + ".json"
    unified_results_path = f"/app/app/data/{basename}"
else:
    unified_results_path = default_data_path

# Sentence transformer model path (used for logging only)
sentence_transformer_model = os.environ.get("MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print(f"📁 Tag vectors path: {tag_vectors_path}")
print(f"📁 Problem data path: {unified_results_path}")
print(f"🤖 Using ONNX embedding API instead of: {sentence_transformer_model}")

# Load tag vectors (pre-computed embeddings)
try:
    with open(tag_vectors_path, 'rb') as f:
        data = pickle.load(f)
    tags = data["tags"]
    tag_vectors = np.array(data["vectors"])
except FileNotFoundError:
    raise FileNotFoundError(
        f"Tag vectors not found: {tag_vectors_path}. Ensure 'app/data/tag_vectors.pkl' exists."
    )

# ONNX Embedding API configuration
ONNX_API_URL = os.environ.get("ONNX_API_URL", "http://localhost:8081")
print(f"🔗 ONNX API URL: {ONNX_API_URL}")

def get_embeddings(texts):
    """Get embeddings from ONNX API instead of sentence-transformers"""
    try:
        if isinstance(texts, str):
            texts = [texts]
        
        response = requests.post(
            f"{ONNX_API_URL}/embed",
            json={"texts": texts},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return np.array(result["vectors"])
        
    except requests.RequestException as e:
        logging.error(f"❌ Failed to get embeddings from ONNX API: {e}")
        raise RuntimeError(f"Embedding service unavailable: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error in embedding: {e}")
        raise

# Create tag-to-vector mapping for fast lookup
tag_to_vector = {tag: tag_vectors[i] for i, tag in enumerate(tags)}

# Load problem data (supports both unified and standard formats)
try:
    with open(unified_results_path, encoding="utf-8") as f:
        loaded = json.load(f)
        if isinstance(loaded, dict) and "results" in loaded and isinstance(loaded["results"], dict):
            # Standard format: { test_info, confidence_analysis, results: { pid -> data } }
            problems = loaded["results"]
        elif isinstance(loaded, dict):
            # Unified format: { pid -> data }
            problems = loaded
        else:
            raise ValueError("Invalid data format: expected dict with results or direct pid mapping")
    
    print(f"📊 Loaded {len(problems)} problems")
    
except FileNotFoundError:
    raise FileNotFoundError(f"Problem data not found: {unified_results_path}")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in problem data: {e}")

def split_text(text, max_length=100):
    """Split text into words while respecting max_length"""
    words = text.split()
    if len(words) <= max_length:
        return [text]
    
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

def cosine_similarity_vectorized(A, B):
    """Compute cosine similarity between two sets of vectors."""
    dot_products = np.dot(A, B.T)
    norms_A = np.linalg.norm(A, axis=1)
    norms_B = np.linalg.norm(B, axis=1)
    
    # Prevent division by zero
    norms_A = np.where(norms_A == 0, 1e-10, norms_A)
    norms_B = np.where(norms_B == 0, 1e-10, norms_B)
    
    return dot_products / np.outer(norms_A, norms_B)

def rule_based_tag_matching(query, tags, top_n=10):
    """Rule-based exact tag matching"""
    query_lower = query.lower()
    matched_tags = []
    
    for tag in tags:
        tag_lower = tag.lower()
        
        # Exact match (highest priority)
        if tag_lower == query_lower:
            matched_tags.append((tag, 1.0))
        # Contains match
        elif tag_lower in query_lower or query_lower in tag_lower:
            matched_tags.append((tag, 0.8))
    
    # Sort by score and return top N
    matched_tags.sort(key=lambda x: x[1], reverse=True)
    return matched_tags[:top_n]

def get_embedding_based_tags(query, top_n=20):
    """Get similar tags using embeddings"""
    try:
        # Get query embedding using ONNX API
        query_embedding = get_embeddings([query])[0]  # Get first (and only) result
        
        # Compute similarities with all tag vectors
        similarities = cosine_similarity_vectorized(
            query_embedding.reshape(1, -1), tag_vectors
        )[0]
        
        # Get top similar tags
        top_indices = np.argsort(similarities)[::-1][:top_n]
        similar_tags = [(tags[i], float(similarities[i])) for i in top_indices]
        
        return similar_tags
        
    except Exception as e:
        print(f"❌ Embedding-based tag matching failed: {e}")
        return []

def recommend_problems_by_tags_and_rate(query, target_tags, min_rate=None, max_rate=None, excluded_tags=None, top_n=20):
    """Recommend problems based on tags and difficulty rate"""
    if excluded_tags is None:
        excluded_tags = set()
    else:
        excluded_tags = set(excluded_tags)
    
    target_tag_set = set(target_tags)
    candidates = []
    
    for problem_id, data in problems.items():
        problem_tags = set(data.get("tags", []))
        
        # Skip if has excluded tags
        if problem_tags.intersection(excluded_tags):
            continue
            
        # Calculate tag match score
        common_tags = target_tag_set.intersection(problem_tags)
        tag_score = len(common_tags) / len(target_tag_set) if target_tag_set else 0
        
        if tag_score > 0:  # Only include problems with at least one matching tag
            rate = data.get("rate")
            
            # Apply rate filtering if specified
            if min_rate is not None and (rate is None or rate < min_rate):
                continue
            if max_rate is not None and (rate is None or rate > max_rate):
                continue
            
            candidates.append({
                "problem_id": problem_id,
                "tag_score": tag_score,
                "rate": rate,
                "tags": list(problem_tags),
                "data": data
            })
    
    # Sort by tag score (descending), then by rate (ascending for easier problems first)
    candidates.sort(key=lambda x: (-x["tag_score"], x["rate"] or float('inf')))
    
    return candidates[:top_n]

def recommend_problems_with_weighted_similarity(query, top_n=20, min_rate=None, max_rate=None, excluded_tags=None):
    """Advanced recommendation with weighted tag similarity"""
    if excluded_tags is None:
        excluded_tags = set()
    else:
        excluded_tags = set(excluded_tags)
    
    # Get query embedding
    try:
        query_embedding = get_embeddings([query])[0]
    except Exception as e:
        print(f"❌ Failed to get query embedding: {e}")
        return []
    
    # Prepare expanded queries for better matching
    expanded_queries = [
        query,
        query + " アルゴリズム",
        query + " 問題",
        query.replace(" ", "")  # Remove spaces for Japanese
    ]
    
    try:
        query_vecs = get_embeddings(expanded_queries)
    except Exception as e:
        print(f"❌ Failed to get expanded query embeddings: {e}")
        query_vecs = query_embedding.reshape(1, -1)  # Fallback to single query
    
    candidates = []
    
    for problem_id, data in problems.items():
        problem_tags = set(data.get("tags", []))
        
        # Skip if has excluded tags
        if problem_tags.intersection(excluded_tags):
            continue
        
        rate = data.get("rate")
        
        # Apply rate filtering if specified
        if min_rate is not None and (rate is None or rate < min_rate):
            continue
        if max_rate is not None and (rate is None or rate > max_rate):
            continue
        
        # Calculate weighted similarity score
        tag_similarities = []
        for tag in problem_tags:
            if tag in tag_to_vector:
                tag_vec = tag_to_vector[tag].reshape(1, -1)
                # Use maximum similarity across all query variations
                sims = cosine_similarity_vectorized(query_vecs, tag_vec)
                max_sim = np.max(sims)
                tag_similarities.append(max_sim)
        
        if tag_similarities:
            # Use mean of top similarities as overall score
            tag_similarities.sort(reverse=True)
            top_sims = tag_similarities[:3]  # Top 3 matching tags
            weighted_score = np.mean(top_sims)
            
            candidates.append({
                "problem_id": problem_id,
                "weighted_score": weighted_score,
                "rate": rate,
                "tags": list(problem_tags),
                "data": data,
                "top_tag_similarities": top_sims
            })
    
    # Sort by weighted score (descending)
    candidates.sort(key=lambda x: -x["weighted_score"])
    
    # Add some randomization to top results to provide variety
    import random
    if len(candidates) > top_n:
        # Take top 50% deterministically, then randomly sample from next 50%
        deterministic_count = max(1, top_n // 2)
        remaining_count = top_n - deterministic_count
        
        top_candidates = candidates[:deterministic_count]
        remaining_candidates = candidates[deterministic_count:top_n * 2]
        
        if remaining_candidates:
            random_sample = random.sample(
                remaining_candidates, 
                min(remaining_count, len(remaining_candidates))
            )
            candidates = top_candidates + random_sample
        else:
            candidates = top_candidates
    
    return candidates[:top_n]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request: Request,
    query: str = Form(...),
    min_rate: int = Form(None),
    max_rate: int = Form(None),
    excluded_tags: str = Form(""),
    recommendation_method: str = Form("weighted")
):
    try:
        # Parse excluded tags
        excluded_tags_list = [tag.strip() for tag in excluded_tags.split(",") if tag.strip()] if excluded_tags else []
        
        print(f"🔍 Query: {query}")
        print(f"📊 Rate range: {min_rate}-{max_rate}")
        print(f"❌ Excluded tags: {excluded_tags_list}")
        print(f"🎯 Method: {recommendation_method}")
        
        # Get rule-based tags
        rule_based_tags = rule_based_tag_matching(query, tags, top_n=5)
        print(f"📋 Rule-based tags: {rule_based_tags}")
        
        # Get embedding-based tags
        embedding_based_tags = get_embedding_based_tags(query, top_n=10)
        print(f"🧠 Embedding-based tags: {embedding_based_tags[:3]}...")  # Show top 3
        
        # Combine and deduplicate tags
        all_tags = {}
        for tag, score in rule_based_tags:
            all_tags[tag] = max(all_tags.get(tag, 0), score)
        
        for tag, score in embedding_based_tags:
            # Boost rule-based matches, otherwise use embedding score
            if tag in all_tags:
                all_tags[tag] = max(all_tags[tag], score * 1.2)  # Boost combined matches
            else:
                all_tags[tag] = score
        
        # Get top tags for display
        top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Choose recommendation method
        if recommendation_method == "weighted":
            recommendations = recommend_problems_with_weighted_similarity(
                query, top_n=20, min_rate=min_rate, max_rate=max_rate, excluded_tags=excluded_tags_list
            )
        else:
            # Tag-based method as fallback
            target_tags = [tag for tag, _ in top_tags[:5]]  # Use top 5 tags
            recommendations = recommend_problems_by_tags_and_rate(
                query, target_tags, min_rate=min_rate, max_rate=max_rate, excluded_tags=excluded_tags_list, top_n=20
            )
        
        print(f"💡 Found {len(recommendations)} recommendations")
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "query": query,
            "min_rate": min_rate,
            "max_rate": max_rate,
            "excluded_tags": excluded_tags,
            "recommendations": recommendations,
            "top_tags": top_tags[:5],  # Limit to 5 tags for display
            "recommendation_method": recommendation_method
        })
        
    except Exception as e:
        print(f"❌ Error in recommendation: {e}")
        import traceback
        traceback.print_exc()
        
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting lightweight problem recommendation service on port {port}")
    print(f"🔗 Using ONNX API at: {ONNX_API_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)