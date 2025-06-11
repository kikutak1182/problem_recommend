import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 1. ベクトルとタグの読み込み
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "../tag/tag_vectors.pkl"), "rb") as f:
    data = pickle.load(f)
tags = data["tags"]
tag_vectors = np.array(data["vectors"])

# 2. モデルのロード
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

with open(os.path.join(BASE_DIR, "../tag/tags_sample1.json"), encoding="utf-8") as f:
    problems = json.load(f)
with open(os.path.join(BASE_DIR, "../tag/problem-models.json"), encoding="utf-8") as f:
    problem_models = json.load(f)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def recommend(request: Request, username: str = Form(...), queries: str = Form(...)):
    # 1. クエリ分割
    query_list = [q.strip() for q in queries.splitlines() if q.strip()]
    if not query_list:
        return templates.TemplateResponse("index.html", {"request": request, "error": "単語・フレーズを入力してください", "result": None})

    # 2. ユーザーレート取得
    user_info_url = f'https://atcoder.jp/users/{username}/history/json'
    user_info = requests.get(user_info_url).json()
    if user_info:
        current_rate = user_info[-1]['NewRating']
    else:
        return templates.TemplateResponse("index.html", {"request": request, "error": "ユーザー情報が取得できませんでした。", "result": None})

    # 3. クエリベクトル化＆類似度計算
    query_vecs = model.encode(query_list)
    interest_vec = np.mean(query_vecs, axis=0)
    sims = tag_vectors @ interest_vec / (np.linalg.norm(tag_vectors, axis=1) * np.linalg.norm(interest_vec) + 1e-8)
    top_idx = np.argsort(sims)[::-1][:5]
    top_tags = [tags[i] for i in top_idx]

    # 4. 問題抽出
    recommend = []
    for pid, v in problems.items():
        tags_str = v.get("tags", "")
        m = re.search(r"\[.*\]", tags_str, re.DOTALL)
        if m:
            try:
                # JSON文字列から余分な部分を削除
                json_str = m.group(0).replace("```json\n", "").replace("\n```", "")
                prob_tags = set(json.loads(json_str))
                if prob_tags & set(top_tags):
                    diff = problem_models.get(pid, {}).get('difficulty')
                    if diff is not None:
                        diff_gap = abs(diff - current_rate)
                        recommend.append((diff_gap, v["title"], v["url"], list(prob_tags & set(top_tags)), diff))
            except Exception as e:
                print(f"Error parsing tags for {pid}: {e}")
                pass
    recommend.sort()
    result = [
        {
            "title": title,
            "url": url,
            "tags": matched_tags,
            "diff": diff
        }
        for _, title, url, matched_tags, diff in recommend[:10]
    ]
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "username": username, "rate": current_rate, "top_tags": top_tags})
