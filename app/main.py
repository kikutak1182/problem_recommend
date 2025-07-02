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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../tag"))

# Load tag vectors and data
with open(os.path.join(TAG_DIR, "tag_vectors.pkl"), "rb") as f:
    data = pickle.load(f)
tags = data["tags"]
tag_vectors = np.array(data["vectors"])

# Load sentence transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load problem data
with open(os.path.join(TAG_DIR, "tags_sample1.json"), encoding="utf-8") as f:
    tag_problems = json.load(f)
with open(os.path.join(TAG_DIR, "problem-models.json"), encoding="utf-8") as f:
    tag_problem_models = json.load(f)

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
        return templates.TemplateResponse("index.html", {"request": request, "error": "ユーザー情報が取得できませんでした。", "result": None})

    # Check if queries are provided for tag-based recommendation
    if queries.strip():
        # Tag-based recommendation
        query_list = [q.strip() for q in queries.splitlines() if q.strip()]
        if not query_list:
            return templates.TemplateResponse("index.html", {"request": request, "error": "単語・フレーズを入力してください", "result": None})

        # クエリベクトル化＆類似度計算
        query_vecs = model.encode(query_list)
        interest_vec = np.mean(query_vecs, axis=0)
        sims = tag_vectors @ interest_vec / (np.linalg.norm(tag_vectors, axis=1) * np.linalg.norm(interest_vec) + 1e-8)
        top_idx = np.argsort(sims)[::-1][:5]
        top_tags = [tags[i] for i in top_idx]

        # 問題抽出
        recommend = []
        for pid, v in tag_problems.items():
            tags_str = v.get("tags", "")
            m = re.search(r"\[.*\]", tags_str, re.DOTALL)
            if m:
                try:
                    json_str = m.group(0).replace("```json\n", "").replace("\n```", "")
                    prob_tags = set(json.loads(json_str))
                    if prob_tags & set(top_tags):
                        diff = tag_problem_models.get(pid, {}).get('difficulty')
                        if diff is not None:
                            diff_gap = abs(diff - current_rate)
                            recommend.append((diff_gap, v["title"], v["url"], list(prob_tags), diff))
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
                "title": title,
                "url": f"https://atcoder.jp/contests/{contest_id}/tasks/{pid}",
                "tags": [],
                "diff": diff
            }
            for _, contest_id, title, diff, pid in recommend[:10]
        ]
        return templates.TemplateResponse("index.html", {"request": request, "result": result, "username": username, "rate": current_rate})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True) 