from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import uvicorn
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def recommend(request: Request, username: str = Form(...)):
    # レート取得
    user_info_url = f'https://atcoder.jp/users/{username}/history/json'
    user_info = requests.get(user_info_url).json()
    if user_info:
        current_rate = user_info[-1]['NewRating']
    else:
        return templates.TemplateResponse("index.html", {"request": request, "error": "ユーザー情報が取得できませんでした。", "result": None})

    # 問題リスト・diff情報取得
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
            "contest_id": contest_id,
            "title": title,
            "diff": diff,
            "pid": pid
        }
        for _, contest_id, title, diff, pid in recommend[:10]
    ]
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "username": username, "rate": current_rate})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True) 