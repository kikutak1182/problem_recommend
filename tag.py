import os
import json
import openai
import random
import re
import requests
from bs4 import BeautifulSoup, Tag

# OpenAI APIキーを環境変数から取得
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# problems.jsonから全件取得
with open("problems.json", encoding="utf-8") as f:
    problems = json.load(f)

# ABC212〜ABC407のC〜G問題のみ抽出
def is_target_abc_problem(p):
    m = re.match(r"abc(\d+)", p["contest_id"])
    if not m:
        return False
    num = int(m.group(1))
    return 212 <= num <= 407 and p["problem_index"] in ["C", "D", "E", "F", "G"]

abc_cg_problems = [p for p in problems if is_target_abc_problem(p)]

# tag-list.txtを読み込む
with open("tag_list.txt", encoding="utf-8") as f:
    tag_examples = [line.strip() for line in f if line.strip()]

# カンマ区切りの文字列に変換
tag_examples_str = ", ".join(tag_examples)

def get_tags_gpt(title, url, editorial_text):
    prompt = f"""
次のAtCoder問題タイトル・URL・解説文から、考えられるアルゴリズムタグを日本語で最大5つまで挙げてください。
タグの候補は以下の中から選んでください（必要に応じて複数選択可）：
{tag_examples_str}
タイトル: {title}
URL: {url}
解説: {editorial_text[:500]}
タグはカンマ区切りのJSON配列で返してください（例: [\"DP\", \"二分探索\"]）
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=100,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return content.strip() if content else "(No response)"

# 既存のキャッシュを読み込む
CACHE_FILE = "tags_sample.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

def resolve_redirect(url: str, timeout: int = 10) -> str:
    """
    与えられた URL がリダイレクトを含む場合、最終到達 URL を返す。
    途中で失敗したら元の URL を返す。
    """
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        return r.url
    except Exception:
        return url

def fetch_problem_editorial_url(contest_id: str, problem_index: str, task_id: str):
    # ① まず「解説 / Editorial」ボタンの URL を拾う
    task_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{task_id}"
    try:
        res = requests.get(task_url, timeout=10)
        if res.ok:
            soup = BeautifulSoup(res.text, "html.parser")
            btn = soup.select_one('a[href$="/editorial"]')
            if btn:
                tmp = "https://atcoder.jp" + btn["href"]
                url = resolve_redirect(tmp)
                print(f"抽出した個別解説ページ: {url}")
                return url
    except Exception:
        pass

    # ② 取れなければ従来ロジック …
    editorial_url = f"https://atcoder.jp/contests/{contest_id}/editorial"
    try:
        res = requests.get(editorial_url, timeout=10)
        if res.status_code != 200:
            print(f"抽出した個別解説ページ: {editorial_url}")
            return editorial_url  # fallback
        soup = BeautifulSoup(res.text, "html.parser")
        candidates = []
        for a in soup.find_all("a", href=True):
            if not isinstance(a, Tag):
                continue
            href_str = str(a.get("href", ""))
            text = a.text.strip()
            # 個別解説ページのパターン
            if href_str.startswith(f"/contests/{contest_id}/editorial/"):
                if text.startswith(f"{problem_index} ") or text.startswith(f"{problem_index}-") or text == problem_index:
                    url = "https://atcoder.jp" + str(href_str)
                    candidates.append(url)
        if candidates:
            url = resolve_redirect(candidates[0])
            print(f"抽出した個別解説ページ: {url}")
            return url
        print(f"抽出した個別解説ページ: {editorial_url}")
        return editorial_url  # fallback
    except Exception as e:
        print(f"抽出した個別解説ページ: {editorial_url}")
        return editorial_url  # fallback

def fetch_editorial_text(editorial_url):
    try:
        res = requests.get(editorial_url, timeout=10)
        if res.status_code != 200:
            return ""
        soup = BeautifulSoup(res.text, "html.parser")
        main = soup.find("div", id="main-container")
        if main:
            return main.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        return ""

# ランダムに2問だけ選ぶ
sample_problems = random.sample(abc_cg_problems, 2)

for p in sample_problems:
    pid = p["id"]
    if pid in results:
        continue  # 既に推論済みならスキップ

    title = p.get("title") or p.get("name")
    contest_id = p["contest_id"]
    problem_index = p["problem_index"]
    url = f"https://atcoder.jp/contests/{contest_id}/tasks/{pid}"

    editorial_url = fetch_problem_editorial_url(contest_id, problem_index, pid)
    editorial_text = fetch_editorial_text(editorial_url)
    tags = get_tags_gpt(title, url, editorial_text)
    results[pid] = {
        "title": title,
        "url": url,
        "editorial_url": editorial_url,
        "tags": tags
    }
    print(f"問題ID: {pid}")
    print(f"タイトル: {title}")
    print(f"URL: {url}")
    print(f"解説リンク: {editorial_url}")
    print(f"推定タグ: {tags}")
    print("-" * 40)

    # 1問ごとにキャッシュ保存
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)