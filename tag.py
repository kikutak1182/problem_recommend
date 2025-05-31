import os
import json
import openai
import random

# OpenAI APIキーを環境変数から取得
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# problems.jsonから全件取得
with open("problems.json", encoding="utf-8") as f:
    problems = json.load(f)

# ABCのD〜F問題のみ抽出
abc_df_problems = [
    p for p in problems
    if p["contest_id"].startswith("abc") and p["problem_index"] in ["D", "E", "F"]
]

# ランダムに2問選ぶ
sample_problems = random.sample(abc_df_problems, 2)

# tag-list.txtを読み込む
with open("tag_list.txt", encoding="utf-8") as f:
    tag_examples = [line.strip() for line in f if line.strip()]

# カンマ区切りの文字列に変換
tag_examples_str = ", ".join(tag_examples)

def get_tags_gpt(title, url):
    prompt = f"""
次のAtCoder問題タイトル・URLから、考えられるアルゴリズムタグを日本語で最大5つまで挙げてください。
タグの候補は以下の中から選んでください（必要に応じて複数選択可）：
{tag_examples_str}
タイトル: {title}
URL: {url}
タグはカンマ区切りのJSON配列で返してください（例: ["DP", "二分探索"]）
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return content.strip() if content else "(No response)"

for p in sample_problems:
    pid = p["id"]
    title = p.get("title") or p.get("name")
    contest_id = p["contest_id"]
    url = f"https://atcoder.jp/contests/{contest_id}/tasks/{pid}"

    print(f"問題ID: {pid}")
    print(f"タイトル: {title}")
    print(f"URL: {url}")
    tags = get_tags_gpt(title, url)
    print(f"推定タグ: {tags}")
    print("-" * 40)