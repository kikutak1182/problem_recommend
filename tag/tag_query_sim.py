import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import re
import requests

# 1. ベクトルとタグの読み込み
with open("tag/tag_vectors.pkl", "rb") as f:
    data = pickle.load(f)
tags = data["tags"]
tag_vectors = np.array(data["vectors"])

# 2. モデルのロード
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 3. 複数単語・フレーズの入力
queries = []
print("興味のあるアルゴリズムや単語・フレーズを複数入力してください（空行で終了）:")
while True:
    q = input("> ").strip()
    if not q:
        break
    queries.append(q)

if not queries:
    print("入力がありませんでした。")
    exit()

# 4. すべてをベクトル化し、平均ベクトルを作る
query_vecs = model.encode(queries)
interest_vec = np.mean(query_vecs, axis=0)

# 5. コサイン類似度計算
sims = tag_vectors @ interest_vec / (np.linalg.norm(tag_vectors, axis=1) * np.linalg.norm(interest_vec) + 1e-8)

# 6. 上位5件のタグを表示
top_idx = np.argsort(sims)[::-1][:5]
top_tags = [tags[i] for i in top_idx]
print("類似度上位タグ:")
for i in top_idx:
    print(f"{tags[i]} (score: {sims[i]:.3f})")

# 7. ユーザーネームからレート取得
username = input("\nAtCoderのユーザーネームを入力してください: ").strip()
if not username:
    print("ユーザーネームが入力されていません。")
    exit()
try:
    user_info_url = f'https://atcoder.jp/users/{username}/history/json'
    user_info = requests.get(user_info_url).json()
    if user_info:
        current_rate = user_info[-1]['NewRating']
        print(f"現在レート: {current_rate}")
    else:
        print("ユーザー情報が取得できませんでした。")
        exit()
except Exception as e:
    print("ユーザー情報取得時にエラー:", e)
    exit()

# 8. tags_sample1.jsonから該当タグを持つ問題を抽出
with open("tag/tags_sample1.json", encoding="utf-8") as f:
    problems = json.load(f)

# 9. problem-models.jsonからdiff情報を取得
with open("tag/problem-models.json", encoding="utf-8") as f:
    problem_models = json.load(f)

recommend = []
for pid, v in problems.items():
    tags_str = v.get("tags", "")
    m = re.search(r"\[.*\]", tags_str, re.DOTALL)
    if m:
        try:
            prob_tags = set(json.loads(m.group(0)))
            if prob_tags & set(top_tags):
                diff = problem_models.get(pid, {}).get('difficulty')
                if diff is not None:
                    diff_gap = abs(diff - current_rate)
                    recommend.append((diff_gap, v["title"], v["url"], list(prob_tags & set(top_tags)), diff))
        except Exception:
            pass

# 10. レート差が近い順にソートし、最大10件表示
recommend.sort()
print("\nおすすめ問題（上位タグ＆レート近い順）:")
for i, (gap, title, url, matched_tags, diff) in enumerate(recommend[:10], 1):
    print(f"{i}. {title} (diff: {diff}) {url}  [タグ: {', '.join(matched_tags)}]")
if not recommend:
    print("該当する問題が見つかりませんでした。")
