import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

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
print("類似度上位タグ:")
for i in top_idx:
    print(f"{tags[i]} (score: {sims[i]:.3f})")
