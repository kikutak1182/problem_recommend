from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# 1. モデルのロード（MiniLM系が軽くておすすめ）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. タグリストの読み込み
with open('tag_list.txt', encoding='utf-8') as f:
    tags = [line.strip() for line in f if line.strip()]

# 3. ベクトル化
tag_vectors = model.encode(tags, show_progress_bar=True)

# 4. 保存（タグとベクトルのペアで保存）
with open('tag_vectors.pkl', 'wb') as f:
    pickle.dump({'tags': tags, 'vectors': tag_vectors}, f)

print('タグベクトルの保存が完了しました')