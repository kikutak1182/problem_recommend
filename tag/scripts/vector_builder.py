from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# 設定ファイルをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.tag_config import config

def build_tag_vectors():
    """
タグのベクトル表現を構築して保存する
    """
    print(f"Loading model: {config.sentence_transformer_model}")
    model = SentenceTransformer(config.sentence_transformer_model)
    
    # タグリストの読み込み
    print(f"Loading tags from: {config.tag_list_path}")
    with open(config.tag_list_path, encoding='utf-8') as f:
        tags = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(tags)} tags")
    
    # ベクトル化
    print("Encoding tags...")
    tag_vectors = model.encode(tags, show_progress_bar=True)
    
    # メタデータ付きで保存
    save_data = {
        'tags': tags,
        'vectors': tag_vectors,
        'metadata': {
            'model_name': config.sentence_transformer_model,
            'created_at': datetime.now().isoformat(),
            'total_tags': len(tags),
            'vector_dimension': tag_vectors.shape[1]
        }
    }
    
    print(f"Saving to: {config.tag_vectors_path}")
    with open(config.tag_vectors_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f'タグベクトルの保存が完了しました ({len(tags)} tags, {tag_vectors.shape[1]} dimensions)')
    return save_data

if __name__ == "__main__":
    build_tag_vectors()