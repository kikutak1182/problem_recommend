#!/usr/bin/env python3
"""
高精度版圧縮FastTextモデル作成スクリプト
より大きな語彙サイズと高次元で精度を重視
"""

from gensim.models.fasttext import load_facebook_model
import compress_fasttext as cft
import os

def main():
    print("Creating high-precision compressed FastText model...")
    
    # 元モデルの読み込み
    print("Loading original model...")
    big = load_facebook_model('cc.ja.300.bin').wv
    print(f"Original model vocab size: {len(big.key_to_index)}")
    
    # 高精度設定で圧縮
    print("Compressing with high-precision settings...")
    small = cft.prune_ft_freq(
        big,
        new_vocab_size=300_000,    # 3倍に増加（より多くの語彙を保持）
        new_ngrams_size=600_000,   # 3倍に増加（より多くのサブワードを保持）
        pq=True,                   # Product Quantization
        qdim=300                   # 最大次元（元と同じ次元を保持）
    )
    
    # 保存
    output_path = 'cc.ja.300.high_precision.bin'
    print(f"Saving to {output_path}...")
    small.save(output_path)
    
    # ファイルサイズ確認
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"High-precision model saved: {file_size:.1f} MB")
    
    # 語彙数確認
    print(f"Compressed model vocab size: {len(small.key_to_index)}")
    
    print("High-precision compressed model creation completed!")

if __name__ == "__main__":
    main()