# AtCoder タグ推定システム

AtCoder問題のアルゴリズムタグを自動推定するシステムです。OpenAIのo4-miniモデルとembeddingを組み合わせた高精度な推定を行います。

## システム概要

### アーキテクチャ

```
タグ推定システム
├── Editorial Crawler     # 解説ページのクローリング
├── Embedding Filter      # 候補タグの絞り込み
├── Confidence System     # 複合信頼度スコアリング
└── Batch Processor      # バッチ処理・結果管理
```

### 主要特徴

- **高精度推定**: OpenAI o4-miniモデルによる自然言語理解
- **効率的処理**: Embeddingによる候補タグフィルタリング（119タグ→8タグ、93%削減）
- **信頼度評価**: 自己評価・検証・類似度の複合スコア
- **並列処理**: ThreadPoolExecutorによる高速化（最大5ワーカー）
- **範囲フィルタ**: コンテスト範囲・難易度による絞り込み
- **簡潔化出力**: 冗長な推論テキストを除去、トークン使用量削減
- **日本語対応**: 日本語解説文からのタグ推定

## 必要なパラメータ・設定

### 1. 環境変数

```bash
# OpenAI API Key (必須)
export OPENAI_API_KEY="your-api-key-here"

# または .env ファイルに設定
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2. 設定ファイル

#### `tag_inference_config.py`
```python
# モデル設定
model_name: str = "o4-mini"
max_tokens: int = 500
temperature: float = 0.3

# タグ設定
min_tags_per_problem: int = 3
max_tags_per_problem: int = 5

# バッチ処理設定
batch_size: int = 5
request_delay: float = 2.0
```

#### `tag_definitions.json`
119個のタグ定義（ID、名前、説明文）
```json
{
  "tags": [
    {
      "id": "DP",
      "name": "DP", 
      "description": "状態を記録して効率的に計算する動的計画法 Dynamic Programming メモ化"
    }
  ]
}
```

### 3. データファイル

- `editorial_mappings.json`: 問題-解説URL対応表（2,389問題）
- `problem_metadata.json`: 問題難易度データ
- `problems_with_tags.json`: 推定結果保存先
- `tag_embeddings.pkl`: タグembeddingキャッシュ

## コマンド使用方法

### バッチ処理コマンド

```bash
# 基本実行（ABC問題のみ）
python scripts/batch_tag_processor.py --contest-types abc

# 難易度200以下をスキップ
python scripts/batch_tag_processor.py --contest-types abc --min-difficulty 200

# テストモード（3問のみ）
python scripts/batch_tag_processor.py --test

# 統計表示のみ
python scripts/batch_tag_processor.py --stats-only

# 複数コンテストタイプ
python scripts/batch_tag_processor.py --contest-types abc,arc,agc

# 既存結果を再処理
python scripts/batch_tag_processor.py --force-reprocess

# 処理数制限
python scripts/batch_tag_processor.py --limit 100

# コンテスト範囲指定（ABC175-199）
python scripts/batch_tag_processor.py --contest-types abc --contest-range 175-199 --min-difficulty 200

# 並列処理（3ワーカー）
python scripts/batch_tag_processor.py --contest-types abc --parallel 3 --limit 50
```

### 個別問題テスト

```bash
# 特定問題のテスト
python scripts/enhanced_tag_inference.py

# 出力形式の簡潔化（一度実行済み）
python scripts/simplify_output_format.py
```

### パラメータ詳細

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `--contest-types` | 処理するコンテストタイプ | abc,arc,agc |
| `--contest-range` | コンテスト番号範囲 (例: 175-199) | なし |
| `--min-difficulty` | 最小難易度閾値 | なし |
| `--limit` | 処理問題数上限 | なし |
| `--parallel` | 並列ワーカー数 (1-5) | 1 |
| `--skip-existing` | 既存タグをスキップ | True |
| `--force-reprocess` | 強制再処理 | False |
| `--test` | テストモード | False |
| `--stats-only` | 統計のみ表示 | False |

## 処理フロー

### 1. Editorial Crawler
```
問題URL → 解説URL抽出 → 日本語解説文取得
- ABC175+, ARC104+, AGC048+ の2,389問題対応
- 日本語解説ページの自動取得
- リトライ機能付きクローリング
```

### 2. Embedding Filter
```
問題文 + 解説文 → Embedding化 → 候補タグフィルタ
- text-embedding-3-small使用
- 119タグから上位8タグを選出
- プロンプトサイズ93%削減効果
```

### 3. Confidence System
```
候補タグ → o4-mini推定 → 複合信頼度評価
├── Self Confidence: モデル自己評価 (重み40%)
├── Verifier Score: 別途検証評価 (重み30%)
└── Embedding Similarity: 類似度スコア (重み30%)
```

### 4. 結果判定
```
複合スコア → 上位3タグ選出 → 品質評価
- 平均信頼度 < 0.6: 低品質フラグ
- 最小信頼度 < 0.4: 低品質フラグ
- 標準偏差 > 0.3: 低品質フラグ
```

## 出力ファイル

システムが生成・更新する主要な出力ファイルの詳細説明：

### 1. メイン結果ファイル

#### `data/problems_with_tags.json`
タグ推定の最終結果を保存するメインファイル（簡潔化済み）
```json
{
  "problems": {
    "abc175_a": {
      "contest_id": "abc175",
      "problem_index": "A",
      "title": "Rainy Season",
      "problem_url": "https://atcoder.jp/contests/abc175/tasks/abc175_a",
      "editorial_url": "https://atcoder.jp/contests/abc175/editorial/51",
      "editorial_id": 51,
      "tags": ["条件判定", "組み合わせ", "累積和"],
      "confidence_scores": [0.89, 0.45, 0.28],
      "avg_confidence": 0.54,
      "low_confidence": true,
      "method": "enhanced_composite_confidence",
      "model": "o4-mini",
      "inferred_at": "2025-08-16T15:39:22.016993"
    }
  },
  "metadata": {
    "total_problems": 59,
    "last_updated": "2025-08-16T15:39:09.371033",
    "inference_model": "o4-mini",
    "inference_method": "enhanced_composite_confidence",
    "format_simplified_at": "2025-08-16T15:39:09.371033",
    "simplified_reason": "Removed verbose reasoning and detailed confidence to reduce token usage",
    "processing_stats": {
      "contest_types": ["abc"],
      "min_difficulty": 200,
      "successful_in_session": 49,
      "failed_in_session": 1
    }
  }
}
```

**ファイル構造**（簡潔化後）:
- `problems`: 各問題のタグ推定結果
  - `tags`: 推定されたタグ名（日本語）
  - `confidence_scores`: 各タグの信頼度
  - `avg_confidence`: 平均信頼度
  - `low_confidence`: 低信頼度フラグ
  - `method`: 推定手法
  - `model`: 使用モデル
  - `inferred_at`: 推定実行時刻
- `metadata`: 処理統計情報
  - `format_simplified_at`: 簡潔化実行時刻
  - `simplified_reason`: 簡潔化理由

### 2. ログファイル

#### `tag_inference.log`
バッチ処理の実行ログ
```
2025-08-16 13:35:38,253 - batch_tag_processor - INFO - ✓ abc175_a: ['条件判定(0.89)', '組み合わせ(0.45)', '累積和(0.28)'] avg=0.54 [LOW]
2025-08-16 13:35:38,264 - batch_tag_processor - INFO - Results saved to problems_with_tags.json
```

#### `editorial_crawler/data/crawler.log`  
解説クローラーの実行ログ
```
2025-08-13 15:45:12,345 - editorial_crawler - INFO - Processing contest: abc175
2025-08-13 15:45:13,123 - editorial_crawler - INFO - ✓ abc175_a: https://atcoder.jp/contests/abc175/editorial/51
```

### 3. キャッシュファイル

#### `vectors/tag_embeddings.pkl`
タグ定義のembeddingキャッシュ（119タグ）
- 用途: 候補タグフィルタリングの高速化
- 更新: tag_definitions.json変更時に自動再生成

#### `vectors/tag_vectors.pkl`
レガシーベクトルファイル（現在未使用）

#### `vectors/model_cache/`
Sentence Transformerモデルキャッシュ
- text-embedding-3-small相当のローカルモデル
- 初回実行時に自動ダウンロード

### 4. 中間データファイル

#### `editorial_crawler/data/editorial_mappings.json`
問題-解説URL対応表（2,389問題）
```json
{
  "editorial_mappings": {
    "abc175_a": {
      "contest_id": "abc175",
      "problem_index": "a", 
      "title": "Rainy Season",
      "problem_url": "https://atcoder.jp/contests/abc175/tasks/abc175_a",
      "editorial_url": "https://atcoder.jp/contests/abc175/editorial/51",
      "editorial_id": "51"
    }
  },
  "metadata": {
    "total_mappings": 2389,
    "crawled_at": "2025-08-13T18:35:28.581",
    "contest_ranges": {
      "abc": "175-407",
      "arc": "104-183", 
      "agc": "048-069"
    }
  }
}
```

### 5. 設定・定義ファイル（入力）

#### `config/tag_definitions.json`
119タグの定義・説明（システム入力）

#### `config/tag_list.txt` 
タグ名一覧（システム入力）

#### `data/problem_metadata.json`
問題難易度データ（システム入力）

### 6. 出力ファイルの管理

#### ファイルサイズ目安（簡潔化後）
- `problems_with_tags.json`: ~41KB（59問題、簡潔化により74.7%削減）
- `editorial_mappings.json`: ~400KB（2,389問題）
- `tag_embeddings.pkl`: ~50KB（119タグembedding）
- `tag_inference.log`: 処理ログ蓄積（定期削除推奨）

#### バックアップ推奨
- `problems_with_tags.json`: 推定結果の主データ
- `editorial_mappings.json`: クローラー実行結果

#### 自動更新タイミング
- **バッチ処理時**: problems_with_tags.json更新
- **クローラー実行時**: editorial_mappings.json更新  
- **設定変更時**: tag_embeddings.pkl再生成

## パフォーマンス

### 処理速度（並列処理対応）
- **平均処理時間**: 28.1秒/問題（シーケンシャル）
- **並列処理時**: 3ワーカーで約65%高速化
- **フェーズ別内訳**:
  - Editorial取得: 2-3秒
  - Embedding計算: 1-2秒
  - o4-mini推定: 20-25秒
  - 信頼度評価: 2-3秒

### API使用量目安
- **Embedding API**: ~0.001円/問題
- **o4-mini API**: ~0.1円/問題（簡潔化後トークン数削減）
- **合計コスト**: ~0.101円/問題

### 精度・効率指標
- **高信頼度タグ**: 平均信頼度 > 0.6
- **フィルタリング効果**: 候補タグ93%削減（119→8タグ）
- **ファイルサイズ削減**: 74.7%削減（簡潔化後）
- **Japanese語彙対応**: English-Japanese variant matching

## トラブルシューティング

### よくあるエラー

1. **OpenAI API Key未設定**
```bash
export OPENAI_API_KEY="your-key"
```

2. **依存関係不足**
```bash
pip install openai numpy scikit-learn sentence-transformers beautifulsoup4
```

3. **メモリ不足**
```bash
# バッチサイズを小さく
batch_size = 1  # in tag_inference_config.py
```

4. **レート制限エラー**
```bash
# 遅延時間を増加
request_delay = 5.0  # in tag_inference_config.py
```

### ログファイル

- `tag_inference.log`: 処理ログ
- `editorial_crawler/data/crawler.log`: クローラーログ

### デバッグコマンド

```bash
# 詳細ログ出力
python scripts/batch_tag_processor.py --test 2>&1 | tee debug.log

# 特定問題の詳細確認
python scripts/test_specific_problems.py
```

## システム要件

### 必須環境
- Python 3.8+
- OpenAI API アクセス
- インターネット接続（Editorial取得用）

### 推奨スペック
- RAM: 4GB以上
- ディスク容量: 1GB以上
- CPU: マルチコア推奨

### 依存ライブラリ
```
openai>=1.99.9
numpy>=2.3.2
scikit-learn>=1.7.1
sentence-transformers>=5.1.0
beautifulsoup4>=4.13.4
requests>=2.32.4
```