# AtCoderタグ推定方法について

## 概要
このシステムは、AtCoder問題に対して適切なタグを自動推定するシステムです。機械学習（OpenAI API）と複数の信頼度スコア手法を組み合わせて、高精度なタグ付けを実現しています。

## システム構成

### 1. タグ定義体系
- 詳細なタグ定義（`config/tag_definitions.json`）
- 100以上の競技プログラミングタグ
- 各タグに対して：
  - ID、名前、エイリアス（別名）
  - 詳細な説明（解法手法、計算量、典型問題等）
    {
      "id": "DP",
      "name": "動的計画法",
      "aliases": ["DP", "動的計画", "Dynamic Programming", "メモ化", "dp"],
      "description": "大きな問題を部分問題に分割し、結果をテーブルに記録・再利用する最適化手法。「i番目までで状態jになる最適値/場合の数」を漸化式で求めるのが典型で、ナップサック問題などが代表例。計算量は(状態数)×(遷移)で決まり、bitDPや区間DPといった頻出パターンも存在する。"
    },

### 2. 推定手法の階層構造

#### 第1段階: 候補絞り込み
- **エンベディング類似度**: 問題文と解説をベクトル化し、タグベクトルとのコサイン類似度で候補を絞り込み
    - `vectors/tag_embeddings.pkl`: タグエンベディング（1.3MB）
    - `vectors/problem_embeddings.pkl`: 問題エンベディング（20.9MB）
- **キーワードマッチング**: ルールベースでタグのエイリアスや関連キーワードを検出
    - 完全一致するとスコアを＋１して、候補に選ばれやすくする

#### 第2段階: 信頼度スコア計算
複合信頼度システム（Evidence + Inference方式）:

1. **Inference Score** (0.0-1.0の範囲)
   - Self-confidence (60%): モデルの自己申告信頼度
   - Embedding similarity (40%): エンベディング類似度スコア

2. **Evidence Score** (0 or 1)
   - ルールベースマッチングで確実にヒットした場合は1、そうでなければ0

3. **Final Score**
   - `Final Score = Evidence Score + Inference Score`
   - 範囲: 0.0-2.0
   - スコア>=1ならルールベースマッチング（完全一致）したということ

### 3. バッチ処理システム
- **OpenAI Batch API**: 大量の問題を効率的に処理
- **並列処理**: 複数の信頼度成分を並行計算
- **結果集約**: バッチ結果の統合とフォーマット変換

## 実装の詳細

### コアコンポーネント
- `confidence_system.py`: メイン信頼度計算エンジン
- `embedding_tag_filter.py`: エンベディングベースの候補絞り込み
- `keyword_matcher.py`: ルールベースキーワードマッチング
- `batch_tag_inference.py`: バッチ処理システム

### 処理フロー
1. 問題文・解説の前処理
2. エンベディング候補フィルタリング（上位20-30候補）
3. LLMによる最終3タグ選択 + 信頼度スコア算出
4. 複合信頼度による最終ランキング

## 利用手法・技術(改善の余地あり)
- **Large Language Model**: Claude/GPT-4による自然言語理解
- **Embedding**: テキストベクトル化による類似度計算
- **JSON Schema**: 構造化出力の制約
- **Batch API**: 大規模処理の効率化

## 特徴
- **多段階絞り込み**: 計算コストと精度のバランス
- **複合信頼度**: 複数指標の組み合わせによる頑健性
- **拡張性**: 新しいタグ・手法の追加が容易
- **実用性**: バッチ処理による大量データ対応

## 精度向上の工夫
- タグ定義の詳細化（典型問題、計算量等を含む）
- エイリアスの拡充（複数言語・表記ゆれ対応）
- 信頼度重み付けの調整
- 問題特性に応じた候補絞り込み

## ファイル構成

### 設定ファイル
- `config/tag_definitions.json`: タグ定義（100以上のタグ、詳細説明、エイリアス）
- `config/tag_definitions_enhanced.json`: 拡張タグ定義

### 主要スクリプト

#### コア信頼度システム
- `scripts/confidence_system.py`: 基本信頼度計算システム
- `scripts/confidence_system_enhanced.py`: 拡張信頼度システム
- `scripts/confidence_precision_analyzer.py`: 精度分析ツール

#### バッチ処理システム  
- `scripts/batch_tag_inference.py`: メインバッチ推定システム
- `scripts/batch_api_processor.py`: バッチAPI処理
- `scripts/create_batch_requests.py`: バッチリクエスト生成
- `scripts/submit_batch_job.py`: バッチジョブ送信
- `scripts/check_batch_status.py`: バッチ状態監視

#### エンベディング処理
- `scripts/embedding_tag_filter.py`: エンベディングベース候補絞り込み
- `scripts/batch_embedding_generator.py`: エンベディング生成
- `scripts/batch_problem_embeddings.py`: 問題エンベディング処理
- `scripts/cached_embedding_filter.py`: キャッシュ済エンベディング利用

#### ユーティリティ
- `scripts/keyword_matcher.py`: キーワードベースマッチング
- `scripts/convert_batch_to_standard_format.py`: フォーマット変換
- `scripts/fix_batch_results.py`: 結果修復ツール
- `scripts/unified_tag_tester.py`: 統合テストツール



### データファイル

#### 基本データ
- `data/problems_with_tags.json`: タグ付き問題データ
- `data/abc250_400_problems.json`: ABC250-400問題データ

#### バッチ処理結果
- `data/abc180_199_batch_requests.jsonl`: バッチリクエスト（830KB）
- `data/abc180_199_batch_results_*.json`: バッチ結果
- `data/batch_tag_results_*.jsonl`: タグ推定結果

#### テスト結果
- `data/abc175_179_test_results_*.json`: テスト結果
- `data/abc180_199_standard_results_*.json`: 標準フォーマット結果

#### エンベディングデータ
- `data/embedding_batch_*.jsonl`: エンベディングバッチデータ（10MB+）
- `data/problem_combined_text_cache.json`: テキストキャッシュ

## エンベディング生成のバッチ処理フロー

### 1. 問題データ準備
```bash
# 問題文と解説を統合したテキストキャッシュを作成
python scripts/editorial_text_cache_builder.py
# → data/problem_combined_text_cache.json
```

### 2. バッチリクエスト生成
```bash
# エンベディング生成用のバッチリクエストファイルを作成
python scripts/batch_embedding_generator.py
# → data/embedding_batch_YYYYMMDD_HHMMSS.jsonl (10MB+)
```

### 3. バッチジョブ投入
```bash
# OpenAI Batch APIにエンベディングジョブを送信
python scripts/submit_batch_job.py --file data/embedding_batch_20250826_130016.jsonl
# → バッチID取得
```

### 4. バッチ状態監視
```bash
# バッチ処理の進行状況を確認
python scripts/check_batch_status.py --batch_id batch_abc123
```

### 5. エンベディング結果処理
```bash
# 完了したバッチ結果をダウンロード・処理してpklファイルに保存
python scripts/batch_problem_embeddings.py --batch_id batch_abc123
# → vectors/problem_embeddings.pkl (20.9MB)
```

### バッチ処理の利点
- **コスト削減**: 通常API料金の50%オフ
- **大量処理**: 数百問題を一括処理
- **並列化**: 複数バッチを同時実行可能
- **信頼性**: 自動リトライ機能

## タグ推定のバッチ処理フロー

### 1. ルールベースでタグ

### 1. タグ推定リクエスト生成
```bash
# 問題範囲を指定してバッチリクエストを生成
python scripts/create_batch_requests.py --start_contest abc180 --end_contest abc199
# → data/abc180_199_batch_requests.jsonl (830KB)
```

### 2. バッチジョブ送信
```bash
# OpenAI Batch APIにタグ推定ジョブを送信
python scripts/submit_batch_job.py --file data/abc180_199_batch_requests.jsonl
# → バッチID取得: batch_68ad91ffaae88190952ba31bb1ddd32b
```

### 3. バッチ状態監視と情報保存
```bash
# バッチ処理の進行状況を確認し、情報をJSONファイルに保存
python scripts/check_batch_status.py --batch_id batch_68ad91ffaae88190952ba31bb1ddd32b
# → data/batch_tag_inference_batch_68ad91ffaae88190952ba31bb1ddd32b_info.json
```

### 4. バッチ結果処理
```bash
# 完了したバッチ結果をダウンロード・処理
python scripts/batch_tag_inference.py --batch_id batch_68ad91ffaae88190952ba31bb1ddd32b
# → data/batch_tag_results_YYYYMMDD_HHMMSS.jsonl
```

### 5. 結果フォーマット変換
```bash
# バッチ結果を標準フォーマットに変換
python scripts/convert_batch_to_standard_format.py --input data/batch_tag_results_20250826_205005.jsonl
# → data/abc180_199_standard_results_YYYYMMDD_HHMMSS.json
```

### 6. 結果修復（必要に応じて）
```bash
# JSON解析エラーがある場合の修復処理
python scripts/fix_batch_results.py --input data/abc180_199_batch_results_20250826_205005.json
# → data/abc180_199_batch_results_fixed_YYYYMMDD_HHMMSS.json
```

### タグ推定バッチ処理の特徴
- **エンベディング候補絞り込み**: 事前に20-30候補に絞ってからLLMに送信
- **構造化出力**: JSON Schemaで出力形式を制約
- **信頼度スコア**: Self-confidence + Embedding similarityの複合スコア
- **Evidence検出**: キーワードマッチングによるルールベース判定

### 出力ファイル形式

#### 標準タグ推定結果フォーマット
```json
{
  "abc175_c": {
    "title": "Walking Takahashi",
    "problem_url": "https://atcoder.jp/contests/abc175/tasks/abc175_c",
    "editorial_url":"https://atcoder.jp/contests/abc175/editorial/52",
    "tags": [
      "動的計画法",
      "全探索",
      "ユークリッドの互除法"
    ],
    "tag_ids": [
      "DP",
      "BRT",
      "EUC"
    ],
    "confidence_scores": [
      0.8500000000000001,
      0.55,
      0.35000000000000003
    ],
    "detailed_scores": {
        "DP": {
          "self_confidence": 0.9,
          "embedding_similarity": 0.7391455817708326,
          "rule_based_score": 0.0
        },
        "BRT": {
          "self_confidence": 0.4,
          "embedding_similarity": 0.7155155166916256,
          "rule_based_score": 0.0
        },
        "EUC": {
          "self_confidence": 0.1,
          "embedding_similarity": 0.7046220533393935,
          "rule_based_score": 0.0
        }
    }
  }
}
```

#### バッチ結果フォーマット
```json
{
  "custom_id": "abc180_a_tag_inference",
  "response": {
    "body": {
      "choices": [{
        "message": {
          "content": "{\"selected_tags\": [...], \"confidence_scores\": [...]}"
        }
      }]
    }
  }
}
```