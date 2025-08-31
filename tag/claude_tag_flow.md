# AtCoderタグ推定フロー

## 設定ファイル
**場所**: `config/tag_inference_config.py`

**設定項目**:
- `default_start_contest: int = 175` - 開始コンテスト番号
- `default_end_contest: int = 407` - 終了コンテスト番号
- `difficulty_threshold: int = 400` - 対象問題の難易度しきい値（この値以上の問題が自動選択される）
- `use_concurrent: bool = False` - 並行処理使用するか
- `concurrent_workers: int = 3` - 並行処理ワーカー数

## ディレクトリ構造
```
scripts/
├── cache/          # テキストキャッシュ関連
├── embedding/      # エンベディング生成関連
├── filtering/      # タグ候補絞り込み関連
└── batch/          # バッチジョブ・結果変換関連
```

## エンベディング生成のバッチ処理フロー

### 1. 問題データ準備（テキストキャッシュ構築）
```bash
# 問題文と解説を統合したテキストキャッシュを作成
python scripts/cache/editorial_text_cache_builder.py
# → data/problem_combined_text_cache.json

# configのデフォルト設定で実行
python scripts/cache/editorial_text_cache_builder.py

# コマンドライン引数で範囲を指定
python scripts/cache/editorial_text_cache_builder.py --start 250 --end 300

# 並行処理を使用
python scripts/cache/editorial_text_cache_builder.py --concurrent --workers 5
```

### 2. エンベディング用バッチリクエスト生成
```bash
# エンベディング生成用のバッチリクエストファイルを作成
python scripts/embedding/batch_embedding_generator.py
# → data/embedding_batch_YYYYMMDD_HHMMSS.jsonl (10MB+)

# キャッシュから生成する場合
python scripts/embedding/batch_embedding_from_cache.py
```

### 3. バッチジョブ投入
```bash
# OpenAI Batch APIにエンベディングジョブを送信
python scripts/batch/submit_batch_job.py --file data/embedding_batch_20250826_130016.jsonl
# → バッチID取得
```

### 4. バッチ状態監視
```bash
# バッチ処理の進行状況を確認
python scripts/batch/check_batch_status.py --batch_id batch_abc123
```

### 5. エンベディング結果処理
```bash
# 完了したバッチ結果をダウンロード・処理してpklファイルに保存
python scripts/embedding/batch_problem_embeddings.py --batch_id batch_abc123
# → vectors/problem_embeddings.pkl (20.9MB)
```

### バッチ処理の利点
- **コスト削減**: 通常API料金の50%オフ
- **大量処理**: 数百問題を一括処理
- **並列化**: 複数バッチを同時実行可能
- **信頼性**: 自動リトライ機能

## タグ推定のバッチ処理フロー

タグ推定は以下の3段階のプロセスで行われます：

### **ステップ1: 候補タグ絞り込み（8個に絞り込み）**
ルールベース + エンベディング類似度スコアを使って、全タグから上位8個の候補タグに絞り込みます。

**処理内容:**
- **ルールベースマッチング**: 312個のキーワードでマッチングし確実にヒットするタグを特定
- **エンベディング類似度**: 問題文・解説とタグ定義のコサイン類似度を計算
- **ハイブリッド候補絞り込み**: ルールマッチしたタグに+1.0ボーナス付与し、エンベディング類似度と合わせて上位12個を選択
- **表示区別**: `[RULE]`（ルールマッチ）と`[EMBED]`（エンベディングのみ）で候補を区別表示

**使用スクリプト:**
- `scripts/filtering/cached_embedding_filter.py` - エンベディング類似度計算
- `scripts/filtering/keyword_matcher.py` - ルールベースマッチング

**実行方法:**
```bash
# この処理はcreate_batch_requests.py内で自動実行されます
python scripts/batch/create_batch_requests.py --start 250 --end 300

# 内部で以下が実行される:
# 1. scripts/filtering/cached_embedding_filter.py の filter_candidate_tags_by_id()
# 2. scripts/filtering/keyword_matcher.py の find_keyword_matches()
# 3. 上位8個の候補タグを選択
```

**入力ファイル:**
- `data/problem_combined_text_cache.json` - 問題文・解説テキストキャッシュ
- `vectors/problem_embeddings.pkl` - 問題エンベディング
- `vectors/tag_embeddings.pkl` - タグエンベディング
- `config/tag_definitions.json` - タグ定義

**出力:**
- 各問題につき上位12個の候補タグリスト（similarity score付き）
- ルールベーススコア保存: `data/abc{start}_{end}_rule_based_scores_{timestamp}.json`

**候補絞り込み結果例:**
```
abc175_c: ビット演算 [RULE] (similarity: 1.387)
abc175_d: グラフ [RULE] (similarity: 1.432), 4 rule-boosted candidates  
abc175_e: 動的計画法 [RULE] (similarity: 1.603), 2 rule-boosted candidates
```

### **ステップ2: Confidence推定（OpenAI Batch API）**
絞り込まれた12個の候補タグそれぞれについて、OpenAI Batch APIを使ってconfidence（自信度）を一括推定します。

**処理内容:**
- 12個の候補タグを含むプロンプトをLLMに送信
- 各タグのconfidence (0.0-1.0) を推定
- Batch APIで大量の問題を効率的に処理（50%コスト削減）

**実行コマンド:**
```bash
# 1. バッチリクエスト生成（候補絞り込み + プロンプト作成）
python scripts/batch/create_batch_requests.py --start 250 --end 300
# → data/abc250_300_batch_requests_20250830_123456.jsonl
# → data/abc250_300_rule_based_scores_20250830_123456.json（ルールベーススコア保存）

# 2. バッチジョブ送信
python scripts/batch/submit_batch_job.py --file data/abc250_300_batch_requests_20250830_123456.jsonl --api-key YOUR_API_KEY
# → バッチID取得: batch_68ad91ffaae88190952ba31bb1ddd32b
# → data/batch_tag_inference_batch_68ad91ffaae88190952ba31bb1ddd32b_info.json（バッチ情報保存）

# 3. バッチ状態監視（完了まで定期的に実行）
python scripts/batch/check_batch_status.py --batch-id batch_68ad91ffaae88190952ba31bb1ddd32b

# 4. バッチ結果取得（完了後）
python scripts/batch/batch_tag_inference.py --batch-id batch_68ad91ffaae88190952ba31bb1ddd32b
# → data/batch_tag_results_YYYYMMDD_HHMMSS.jsonl

# 5. バッチ結果修正（必要に応じて）
python scripts/batch/fix_batch_results.py --input data/batch_tag_results_YYYYMMDD_HHMMSS.jsonl
# → data/batch_tag_results_YYYYMMDD_HHMMSS_fixed.jsonl
```

**ABC175-200での実際の実行例:**
```bash
# 実行例（2025年8月30日）
python scripts/batch/create_batch_requests.py --start 175 --end 200
# → data/abc175_200_batch_requests_20250830_154521.jsonl (103 requests)
# → data/abc175_200_rule_based_scores_20250830_154521.json

python scripts/batch/submit_batch_job.py --file data/abc175_200_batch_requests_20250830_154521.jsonl --api-key sk-proj-...
# → Batch ID: batch_68b29f4df6848190af6691ae4d13c553
# → Status: validating
# → File ID: file-6iBY66UtpdoPmxhKggmsMj

python scripts/batch/check_batch_status.py --batch-id batch_68b29f4df6848190af6691ae4d13c553
```

**入力ファイル:**
- `data/abc250_300_batch_requests_20250830_123456.jsonl` - バッチAPIリクエストファイル

**出力ファイル:**
- `data/batch_tag_results_YYYYMMDD_HHMMSS.jsonl` - バッチAPI結果（各問題のLLM推定結果）
- `data/batch_tag_inference_{batch_id}_info.json` - バッチジョブ情報（ID、ファイルパス、ステータス等）

**バッチリクエストの内容例:**
```json
{
  "custom_id": "abc250_c",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "問題文・解説: ...\n\n候補タグ: [DP, BFS, Graph, ...]\n\n各タグのconfidenceを推定してください。"
      }
    ],
    "response_format": {"type": "json_object"}
  }
}
```

### **ステップ3: スコア正規化と最終出力**
3つのスコア成分を正規化して統合し、最終的なタグ推定結果を出力します。

**スコア成分と正規化:**
- **ルールベーススコア**: 0 or 1 (そのまま)
- **self_confidence**: 0~0.5 (LLMの出力を0.5倍)  
- **embedding_similarity**: 0~0.5 (類似度を0.5倍)

**最終スコア計算:**
```
Final Score = rule_based_score(= 0 or 1) + (self_confidence * 0.5) + (embedding_similarity * 0.5)
範囲: 0.0 ~ 2.0
```

**実行コマンド:**
```bash
# スコア統合と標準フォーマット出力
python scripts/batch/convert_batch_to_standard_format.py --input data/batch_tag_results_20250830_143256.jsonl
# → data/abc250_300_final_results_20250830_143500.json

# ルールベーススコアを含めた変換
python scripts/batch/convert_batch_to_standard_format.py --input data/batch_tag_results_20250830_143256.jsonl --rule-scores data/abc250_300_rule_based_scores_20250830_143000.json
```

**入力ファイル:**
- `data/batch_tag_results_20250830_143256.jsonl` - バッチAPI結果
- `data/abc250_300_rule_based_scores_20250830_143000.json` - ルールベーススコア（ステップ1で生成）

**使用スクリプト:**
- `scripts/filtering/confidence_system.py` - スコア正規化・統合処理
- `scripts/batch/convert_batch_to_standard_format.py` - 最終フォーマット変換（ルールベーススコア統合、複合スコア計算、URL修正を含む）

**最終出力ファイル:**
- `data/abc250_300_final_results_20250830_143500.json` - 標準タグ推定結果

**最終出力フォーマット例:**
```json
{
  "abc250_c": {
    "title": "Problem Title",
    "problem_url": "https://atcoder.jp/contests/abc250/tasks/abc250_c",
    "editorial_url": "https://atcoder.jp/contests/abc250/editorial",
    "tags": ["動的計画法", "全探索", "グラフ"],
    "tag_ids": ["DP", "BRT", "GRA"],
    "confidence_scores": [1.85, 1.2, 0.95],
    "detailed_scores": {
      "DP": {
        "rule_based_score": 1.0,
        "self_confidence": 0.45,
        "embedding_similarity": 0.4,
        "final_score": 1.85
      },
      "BRT": {
        "rule_based_score": 0.0,
        "self_confidence": 0.4,
        "embedding_similarity": 0.8,
        "final_score": 1.2
      }
    }
  }
}
```

## 使用するスクリプト一覧

### テキストキャッシュ関連（scripts/cache/）
- `editorial_text_cache_builder.py` - 問題文・解説テキストキャッシュ構築
- `editorial_text_extractor.py` - 解説テキスト抽出

### エンベディング生成関連（scripts/embedding/）
- `batch_embedding_generator.py` - エンベディング用バッチリクエスト生成
- `batch_embedding_from_cache.py` - キャッシュからエンベディング生成
- `batch_problem_embeddings.py` - 問題エンベディング処理

### タグ候補絞り込み関連（scripts/filtering/）
- `cached_embedding_filter.py` - エンベディングベース候補絞り込み
- `keyword_matcher.py` - キーワードマッチング
- `confidence_system.py` - 信頼度システム

### バッチジョブ・結果変換関連（scripts/batch/）
- `create_batch_requests.py` - タグ推定バッチリクエスト生成
- `submit_batch_job.py` - バッチジョブ送信
- `check_batch_status.py` - バッチ状態監視・結果ダウンロード
- `convert_batch_to_standard_format.py` - バッチ結果の標準フォーマット変換（複合スコア計算・URL修正含む）

## タグ推定バッチ処理の特徴
- **難易度フィルタ**: 難易度400未満の問題は自動除外
- **ハイブリッド候補絞り込み**: 312キーワードのルールベース + エンベディング類似度で12候補に絞り込み
- **構造化出力**: JSON Schemaで出力形式を制約
- **複合信頼度スコア**: Rule-based(0/1) + Self-confidence(0-0.5) + Embedding similarity(0-0.5) = 最大2.0
- **Evidence検出**: キーワードマッチングによる確実な判定とスコアブースト（+1.0）
- **コスト効率**: Batch API使用で50%コスト削減

## 出力ファイル形式

### 標準タグ推定結果フォーマット
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

### バッチ結果フォーマット
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