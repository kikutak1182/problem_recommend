# AtCoder Editorial Crawler

問題URLと解説URLを相互に結びつけるデータベースを作成するクローラーシステムです。

## 概要

このクローラーは以下の仕様に従ってAtCoderの公式解説URLを抽出します：

- **対象コンテスト**: ABC175以降、ARC104以降、AGC048以降
- **対象問題**: 各コンテストの全問題（A-G, Ex含む）
- **言語**: 日本語解説のみ (`?editorialLang=ja`)
- **抽出方法**: h3/h4見出しから公式解説リンクを特定

## ディレクトリ構造

```
editorial_crawler/
├── crawler_config.py      # クローラー設定
├── contest_filter.py      # コンテスト フィルタリング
├── editorial_extractor.py # 解説URL抽出ロジック
├── database_builder.py    # データベース構築
├── validator.py          # データ検証・修復
├── run_crawler.py        # メイン実行スクリプト
├── README.md             # このファイル
└── data/                 # 出力データ
    ├── editorial_mappings.json  # 生成されたデータベース
    └── crawler.log              # 実行ログ
```

## 使用方法

### 1. テスト実行（推奨）

```bash
python run_crawler.py --test
```

5つのコンテストのみを処理してテスト実行します。

### 2. 統計情報の確認

```bash
python run_crawler.py --stats-only
```

対象コンテストの統計情報を表示します。

### 3. 特定のコンテストタイプのみ処理

```bash
python run_crawler.py --contest-types abc --limit 10
```

ABC コンテストのみ、最大10コンテストを処理します。

### 4. フル実行

```bash
python run_crawler.py
```

全対象コンテストを処理します（時間がかかります）。

### 5. データベース検証のみ

```bash
python run_crawler.py --validate-only
```

既存のデータベースを検証します。

### 6. 問題の修復

```bash
python run_crawler.py --fix-issues
```

データベースの一般的な問題を修復します。

## 出力データ形式

```json
    {
      "problem_id": "abc407_g",
      "title": "",
      "contest_id": "abc407",
      "difficulty": 2190,
      "problem_url": "https://atcoder.jp/contests/abc407/tasks/abc407_g",
      "editorial_url": "https://atcoder.jp/contests/abc407/editorial/13077"
    },
```

## 設定

`crawler_config.py`で以下を設定できます：

- **対象コンテスト閾値**: `target_contests`
- **リクエスト間隔**: `request_delay` (デフォルト: 1.0秒)
- **リトライ設定**: `max_retries`, `retry_delay`
- **User-Agent**: `user_agent`

## クローラーの礼儀

- 適切なUser-Agentを設定
- リクエスト間に1秒の待機時間
- 失敗時のリトライ機構
- 詳細なログ記録

## エラーハンドリング

- ネットワークエラー時の自動リトライ
- 部分的な失敗時の継続処理
- 進行状況の定期保存
- 詳細なエラーログ

## 注意事項

- **レート制限**: AtCoderサーバーに負荷をかけないよう適切な間隔でアクセス
- **データサイズ**: 全コンテスト処理時は数千の問題が対象となります
- **実行時間**: フル実行は数時間かかる場合があります
- **エラー継続**: 一部のコンテストで失敗しても処理を継続します

## トラブルシューティング

### よくある問題

1. **ネットワークエラー**
   ```bash
   # リトライ設定を調整
   # crawler_config.py の max_retries を増加
   ```

2. **データベース破損**
   ```bash
   python run_crawler.py --fix-issues
   ```

3. **進行状況の確認**
   ```bash
   tail -f data/crawler.log
   ```

### ログの確認

実行ログは `data/crawler.log` に保存されます：

```bash
# リアルタイムでログを監視
tail -f tag/editorial_crawler/data/crawler.log

# エラーのみを表示
grep ERROR tag/editorial_crawler/data/crawler.log
```

## 開発者向け

### 個別コンポーネントのテスト

```bash
# コンテストフィルター
python contest_filter.py

# データベースビルダー
python database_builder.py

# バリデーター
python validator.py
```

### 新しい抽出ルールの追加

`editorial_extractor.py`の以下のメソッドを修正：

- `_extract_problem_index_from_heading()`: 見出しパターン
- `_is_valid_editorial_link()`: 解説リンクパターン