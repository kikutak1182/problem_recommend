# Repository Guidelines

## プロジェクト構成
- `app/`: FastAPI 本体。`main.py` がエントリ、`templates/` は Jinja2 テンプレート、`data/` は公開可能な生成済み JSON を格納。
- `tag/`: タグ推定・データ取得のユーティリティ。`config/` 設定、`scripts/` バッチ/埋め込み、`editorial_crawler/` クローラ、`vectors/` 埋め込みベクトル、`data/` 中間成果物。
- ルート: `requirements.txt` 依存、`render.yaml` デプロイ設定、`README.md` プロジェクト概要。
- テスト: 既定のディレクトリは未設置。追加時は `tests/` を作成し、モジュール別に配置。

## ビルド・実行・開発コマンド
- セットアップ: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- ローカル起動: `uvicorn app.main:app --reload`（デフォルトは `http://127.0.0.1:8000`）
- 本番起動: Render では `render.yaml` の `startCommand`（`uvicorn app.main:app --host 0.0.0.0 --port $PORT`）を使用。
- ユーティリティ実行例: `python tag/scripts/batch/create_batch_requests.py`（バッチ作成）／`python tag/scripts/embedding/batch_problem_embeddings.py`（埋め込み生成）。

## コーディング規約・命名
- Python 3 系、PEP 8、インデント 4 スペース、`snake_case`。モジュール・パッケージ名も小文字スネーク。
- 型ヒントを新規・変更箇所に付与。文字列は f-string を優先。
- フォーマット/リンタ（推奨）: `black`, `isort`, `ruff`。
  - 例: `pip install black isort ruff && black . && isort . && ruff .`

## テスト方針
- フレームワーク: `pytest` を推奨。`tests/test_<module>.py` 命名。
- 実行: `pytest -q`
- 推奨カバレッジ: 推薦ロジックとタグ推定の主要分岐を網羅。FastAPI は `TestClient` を用いたエンドポイントテスト、データ処理はモック/小規模サンプルで検証。

## コミット / PR ガイドライン
- コミット: 簡潔な現在形。必要に応じて接頭辞（`feat:`, `fix:`, `docs:` など）を使用。
  - 例: `feat: タグに基づく推薦を統合` / `fix: render.yaml のテンプレートパス`
- 1 コミット 1 意図・関連差分に限定。説明は “なぜ” を短く含める。
- PR: 目的、変更点、動作確認手順、スクリーンショット（UI 変更時）、関連 Issue を記載。大規模変更は簡潔な設計メモを併記。

## セキュリティ / 設定
- 秘密情報は `.env`（Git 管理外）に保存し、Render の環境変数を使用。API キーはコミットしない。
- 大容量ファイルや学習済みモデルは追跡しない。必要ならダウンロード手順/スクリプト化。
- `app/data/` は公開可能な成果物のみを配置。個人情報・秘匿データは含めない。
