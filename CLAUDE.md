あなたは熟練のMLエンジニア兼インフラエンジニアです。
以下の仕様で MiniLM多言語 Sentence-Transformers モデルを ONNX へ変換→INT8量子化→Torch不要ランタイムでFastAPI推論できる一式を作ってください。
最終成果物はローカルで docker run 可能、さらに Cloud Run にそのままデプロイ可能な状態にしてください。

目的

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 を ONNX に変換し、INT8 量子化する

ランタイムでは PyTorch非依存（onnxruntime + tokenizers のみ）で FastAPI により /embed API を提供

Sentence-Transformers と同等の mean pooling + L2 normalize を再現（互換埋め込み）

コンテナサイズをできるだけ小さく（目標 100–200MB台）

ディレクトリ構成（作成してほしい）
app/
  export_onnx.py          # 変換（ローカル/Colabで実行; Torchはここだけで使用可）
  quantize.py             # 量子化（任意だがデフォルトで実行可能に）
  main.py                 # 推論API（Torch不要; onnxruntime + tokenizers）
  requirements.txt        # ランタイム依存（軽量）
  Dockerfile              # Cloud Run 向け（Torchなし）
  README.md               # 手順書（検証手順・Cloud Run手順含む）
  tokenizer/              # export時に保存される
  model_fp32.onnx         # export結果（生成）
  model_int8.onnx         # 量子化結果（生成; こちらを既定で使う）

要求仕様（重要）

ONNX変換 (export_onnx.py)

モデル: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

transformers + torch を使用して ローカル実行（※ランタイムでは使わない）

mean pooling（attention_mask でマスクしてトークン平均）→ L2 normalize までを 1つのONNX出力に含める（＝ONNXの出力がそのまま文章ベクトル）

dynamic axes（batch, seq）を設定

tokenizer.save_pretrained("tokenizer/") で tokenizer.json を含む形で保存

成功時に model_fp32.onnx を生成

INT8 量子化 (quantize.py)

ONNX Runtime の Dynamic Quantization を使用（onnxruntime.quantization.quantize_dynamic）

入力: model_fp32.onnx → 出力: model_int8.onnx

スクリプト内で進捗/完了ログを出す

量子化後に onnx.checker で検証し簡易レポート表示（ノード数・サイズ差分）

推論API (main.py)

依存は onnxruntime tokenizers fastapi uvicorn numpy のみ（transformers や torch は使わない）

tokenizer/tokenizer.json を tokenizers でロード

[CLS]/[SEP] を自動付与（BERT系テンプレート）

max_length=256 でトリム＆ゼロパディング

推論は onnxruntime.InferenceSession で model_int8.onnx をロード（環境変数 MODEL_PATH で切替可能）

出力は L2 正規化済みベクトル（再度 L2 正規化をかけてもOK）

エンドポイント：

GET /healthz → {ok: true}

POST /embed（body: { "texts": ["...", "..."] }）→ { "vectors": [[...], [...]] }

GET /embed?q=テキスト → 単発ベクトル返却

例外処理・バッチ処理・簡単なログを含める

Dockerfile

ベース: python:3.11-slim

pip install -r requirements.txt（--no-cache-dir）

TorchやTransformersを入れない

uvicorn で PORT 環境変数を使い 0.0.0.0 で起動

最終イメージが小さくなるよう apt キャッシュ削除等の最適化

可能なら multi-stage は不要（単純でOK）

requirements.txt（ランタイム）

fastapi、uvicorn[standard]、onnxruntime、tokenizers、numpy

バージョンは最新安定（例：fastapi>=0.114, onnxruntime>=1.17, tokenizers>=0.19）

README.md

目的と全体像

準備（ローカル）：pip install torch transformers onnx onnxruntime onnxruntime-tools など/ python export_onnx.py / python quantize.py

ローカル実行：docker build → docker run -e PORT=8080 -p 8080:8080 ...

APIテスト：curl "http://localhost:8080/embed?q=ワーシャルフロイド"

Cloud Run デプロイ（コンソール操作手順）

Artifact Registry リポジトリ作成

コンテナ登録

Cloud Run → コンテナURL指定→ リージョン→ 未認証許可（任意）→ メモリ 1〜2GiB → デプロイ

精度メモ：INT8 量子化に伴う誤差の注意、必要なら model_fp32.onnx で起動する方法も記載

トラブルシュート（tokenizer.json が読めない、CLS/SEP ID、動的軸、PORT未設定 等）

受け入れ条件（動作確認）

export_onnx.py 実行で tokenizer/ と model_fp32.onnx が生成されること

quantize.py 実行で model_int8.onnx が生成されること

docker run で起動し、GET /healthz が {"ok": true} を返すこと

GET /embed?q=ベルマンフォード が H次元ベクトルを返すこと（H は 384 であることをログに表示）

量子化版と非量子化版の コサイン類似度の差分が平均で小さい（例：サンプル10文で平均差分 < 0.02 など、簡易ベンチのスクリプトも入れて良い）

品質

型ヒント、簡潔なDocstring、ログ（起動時にモデルパス・次元数など）を出す

例外時は 400/500 を適切に返す

依存は最小限、不要ファイルは .dockerignore で除外（あれば）

以上を満たすコード・ファイル一式を生成してください。
