# ONNX Sentence Embedding API

高性能・軽量な日本語対応文章埋め込みAPI。PyTorchに依存せず、ONNX Runtime + INT8量子化で大幅な軽量化を実現。

## 🎯 目的と全体像

- **モデル**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **変換**: Sentence-Transformers → ONNX → INT8量子化
- **ランタイム**: PyTorch非依存（`onnxruntime` + `tokenizers`のみ）
- **サイズ**: ~100-200MB（vs 1.7GB PyTorch版）
- **互換性**: Sentence-Transformersと同等の埋め込みベクトル出力

## 🛠️ セットアップ（ローカル）

### 1. 依存関係のインストール（変換用）

```bash
# PyTorchとTransformersは変換時のみ必要
pip install torch transformers sentence-transformers onnx onnxruntime onnxruntime-tools
```

### 2. ONNX変換の実行

```bash
# Step 1: Sentence-TransformersモデルをONNXに変換
python export_onnx.py

# Step 2: INT8量子化（サイズ削減）
python quantize.py
```

**生成されるファイル**:
- `model_fp32.onnx` - FP32 ONNXモデル
- `model_int8.onnx` - INT8量子化モデル（本番用）
- `tokenizer/` - Tokenizerファイル

## 🐳 Docker実行

### ローカルでのテスト

```bash
# Dockerイメージのビルド
docker build -t onnx-embedding-api .

# コンテナの起動
docker run -d \
    --name embedding-api \
    -e PORT=8080 \
    -p 8080:8080 \
    onnx-embedding-api

# ヘルスチェック
curl http://localhost:8080/healthz
```

### APIテスト

```bash
# 単発埋め込み
curl "http://localhost:8080/embed?q=ワーシャルフロイド"

# バッチ埋め込み
curl -X POST "http://localhost:8080/embed" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["動的計画法", "グラフ理論", "深さ優先探索"]}'

# API情報
curl http://localhost:8080/
```

## ☁️ Cloud Run デプロイ

### 1. Artifact Registry リポジトリ作成

```bash
# GCPプロジェクト設定
export PROJECT_ID=your-project-id
export REGION=asia-northeast1
export REPO_NAME=onnx-embedding

# Artifact Registry リポジトリ作成
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --project=$PROJECT_ID
```

### 2. コンテナ登録

```bash
# Docker認証設定
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# イメージのタグ付け
docker tag onnx-embedding-api \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/embedding-api:latest

# イメージのプッシュ
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/embedding-api:latest
```

### 3. Cloud Run デプロイ

```bash
# Cloud Runサービスのデプロイ
gcloud run deploy embedding-api \
    --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/embedding-api:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300s \
    --concurrency 10 \
    --max-instances 100
```

**または Google Cloud Console での操作**:
1. Cloud Run → サービスを作成
2. コンテナイメージURL: `${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/embedding-api:latest`
3. リージョン: `asia-northeast1`
4. 認証: 未認証の呼び出しを許可（任意）
5. メモリ: 1-2 GiB
6. CPU: 1
7. リクエストタイムアウト: 300秒

## 📊 API仕様

### エンドポイント

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/healthz` | GET | ヘルスチェック |
| `/embed` | POST | バッチ埋め込み |
| `/embed?q=<text>` | GET | 単発埋め込み |
| `/docs` | GET | APIドキュメント |

### レスポンス例

```json
// GET /embed?q=動的計画法
{
    "vector": [0.1, -0.2, 0.3, ...],  // 384次元
    "dimension": 384,
    "text": "動的計画法"
}

// POST /embed
{
    "vectors": [[0.1, -0.2, ...], [0.3, 0.4, ...]],
    "dimension": 384,
    "count": 2
}
```

## ⚙️ 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `MODEL_PATH` | `model_int8.onnx` | モデルファイルパス |
| `TOKENIZER_PATH` | `tokenizer/` | Tokenizerディレクトリ |
| `MAX_LENGTH` | `256` | 最大トークン長 |
| `PORT` | `8080` | APIポート |

## 🔧 精度とパフォーマンス

### INT8量子化の影響

- **サイズ削減**: 約75-80%（例: 400MB → 100MB）
- **推論速度**: 2-3倍高速化
- **精度低下**: コサイン類似度で平均 < 0.02の誤差
- **メモリ使用量**: 大幅削減（1/4程度）

### FP32モデルの使用（高精度が必要な場合）

```bash
# FP32モデルで起動
docker run -e MODEL_PATH=model_fp32.onnx -p 8080:8080 onnx-embedding-api
```

### ベンチマーク例

```python
# 量子化品質テスト（例）
texts = ["動的計画法", "グラフ理論", "深さ優先探索"]
fp32_embeddings = embed_fp32(texts)
int8_embeddings = embed_int8(texts)

# コサイン類似度の差分
for i, text in enumerate(texts):
    similarity = cosine_similarity([fp32_embeddings[i]], [int8_embeddings[i]])[0][0]
    print(f"{text}: {similarity:.4f}")  # 通常 > 0.98
```

## 🚨 トラブルシューティング

### よくある問題

**1. `tokenizer.json が読めない`**
```
FileNotFoundError: Tokenizer file not found: tokenizer/tokenizer.json
```
→ `python export_onnx.py`を実行してtokenizerを生成してください。

**2. `CLS/SEP トークンIDが不正`**
```
警告: CLS/SEP token IDs may be incorrect
```
→ モデル固有のトークンIDを確認してください。通常は CLS:101, SEP:102, PAD:0

**3. `動的軸エラー`**
```
ONNX Runtime error: Invalid input shape
```
→ `export_onnx.py`でdynamic_axesが正しく設定されているか確認してください。

**4. `PORT未設定エラー`**
```
uvicorn: PORT environment variable not set
```
→ `export PORT=8080`を設定するか、`-e PORT=8080`でコンテナを起動してください。

**5. `メモリ不足`**
```
Container killed due to memory limit
```
→ Cloud Runのメモリを1-2GiBに増やしてください。

### デバッグ用コマンド

```bash
# コンテナログの確認
docker logs embedding-api

# コンテナ内での確認
docker exec -it embedding-api /bin/bash
ls -la
python -c "import onnxruntime; print(onnxruntime.__version__)"

# モデルファイルの確認
python -c "
import onnx
model = onnx.load('model_int8.onnx')
print(f'Input: {[inp.name for inp in model.graph.input]}')
print(f'Output: {[out.name for out in model.graph.output]}')
"
```

## 📝 受け入れ条件チェックリスト

- [ ] `python export_onnx.py`で`tokenizer/`と`model_fp32.onnx`が生成される
- [ ] `python quantize.py`で`model_int8.onnx`が生成される
- [ ] `docker run`で起動し、`GET /healthz`が`{"ok": true}`を返す
- [ ] `GET /embed?q=ベルマンフォード`が384次元ベクトルを返す
- [ ] 量子化版と非量子化版のコサイン類似度差分が平均 < 0.02
- [ ] コンテナサイズが100-200MB台
- [ ] Cloud Runで正常動作

## 🏗️ ファイル構成

```
app/
├── export_onnx.py          # ONNX変換（ローカル実行）
├── quantize.py             # INT8量子化
├── main.py                 # FastAPI推論サーバー
├── requirements.txt        # ランタイム依存関係
├── Dockerfile              # Cloud Run用
├── .dockerignore          # Docker除外設定
├── README.md              # このファイル
├── tokenizer/             # Tokenizerファイル（生成）
├── model_fp32.onnx        # FP32モデル（生成）
└── model_int8.onnx        # INT8モデル（生成・本番用）
```

## 📚 参考情報

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Tokenizers Library](https://huggingface.co/docs/tokenizers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)