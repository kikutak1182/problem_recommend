# AtCoder Problem Recommendation App - 起動方法

## 📋 前提条件
- Python 3.11以上
- 仮想環境の作成・アクティベート

## 🚀 アプリケーション起動手順

### 1. 仮想環境のセットアップ
```bash
# プロジェクトのルートディレクトリで実行
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. アプリケーション起動
```bash
# ルートディレクトリから起動
python3 app/main.py
```

### 4. アクセス
ブラウザで以下のURLにアクセス：
```
http://127.0.0.1:8000
```

## 📁 ディレクトリ構成
```
problem_recommend/          # ← ここから起動
├── app/
│   ├── main.py            # アプリケーションメイン
│   ├── data/              # 問題データとエンべディング
│   └── templates/         # (未使用)
├── templates/
│   └── index.html         # HTMLテンプレート
├── requirements.txt       # 依存関係
└── venv/                  # 仮想環境
```

## 🔧 トラブルシューティング

### テンプレートエラーが発生する場合
`main.py`の14行目を以下のように修正：
```python
templates = Jinja2Templates(directory="templates")
```

### 依存関係エラーが発生する場合
```bash
# pipのアップデート
pip install --upgrade pip

# 個別インストール
pip install fastapi uvicorn jinja2 requests python-multipart
pip install sentence-transformers numpy torch
```


## 📊 メモリ最適化情報
- **タグ数**: 182個 → 108個（40%削減）
- **エンべディング**: 0.28MB → 0.16MB（43%削減）
- **Render 512MB制限**: 対応済み

## 🤖 問題推薦システムの詳細アルゴリズム

### 📝 入力パラメータ
- **username**: AtCoderユーザー名（オプション）
- **queries**: 推薦に使用するタグ・キーワード（改行区切りで複数入力可能）

### 🔄 推薦処理フロー

#### 1️⃣ **ユーザーレート取得** (username入力時)
```
IF username が入力されている:
    AtCoder APIから最新レート取得
    current_rate = ユーザーの最新レート
ELSE:
    current_rate = None（レート制限なし）
```

#### 2️⃣ **クエリ前処理と拡張**
```
Step 1: クエリ文字化け対応（UTF-8デコード）
Step 2: 改行区切りでクエリリスト生成
Step 3: タグエイリアス辞書でクエリ拡張
        例: "DP" → ["動的計画法", "DP", "Dynamic Programming", "メモ化", "dp"]
Step 4: 重複除去で最終クエリリスト作成
```

#### 3️⃣ **類似タグ検索（メモリ最適化版）**
```
Step 1: 入力クエリをSentenceTransformerでエンコード
Step 2: 事前計算済み108個のタグエンべディングと類似度計算
Step 3: エイリアス完全一致にボーナススコア（1.0または0.95）付与
Step 4: 類似度上位5タグを選出
        重み: [1.0, 0.8, 0.6, 0.4, 0.2]
```

#### 4️⃣ **問題関連度計算（O(N×5)アルゴリズム）**
```
FOR 各問題 in problems_data.json:
    relevance = 0.0
    FOR i, similar_tag in 上位5タグ:
        IF similar_tag in 問題のタグリスト:
            relevance += weights[i]  # 1.0, 0.8, 0.6, 0.4, 0.2
    
    IF relevance > 0:
        external_APIから問題の難易度取得
        IF current_rate is None OR 難易度 > current_rate:
            候補リストに追加
```

#### 5️⃣ **最終選出とランダム化**
```
Step 1: 関連度降順でソート
Step 2: 上位20問を選出
Step 3: 20問からランダムに5問選択
Step 4: 選択された5問を難易度昇順でソート
```

### 📊 **具体例: username="tourist", queries="動的計画法\nDFS"**

1. **レート取得**: `tourist`の最新レート（例: 3500）を取得
2. **クエリ拡張**: 
   - "動的計画法" → ["動的計画法", "DP", "Dynamic Programming", "メモ化", "dp"]
   - "DFS" → ["DFS", "深さ優先探索", "depth-first search", "dfs", "深さ優先"]
3. **類似タグ**: ["動的計画法", "深さ優先探索", "木DP", "bitDP", "グラフ"]
4. **問題候補**: 難易度3500以上の問題のみに限定
5. **関連度計算**: 
   - 動的計画法タグ有り → +1.0
   - 深さ優先探索タグ有り → +0.8
   - 木DPタグ有り → +0.6
6. **最終結果**: 上位20問からランダム5問を表示

### 📊 **具体例: username="", queries="二分探索"**

1. **レート取得**: なし（全難易度対象）
2. **クエリ拡張**: ["二分探索", "binary search", "答えで二分探索", "バイナリサーチ"]
3. **類似タグ**: ["二分探索", "最適化問題", "数学", "貪欲法", "累積和"]
4. **問題候補**: 全問題が対象
5. **最終結果**: 関連度順上位20問からランダム5問

## 🌐 本番デプロイ（Render用）
```bash
# 本番起動コマンド（render.yamlで指定）
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## 📝 使用方法
1. **タグベース推薦**: 「動的計画法」「DFS」などのタグを入力
2. **レート制限**: ユーザー名入力で該当レート以上の問題のみ表示
3. **ランダム選出**: 関連度上位20問から5問をランダム選出