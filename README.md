Atcoderのユーザーネームと学びたいアルゴリズムから、関連した自分のレートに近いdifficultyの問題を提案してくれます。
その過程でAtcoderの問題のタグ付けと、URLの結びつけを行なったので、個人開発などご自由にお使いください。

アプリURL:　https://problem-recommend-150123537389.asia-northeast1.run.app/

下図：アプリのイメージ
<img width="1400" height="875" alt="image" src="https://github.com/user-attachments/assets/a5e12bc7-1cb1-463b-8226-147a1c1b3a34" />

<img width="1400" height="873" alt="image" src="https://github.com/user-attachments/assets/960df563-e88a-4b73-897e-70d5175f83e9" />



AI、自然言語処理、ルールベースからタグを自作しました。
関連する問題を検索してユーザーのレートに近い問題を推薦します。複数のアルゴリズム/単語を組み合わせた検索が可能です。
Google Cloud Runでデプロイしています。


ディレクトリ構成
```
  .
  ├── .dockerignore
  ├── .gitignore
  ├── Dockerfile
  ├── README.md                          # プロジェクト説明
  ├── requirements.txt                   # Python依存関係
  └── app
      ├── main.py                        # アプリケーション本体
      ├── templates
      │   └── index.html                 # メインページHTML
      └── data
          ├── editorial_mappings.json    # 解説URLデータ
          ├── problems_data.json         # 問題タグデータ
          └── tag_definitions.json       # タグ定義データ

```

