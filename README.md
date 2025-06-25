Atcoderのユーザーネームと学びたいアルゴリズムを入力すると、関連した自分のレートに近いdifficultyの問題を提案してくれます。


![スクリーンショット 2025-06-25 23 17 13](https://github.com/user-attachments/assets/730f19f4-c732-4229-918e-4d740d567440)
(使用例の画像　関連タグが表示されていないので修正予定)


sentencetransformerで入力アルゴリズムとタグとのコサイン類似度を計算し、類似度の高いタグを抽出、関連する問題を検索してユーザーのレートに近い問題を推薦します。
複数のアルゴリズム/単語を組み合わせた検索が可能です。

https://atcoder-problem-recommend.onrender.com/
