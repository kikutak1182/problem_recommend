import json
import re

# ファイルパス
TAGS_JSON = "tag/tags_sample1.json"
TAG_LIST = "tag/tag_list.txt"

# 1. tags_sample1.jsonから全タグを抽出
with open(TAGS_JSON, encoding="utf-8") as f:
    data = json.load(f)

all_tags = set()
for v in data.values():
    tags_str = v.get("tags", "")
    # タグ部分を抽出（```json\n[...]\n``` 形式 or [...] 形式）
    m = re.search(r"\[.*\]", tags_str, re.DOTALL)
    if m:
        try:
            tags = json.loads(m.group(0))
            all_tags.update(tags)
        except Exception:
            pass

# 2. tag_list.txtの既存タグを取得
with open(TAG_LIST, encoding="utf-8") as f:
    tag_list = set(line.strip() for line in f if line.strip())

# 3. 差分を抽出
diff_tags = sorted(all_tags - tag_list)

# 4. 差分があればtag_list.txtに追記
if diff_tags:
    with open(TAG_LIST, "a", encoding="utf-8") as f:
        for tag in diff_tags:
            f.write(tag + "\n")
    print("以下のタグをtag_list.txtに追記しました：")
    for tag in diff_tags:
        print(tag)
else:
    print("新しいタグはありませんでした。")
