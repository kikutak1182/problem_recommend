import requests

# ユーザーネームの入力
username = input('AtCoderのユーザーネームを入力してください: ')

# レート履歴取得
user_info_url = f'https://atcoder.jp/users/{username}/history/json'
user_info = requests.get(user_info_url).json()
if user_info:
    current_rate = user_info[-1]['NewRating']
    print(f'現在レート: {current_rate}')
else:
    print('ユーザー情報が取得できませんでした')
    exit()

# 問題リスト・diff情報取得
problems = requests.get('https://kenkoooo.com/atcoder/resources/problems.json').json()
problem_models = requests.get('https://kenkoooo.com/atcoder/resources/problem-models.json').json()

# レートとdiffが近い問題を抽出
recommend = []
for p in problems:
    pid = p['id']
    title = p['title']
    contest_id = p['contest_id']
    diff = problem_models.get(pid, {}).get('difficulty')
    if diff is not None:
        diff_gap = abs(diff - current_rate)
        recommend.append((diff_gap, contest_id, title, diff, pid))
recommend.sort()
print('\n--- レートとdiffが近いおすすめ問題 ---')
for i, (gap, contest_id, title, diff, pid) in enumerate(recommend[:10], 1):
    print(f'{i}. {contest_id} - {title} (diff: {diff}) https://atcoder.jp/contests/{contest_id}/tasks/{pid}') 