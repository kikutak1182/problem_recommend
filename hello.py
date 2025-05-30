import requests, pandas as pd, time

BASE = "https://kenkoooo.com/atcoder"
def get_user_rating(user):
    r = requests.get(f"https://atcoder-api.herokuapp.com/user/{user}", timeout=10)
    r.raise_for_status()
    return r.json()["rating"]

def get_solved(user):
    url = f"{BASE}/atcoder-api/v3/user/submissions?user={user}&from_second=0"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return {f"{s['contest_id']}/{s['problem_id']}"
            for s in r.json() if s["result"] == "AC"}

def load_problem_meta():
    problems = pd.read_json(f"{BASE}/resources/problems.json")
    models  = pd.read_json(f"{BASE}/resources/problem-models.json")
    return problems.merge(models[['contest_id','problem_id','difficulty']],
                          on=['contest_id','problem_id'])

def recommend(user, diff=200, topk=20):
    rating = get_user_rating(user)
    solved = get_solved(user)
    time.sleep(1)                 # polite
    df = load_problem_meta()
    cand = df[(df.difficulty>-1) &
              (abs(df.difficulty - rating) <= diff) &
              ~df.apply(lambda r: f"{r.contest_id}/{r.problem_id}" in solved, axis=1)]
    return cand.sort_values('difficulty').head(topk)[['contest_id','problem_id','difficulty']]

if __name__ == "__main__":
    print(recommend("tourist"))
