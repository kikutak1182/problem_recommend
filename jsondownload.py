import requests

url = "https://kenkoooo.com/atcoder/resources/problems.json"
response = requests.get(url)
with open("problems.json", "w", encoding="utf-8") as f:
    f.write(response.text)