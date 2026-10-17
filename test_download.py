import json
from pathlib import Path

import requests

game_id = "2023021190"
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

response = requests.get(url, timeout=30)
response.raise_for_status()
data = response.json()

output = Path("data/raw/test") / f"{game_id}.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(data, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Download successful：{output}")
print(f"Top-level fields in JSON：{list(data.keys())}")