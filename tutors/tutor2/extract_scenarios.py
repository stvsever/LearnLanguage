import json
from pathlib import Path

path = Path(__file__).resolve().parent / "scenarios_out" / "scenarios_advanced.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

titles = list(data.keys())
total = len(titles)

for i, title in enumerate(titles, start=1):
    print(f"{title} ({i}/{total})")
