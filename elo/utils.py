import json
from typing import Dict, List


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r") as file:
        for line in file:
            print(line)
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
