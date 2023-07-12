import json
from typing import Dict, List


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str, write_flag: str = "w"):
    with open(file_path, write_flag, encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
