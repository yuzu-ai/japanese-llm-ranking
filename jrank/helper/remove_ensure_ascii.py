import json


def load_jsonl(input_path):
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


# Specify your input and output paths
input_path = "./matchups/rakuda_koukou_v0.jsonl"
output_path = "./matchups/rakuda_koukou_v0.jsonl"

# Load the data
data = load_jsonl(input_path)

# Save the data with ensure_ascii=False
save_jsonl(data, output_path)
