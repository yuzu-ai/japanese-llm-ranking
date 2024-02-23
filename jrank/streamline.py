"""This streamline script runs the entire pipeline from start to finish."""

import json
import subprocess

# Load the configuration file
try:
    config_file_path = "config.json"  # Replace with the actual file path
    with open(config_file_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Config file `config.json` not found. Please check the file path.")
    exit(1)

# Run the local models
local_models = config["models"]["local_models"]
if len(local_models) > 0:
    for model in local_models:
        print(f"Running {model['id']}")
        command = [
            "python3",
            "gen_local_model_answer.py",
            "--bench_name",
            f"{config['benchmark_name']}",
            "--model-path",
            f"{model['path']}",
            "--model-id",
            f"{model['id']}",
            "--conv_template",
            f"{model['conversation_template']}",
            "--repetition_penalty",
            f"{config['repetition_penalty']}",
            "--max_new_tokens",
            f"{config['max_token_length']}",
            "--top_p",
            f"{config['top_p']}",
            "--n",
            "200"
        ]

        subprocess.run(command, check=True)
else:
    print("No local models to run.")

# Run the api models
api_models = config["models"]["api_models"]
if len(api_models) > 0:
    for model in api_models:
        print(f"Running {model['id']}")
        command = [
            "python3",
            "gen_api_model_answer.py",
            "--model",
            f"{model['path']}",
            "--benchmark",
            f"{config['benchmark_name']}",
            "--question-begin",
            "0",
            "--question-end",
            "1",
        ]

        subprocess.run(command, check=True)
else:
    print("No API models to run.")

# Run the judgment
print("Time for judgement!")
judge_models = config["models"]["judge_models"]
if len(judge_models) > 0:
    for model in judge_models:
        command = [
            "python3",
            "gen_judgment.py",
            "--bench_name",
            f"{config['benchmark_name']}",
            "--judge-model",
            f"{model['path']}",
            "--baseline-model",
            f"{config['models']['baseline_model']}",
            "--mode",
            f"{config['evaluation_mode']}",
            "--n",
            "200"
        ]

        subprocess.run(command, check=True)
else:
    print("No judge models to run.")

# Done
print("Rakuda completed run. Please check result files.")
