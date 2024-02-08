"""Generate answers files in batch"""
import subprocess
from tqdm import tqdm

models_list = [
  {"model_path": "line-corporation/japanese-large-lm-1.7b-instruction-sft", "model_id": "line-1.7b", "template_file": "line.json"},
  # {"model_path": "meta-llama/Llama-2-13b-chat-hf", "model_id": "llama2-13b", "template_file": "llama2-chat-ja.json"},
  {
    "model_path": "rinna/nekomata-14b-instruction", "model_id": "rinna-14b", "template_file": "rinna.json"
    }
  ]

for model in tqdm(models_list):
  model_path = model["model_path"]
  model_id = model["model_id"]
  template_file = model["template_file"]

  # Run command based on model
  # Example command: python run_model.py --model_path <model_path> --model_id <model_id> --template_file <template_file>
  command = f"python3 gen_model_answer.py --bench_name rakuda_v2 --model-path {model_path} --model-id {model_id} --conv_template ./templates/{template_file}"
  # Execute the command
  subprocess.run(command, shell=True)  # Uncomment this line to execute the command
