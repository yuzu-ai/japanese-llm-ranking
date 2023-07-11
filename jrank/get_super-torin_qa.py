# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py
"""Generate answers with GPT models"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import concurrent.futures
import os
import time
import json, requests 

import shortuuid
import tqdm
from utils import load_jsonl, save_jsonl

auth_key = os.getenv("SUPER_TORIN_KEY")
assert auth_key, "Please set SUPER_TORIN_KEY environment variable"

api_server_url = 'https://api.tringpt.com/'

headers = {'Authorization': 'Bearer {}'.format(auth_key)}


def get_answer(question_id: int, question: str, max_tokens: int):
    ans = {
        "answer_id": shortuuid.uuid(),
        "question_id": question_id,
        "model_id": "super-torin-sama-alpha2",
        "metadata": {},
    }

    send_body = {
        'text': f"[#ユーザー]\n{question}\n\n[#アシスタント]\n",
        'length': max_tokens,				# 出力するトークン数（1～300）　出力が重いと途中で強制終了する場合があります
        'temperature': 0.7,			# ランダム度（0～2.5）　語彙が単調に感じる場合は上げてみてください
        'top_p': 0.7,				# Top Pサンプリング（0.01～1.0）　1より低いほど確率の低いトークンが除外される。極端に関係のない語彙が出ることを防ぎます
        'rep_pen': 1.15,			# 繰り返しペナルティ（1.0～2.0）　値が高すぎると出力が突飛になりすぎる可能性があります
    }

    for _ in range(3):
        try:
            
            response = requests.post(api_server_url+'/api', headers=headers, json=send_body)
            response_array = json.loads(response.text)
            print(response_array)

            ans["text"] = response_array["data"][0]
            return ans
        except Exception as e:
            print("[ERROR]", e)
            ans["text"] = "#ERROR#"
            time.sleep(5)
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Torin answer generation.")
    parser.add_argument("-q", "--question")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    questions = load_jsonl(os.path.expanduser(args.question))

    answers = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer, question["question_id"], question["text"], args.max_tokens
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            answers.append(future.result())

    answers.sort(key=lambda x: x["question_id"])

    save_jsonl(answers, args.output)
