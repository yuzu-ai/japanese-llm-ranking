# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py
"""Generate answers with GPT models"""
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import argparse
import concurrent.futures
import os
import time

import openai
import shortuuid
import tqdm
from utils import load_jsonl, save_jsonl

openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "Please set OPENAI_API_KEY environment variable"

# MODEL = "gpt-3.5-turbo-0301"
# MODEL_ID = "gpt-3.5-turbo-0301:20230614"

MODEL = "gpt-4"
MODEL_ID = "gpt-4:20230713"


def get_answer(question_id: int, question: str, max_tokens: int):
    """
    Retrieve answer from GPT model.

    Args:
        question_id (int): The ID of the question.
        question (str): The question to ask the GPT model.
        max_tokens (int): The maximum number of tokens to generate in the response.

    Returns:
        dict: A dictionary containing the answer ID, question ID, model ID, and the generated text response.
    """

    ans = {
        "answer_id": shortuuid.uuid(),
        "question_id": question_id,
        "model_id": MODEL_ID,
        "metadata": {},
    }
    for _ in range(3):
        try:
            time.sleep(15)
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "あなたは役立つアシスタントです。"},
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_tokens=max_tokens,
            )
            ans["text"] = response["choices"][0]["message"]["content"]
            return ans
        except Exception as e:
            print("[ERROR]", e)
            ans["text"] = "#ERROR#"
            time.sleep(15)
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT answer generation.")
    parser.add_argument("-q", "--question")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    questions = load_jsonl(os.path.expanduser(args.question))

    if os.path.exists(os.path.expanduser(args.output)):
        answers = load_jsonl(os.path.expanduser(args.output))
        answers = [a for a in answers if a["text"] != "#ERROR#"]
    else:
        answers = []

    questions = [
        q
        for q in questions
        if q["question_id"] not in [a["question_id"] for a in answers]
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for q in questions:
            future = executor.submit(
                get_answer, q["question_id"], q["text"], args.max_tokens
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            answers.append(future.result())

    answers.sort(key=lambda x: x["question_id"])

    save_jsonl(answers, args.output)
