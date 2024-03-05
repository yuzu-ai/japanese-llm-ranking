"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo --bench-name rakuda_v2_test
python3 gen_api_answer.py --model gpt-4 --bench-name rakuda_v2_test
python3 gen_api_answer.py --model claude-2 --bench-name rakuda_v2_test
"""
import argparse
import concurrent.futures
import json
import os
import time

import shortuuid
import tqdm
from helper_api import chat_completion_anthropic, chat_completion_openai, chat_completion_palm
from common import (
    load_questions,
    reorg_answer_file,
    temperature_config,
)
from fastchat.model.model_adapter import get_conversation_template


def get_answer(
    question: dict,
    model: str,
    num_choices: int,
    max_tokens: int,
    answer_file: str,
):
    """Get an answer for a question."""
    if args.force_temperature:
        temperature = args.force_temperature
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        print(f"Model: {model}")
        conv = get_conversation_template(model)
        if conv.system_message:
            conv.system_message = "あなたは役立つアシスタントです。日本語で答えてください。"
        # print(conv)
        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            if 'claude' in model:
                output = chat_completion_anthropic(model, conv, temperature, max_tokens)
            elif 'palm' in model:
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            else:
                output = chat_completion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    print("ANSWER FILE: ", answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as file_out:
        file_out.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="rakuda_v2",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    args = parser.parse_args()

    # if args.openai_api_base is not None:
    #     openai.api_base = args.openai_api_base

    question_file = f"data/{args.benchmark}/questions.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    args.answer_file = f"data/{args.benchmark}/answers/{args.model}.jsonl"
    print(f"Output to {args.answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                args.answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(args.answer_file)
