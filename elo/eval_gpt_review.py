# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/eval_gpt_review.py
import argparse
import json
import os
import time
import re 

import openai
from tqdm import tqdm

import shortuuid
import logging
import numpy as np
import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys

#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'
assert openai.api_key, 'Please set OPENAI_API_KEY environment variable'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 3
REQ_TIME_GAP = 2

def get_eval(sys_prompt, user_prompt: str, max_tokens: int, model: str):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            #logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(min(5*(i+1), 100))
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def parse_three_class_score(review):
    try:
        #score = int(review.strip().split("\n")[-1].strip())

        matches = re.findall('\d', review)
        score = int(matches[-1])

        return score

    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1
    
def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id
    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1

def get_review(reviewers, prompts, question, answer1, answer2, max_tokens: int, model: str):

    assert (
        answer1['question_id']
        == question['question_id']
        == answer2['question_id']
    )

    sys_prompt, prompt, reviewer_id = gen_prompt(
                reviewers, prompts, question['category'], question['text'], answer1["text"], answer2["text"]
            )
    
    review = get_eval(sys_prompt, prompt, max_tokens, model)

    score = parse_three_class_score(review)

    review_json = {
        "review_id": shortuuid.uuid(),
        "question_id": question['question_id'],
        "answer1_id": answer1["answer_id"], 
        "answer2_id": answer2["answer_id"],
        "answer1": answer1['text'],
        "answer2": answer2['text'],
        "model1_id": answer1['model_id'],
        "model2_id": answer2['model_id'],
        "reviewer_id": reviewer_id,
        "text": review,
        "score": score,
        "metadata": {},
    }

    return review_json

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument("-m", "--model", default='gpt-3.5-turbo-0301')
    parser.add_argument("-id", "--id-key", default='question_id')
    parser.add_argument("-l", "--limit", type=int, default=1000)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    model_names = [elt.split('/')[-1].replace('.jsonl', '') for elt in args.answer_file_list]

    if not os.path.isdir(args.output_review_file):
        dest = args.output_review_file
    else:
        threeclass_suff = "_threeclass" if 'threeclass' in args.prompt_file else ""
        dest = os.path.join(
            args.output_review_file,
            '_vs_'.join(model_names) + f'_{args.model}_reviewer{threeclass_suff}' + '.jsonl'
        )

    print("Loading data...",file=sys.stderr)
    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    question_ids = set(question[args.id_key] for question in question_jsons)
    question_jsons = sorted(question_jsons, key=lambda x: x[args.id_key])
    answer1_jsons = sorted(
        [answer for answer in answer1_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )
    answer2_jsons = sorted(
        [answer for answer in answer2_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )

    print(f"Number of questions: {len(question_jsons)}",file=sys.stderr)
    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    handles = []
    reviews = []
    total_len = len(question_jsons)
    question_idx_list = list(range(min(total_len, args.limit)))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
            
        for i in tqdm(question_idx_list):
            print(f"Question {i+1}/{total_len}",file=sys.stderr)
           
            # Submit tasks to thread pool
            futures.append(executor.submit(get_review, reviewer_jsons, prompt_jsons, question_jsons[i], answer1_jsons[i], answer2_jsons[i], args.max_tokens, args.model))
            logger.info(
                f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
            )
            # To avoid the rate limit set by OpenAI
            time.sleep(REQ_TIME_GAP)        

        for future in as_completed(futures):
            reviews.append(future.result())


    with open(dest, "w") as output_review_file:
        for idx, review in enumerate(reviews):
            output_review_file.write(json.dumps(reviews[idx],ensure_ascii=False) + "\n")
            output_review_file.flush()
