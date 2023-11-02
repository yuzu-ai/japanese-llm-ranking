"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all|pairwise-n] --judge-model [gpt-4|gpt-3.5-turbo|claude-2] --n ["all"|int]

python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all|pairwise-n] --judge-model [gpt-4|gpt-3.5-turbo|claude-2] --n ["all"|int]

python gen_judgment.py --bench-name rakuda_v2_test --model-list claude-2 gpt-3.5-turbo line-1.7b --parallel 1 --mode pairwise-n --judge-model claude-2 --n 2

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model claude-2 --n 2000

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model gpt-4 --n 1400

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo  rwkv-world-jp-v1 stablelm-alpha-7b-v2 weblab-10b-instruction-sft llm-jp-13b-instruct youri-7b-instruction stablelm-gamma-7b stablelm-beta-70b --parallel 2 --mode pairwise-n --judge-model gpt-4 --n 2200

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rwkv-world-jp-v1 stablelm-alpha-7b-v2 weblab-10b-instruction-sft llm-jp-13b-instruct  youri-7b-instruction stablelm-gamma-7b stablelm-beta-70b --parallel 2 --mode pairwise-n --judge-model claude-2 --n 12000


"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, combinations
from functools import partial
import json
from random import shuffle, choice
import os
import numpy as np
from tqdm import tqdm

from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)

def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers[judge.model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return 


def make_n_match_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
    cache_file=None,
    n=None,
):
    
    matches = []

    if os.path.exists(cache_file):
        print(f"Importing matches from cache {cache_file}")
        with open(output_file, 'r') as file:
            for line in file:
                if len(matches) == n:
                    break
                data = json.loads(line)
                model1_id = next(
                    (key for key in models if key == data["model1_id"]), None
                )
                model2_id = next(
                    (key for key in models if key == data["model2_id"]), None
                )
                question_id = next(
                    q["question_id"]
                    for q in questions
                    if q["question_id"] == data["question_id"]
                )
                if model1_id and model2_id and question_id and judge.model_name == data["judge"][0]:
                    matches.append((model1_id, model2_id, question_id))
                else:
                    #print(f"{model1_id} {model2_id} {question_id} {judge.model_name}")
                    print(f"Rejecting {data['model1_id']} {data['model2_id']} {data['judge']}")
                    #raise RuntimeError("Match in cache does not match the current settings")

        print(f"Number of matches imported from cache {len(matches)}")

    
    #all_possible_pairs = list(combinations(models, 2))
    all_possible_pairs = list(permutations(models, 2))
    all_possible_new_matches = []
    for model1_id, model2_id in all_possible_pairs:
        for q in questions:
            if multi_turn and len(q["turns"]) != 2:
                continue
            if (model1_id, model2_id, q["question_id"]) not in matches:
                all_possible_new_matches.append(
                    (model1_id, model2_id, q["question_id"])
                )
    shuffle(all_possible_new_matches)

    # Create a dictionary to count the number of matches for each bot pair
    model_match_counts = {model: 0 for model in models}

    # Update the dictionary with the number of existing matches
    for match in matches:
        model_match_counts[match[0]] += 1
        model_match_counts[match[1]] += 1
    
    while len(matches) < n and all_possible_new_matches:
        # Sort bots by the number of matches they have participated in
        sorted_bots = sorted(model_match_counts.items(), key=lambda x: x[1])

        # Select a bot with fewest matches
        selected_bot = choice(
            [bot for bot, count in sorted_bots if count == sorted_bots[0][1]]
        )

        # Select a match involving the selected bot
        selected_matches = [
            match for match in all_possible_new_matches if selected_bot in match
        ]

        # Select a match and add it to matches
        possible_match = choice(selected_matches)
        if possible_match not in matches:
            matches.append(possible_match)
            all_possible_new_matches.remove(possible_match)

            # Update the number of matches for each bot
            model_match_counts[possible_match[0]] += 1
            model_match_counts[possible_match[1]] += 1
        else:
            raise RuntimeError(
                "Match already exists in matches despite being drawn from all_possible_new_matches"
            )

    print(f"Finished generating matchups")

    match_pairs = []
    for match in matches:
        m_1 = match[0]
        m_2 = match[1]
        q_id = match[2]

        a_1 = model_answers[m_1][q_id]
        a_2 = model_answers[m_2][q_id]

        q = [q for q in questions if q["question_id"] == q_id][0]
        match = MatchPair(
            dict(q),
            m_1,
            m_2,
            a_1,
            a_2,
            judge,
            ref_answer=ref_answers[judge.model_name][q_id] if ref_answers is not None else None,
            multi_turn=multi_turn,
        )

        match_pairs.append(match)

    return match_pairs


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-ja"])
    # judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    # judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    # judges["default-mt"] = Judge(
    #     judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    # )
    # judges["math-mt"] = Judge(
    #     judge_model,
    #     judge_prompts["pair-math-v1-multi-turn"],
    #     ref_based=True,
    #     multi_turn=True,
    # )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="rakuda_v2",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "pairwise-n", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`pairwise-n` runs pairwise comparision between n pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--n", type=int, help="For pairwise-n mode, run `n` judgments."
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Whether to skip user confirmation before starting the evaluation.",
    )
    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/questions.jsonl"
    answer_dir = f"data/{args.bench_name}/answers"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        elif args.mode == "pairwise-n":
            make_match_func = partial(make_n_match_pairs, cache_file=output_file, n=args.n)
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    print(f"model_answers: {model_answers.keys()}")
    print(f"models: {models}")
    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    # matches += make_match_func(
    #     question_math,
    #     models,
    #     model_answers,
    #     judges["math"],
    #     baseline_model,
    #     ref_answers,
    # )
    # matches += make_match_func(
    #     question_default,
    #     models,
    #     model_answers,
    #     judges["default-mt"],
    #     baseline_model,
    #     multi_turn=True,
    # )
    # matches += make_match_func(
    #     question_math,
    #     models,
    #     model_answers,
    #     judges["math-mt"],
    #     baseline_model,
    #     ref_answers,
    #     multi_turn=True,
    # )

    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    # print("Matches:")
    # print(matches)
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    if not args.skip_confirm:
        input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass