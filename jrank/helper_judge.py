"""Helper functions for judgment result generation."""

import os
import json
import time
import re
import ast
import dataclasses
from typing import Optional

from fastchat.model.model_adapter import get_conversation_template

from helper_api import chat_completion_anthropic, chat_completion_openai
from common import (
    one_score_pattern,
    one_score_pattern_backup,
    two_score_pattern,
    two_score_pattern_backup,
    reverse_model_map,
    TIE_DELTA,
    NEED_REF_CATS,
)


@dataclasses.dataclass
class Judge:
    """A judge model."""

    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    """A single model competing on a question."""

    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: Optional[dict] = None
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    """A pair of models competing on a question."""

    question: dict
    model1_id: str
    model2_id: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: Optional[dict] = None
    multi_turn: bool = False


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file, encoding="utf-8") as file_in:
        for line in file_in:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model1_id, model2_id = gamekey
    if model1_id < model2_id:
        return gamekey, result
    else:
        new_gamekey = (qid, model2_id, model1_id)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def run_judge_single(question, answer, judge, ref_answer=None, multi_turn=False):
    """Run a single answer grading judge."""
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if "gpt" in model:
        judgment = chat_completion_openai(model, conv, temperature=0.0, max_tokens=2048)
    elif "claude" in model:
        judgment = chat_completion_anthropic(
            model, conv, temperature=0.0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def play_a_match_single(match: MatchPair, output_file: str):
    """Play a single match."""
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, multi_turn=multi_turn
        )

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, turn: {turn}, model: {model}, "
            f"score: {score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as file_out:
            file_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    """Run a pairwise answer grading judge."""
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )

    winner = "error"

    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
        conv.set_system_message(system_prompt)
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif "claude" in model:
        if system_prompt != "You are a helpful assistant.":
            user_prompt = "[Instruction]\n" + system_prompt + "\n\n" + user_prompt
            conv.messages[0][1] = user_prompt
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[A]]":
        if "[[A]]" in judgment:
            winner = "A"
        elif "[[B]]" in judgment:
            winner = "B"
        elif "[[C]]" in judgment:
            winner = "tie"
        else:
            winner = "error"
    elif judge.prompt_template["output_format"] == "[[rating_a,rating_b]]":
        match = re.search(two_score_pattern, judgment)
        if not match:
            match = re.search(two_score_pattern_backup, judgment)
        if match:
            scores = [ast.literal_eval(s.strip()) for s in match.groups()]
            if abs(scores[0] - scores[1]) <= TIE_DELTA:
                winner = "tie"
            elif scores[0] > scores[1]:
                winner = "A"
            else:
                winner = "B"
        else:
            winner = "error"
    elif judge.prompt_template["output_format"] == "last":
        try:
            matches = re.findall("\d", judgment)
            score = int(matches[-1])
            assert score in [1, 2, 3]
            winner = score
        except Exception as e:
            print(e)
            winner = "error"

    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return winner, user_prompt, judgment


def play_a_match_pair(match: MatchPair, output_file: str):
    """Play a pair match."""
    (
        question,
        model1_id,
        model2_id,
        answer_1,
        answer_2,
        judge,
        ref_answer,
        multi_turn,
    ) = (
        match.question,
        match.model1_id,
        match.model2_id,
        match.answer_1,
        match.answer_2,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                if (
                    data["model1_id"] == model1_id
                    and data["model2_id"] == model2_id
                    and data["question_id"] == question["question_id"]
                    and data["judge"][0] == judge.model_name
                ):
                    return data

    if judge.prompt_template["type"] == "pairwise":
        winner, prompt, judgment = run_judge_pair(
            question, answer_1, answer_2, judge, ref_answer, multi_turn=multi_turn
        )
        # g2_winner, g2_user_prompt, g2_judgment = run_judge_pair(
        #     question, answer_2, answer_1, judge, ref_answer, multi_turn=multi_turn
        # )

        g1_map = {"A": "model1_id", "B": "model2_id"}
        # g2_map = {"A": "model2_id", "B": "model1_id"}
        winner = g1_map.get(winner, winner)
        # g2_winner = g2_map.get(g2_winner, g2_winner)
        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2

        result = {
            "question_id": question_id,
            "model1_id": model1_id,
            "model2_id": model2_id,
            "winner": winner,
            # "g1_winner": g1_winner,
            # "g2_winner": g2_winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "prompt": prompt,
            "judgment": judgment,
            # "g2_user_prompt": g2_user_prompt,
            # "g2_judgment": g2_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }

        print(
            f"question: {question_id}, turn: {turn}, model1_id: {model1_id}, model2_id: {model2_id}, "
            f"winner: {winner},"  # g2_winner: {g2_winner}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    elif judge.prompt_template["type"] == "single":
        m1_score, m1_user_prompt, m1_judgment = run_judge_single(
            question, answer_1, judge
        )
        m2_score, m2_user_prompt, m2_judgment = run_judge_single(
            question, answer_2, judge
        )

        if abs(m1_score - m2_score) <= TIE_DELTA:
            winner = "tie"
        elif m1_score > m2_score:
            winner = "model1_id"
        else:
            winner = "model2_id"

        question_id = question["question_id"]
        result = {
            "question_id": question_id,
            "model1_id": model1_id,
            "model2_id": model2_id,
            "winner": winner,
            # "g1_winner": winner,
            # "g2_winner": winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": m1_user_prompt,
            "g1_judgment": m1_judgment,
            "g2_user_prompt": m2_user_prompt,
            "g2_judgment": m2_judgment,
            "m1_score": m1_score,
            "m2_score": m2_score,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, model1_id: {model1_id}, model2_id: {model2_id}, "
            f"winner: {winner}, m1_score: {m1_score}, m2_score: {m2_score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as file_out:
            file_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def load_pairwise_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename, encoding="utf-8"):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model1_id, model2_id = (
            obj["question_id"],
            obj["model1_id"],
            obj["model2_id"],
        )

        if judge not in judge_dict:
            judge_dict[judge] = {}

        if "winner" in obj:
            winner = obj["winner"]
        elif "g1_winner" in obj and "g2_winner" in obj:
            g1_winner, g2_winner = obj["g1_winner"], obj["g2_winner"]
            if g1_winner == g2_winner:
                winner = g1_winner
            else:
                winner = "inconsistent"
        else:
            raise ValueError(f"Invalid keys: {list(obj.keys())}")

        gamekey = (qid, model1_id, model2_id)
        winners = (winner,)

        judge_dict[judge][gamekey] = {
            "winners": winners,
            "g1_judgment": obj["g1_judgment"],
            "g2_judgment": obj["g2_judgment"],
        }

    # Make the model names sorted in the game keys
    normalized = {}
    for judge, value in judge_dict.items():
        normalized[judge] = normalize_game_key_dict(value)
    return normalized


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename, encoding="utf-8"):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def resolve_pairwise_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct pairwise judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "pair-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "pair-v2-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "pair-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "pair-v2")]


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "single-v1")]


def get_pairwise_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model1_id, model2_id = gamekey
        if model1_id < model2_id:
            res = judgment_dict[gamekey]
            g1_judgment, g2_judgment = res["g1_judgment"], res["g2_judgment"]
        else:
            new_gamekey = (qid, model2_id, model1_id)
            res = judgment_dict[new_gamekey]

            # model1_id, model2_id = model1_id, model2_id
            g1_judgment, g2_judgment = res["g2_judgment"], res["g1_judgment"]

        return (
            f"**Game 1**. **A**: {model1_id}, **B**: {model2_id}\n\n"
            f"**Judgment**: {g1_judgment}"
            + "\n\n`--------------------------`\n\n"
            + f"**Game 2**. **A**: {model2_id}, **B**: {model1_id}\n\n"
            f"**Judgment**: {g2_judgment}"
        )
    except KeyError:
        return "N/A"


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        _q_id, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"
