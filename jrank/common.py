"""
Common data structures and utilities.
"""

import glob
import json
import os
import re
from typing import Optional

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

reverse_model_map = {
    "model1_id": "model2_id",
    "model2_id": "model1_id",
}


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r", encoding="utf-8") as q_file:
        for line in q_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename, encoding="utf-8") as file_in:
            for line in file_in:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def check_data(questions, model_answers, ref_answers, models, judges):
    """Check the data integrity."""
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def get_model_list(answer_dir):
    """Get the list of model names from the answer directory."""
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names


# common
def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as file_in:
        for answer in file_in:
            qid = json.loads(answer)["question_id"]
            answers[qid] = answer

    q_ids = sorted(list(answers.keys()))
    with open(answer_file, "w", encoding="utf-8") as file_out:
        for qid in q_ids:
            file_out.write(answers[qid])
