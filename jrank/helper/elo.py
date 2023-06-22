import copy
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from registry import StandingsRegistry
from reviewer_gpt import get_review
from tqdm import tqdm
from utils import load_jsonl, save_jsonl

# Choose the fundamental parameters that control convergence rate
BETA = 0.09  # Computed to be near optimal # Can be overwritten below if K is set explicitly
ETA = -0.6  # Home field advantage parameter # Computed from observed win rates

# Meaningless shift parameter
INITIAL_ELO = 1000  # Commonly used
# INITIAL_ELO = 0.0 # Easier to understand

# Meaningless scale parameters
BASE = 10.0  # Exponent base
S = 400.0  # Point scale factor

# S = 1.0 # Some of the formulas in the literature assume this
# BASE = np.e # Some of the formulas in the literature assume this

# K scales the point exchange rate.
# In the literature it is viewed as a dependent parameter of BETA
# For S=1 and BASE= np.e , K = BETA
K = BETA * (S / np.log(BASE))  #

# In common practice K is set directly and BETA is then the derived parameter
# K = 32 # Commonly used in literature
K = 15  # Computed to be near-optimal

# If K set explicitly, the input value for BETA is overwritten
if K != BETA * (S / np.log(BASE)):
    BETA = K / (S / np.log(BASE))

# We use bootstrap for the error estimate
BOOTSTRAP_SAMPLES = 10000


class Bot:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path

        if os.path.isfile(self.cache_path):
            self.cache = self.load_cache(cache_path)
            self.model_id = list(self.cache.values())[0][
                "model_id"
            ]  # Unique identifier
        else:
            raise NotImplementedError(
                "Please precompute all bot responses using the get_model_qa.py script"
            )

    def load_cache(self, file_path: str) -> List[Dict]:
        cache = {item["question_id"]: item for item in load_jsonl(file_path)}
        return cache

    def get_response(self, question_id):
        try:
            return self.cache[question_id]
        except KeyError:
            raise NotImplementedError(
                "All responses should be precomputed using the get_model_qa.py script"
            )


class Referee:
    def __init__(
        self,
        cache_path: str,
        reviewer_path: str,
        prompt_path: str,
        model: str,
    ):
        self.cache_path = cache_path
        self.reviewer_path = reviewer_path
        self.prompt_path = prompt_path
        self.reviewers = load_jsonl(reviewer_path)
        self.prompts = load_jsonl(prompt_path)
        self.model = model

        # Load cache if exists; else create an empty cache file
        if os.path.isfile(self.cache_path):
            self.cache = self.load_cache(cache_path)
        else:
            self.cache = {}
            self.add_to_cache([], self.cache_path)

    def load_cache(self, file_path: str) -> List[Dict]:
        cache = load_jsonl(file_path)
        random.shuffle(cache)

        # Check that the cache contains no duplicate entries
        if len(cache) != len(
            set(
                [
                    (item["question_id"], item["model1_id"], item["model2_id"])
                    for item in cache
                ]
            )
        ):
            entries = {}
            for entry in cache:
                key = (entry["question_id"], entry["model1_id"], entry["model2_id"])
                if key not in entries:
                    entries[key] = []
                entries[key].append(entry["review_id"])

            duplicates = [value for key, value in entries.items() if len(value) > 1]

            print(duplicates)

            raise ValueError("Cache contains duplicate entries")
        else:
            return {
                (item["question_id"], item["answer1_id"], item["answer2_id"]): item
                for item in cache
            }

    def add_to_cache(self, data: dict, file_path: str):
        if data:
            self.cache[
                (
                    data["question_id"],
                    data["answer1_id"],
                    data["answer2_id"],
                )
            ] = data

            save_jsonl([data], file_path, write_flag="a")
        else:
            save_jsonl([], file_path, write_flag="a")

    def get_result(self, question, response1, response2):
        try:
            # If a cached result exists, return it
            return self.cache[
                (
                    question["question_id"],
                    response1["answer_id"],
                    response2["answer_id"],
                )
            ]
        except KeyError:
            # If not call the reviewer
            review_json = get_review(
                self.reviewers,
                self.prompts,
                question,
                response1,
                response2,
                self.model,
            )

            # Update the cache
            self.add_to_cache(review_json, self.cache_path)
            return review_json


class EloRanker:
    def __init__(
        self, bots: List[Bot], questions_path: str, referee: Referee, verbose=False
    ):
        self.bots = {bot.model_id: bot for bot in bots}
        self.questions_path = questions_path
        self.questions = {
            question["question_id"]: question for question in load_jsonl(questions_path)
        }
        self.referee = referee
        self.verbose = verbose
        self.matches = []

    def update_elo(self, elo1: float, elo2: float, bot1points: int) -> int:
        expected_bot1 = 1 / (1 + BASE ** ((elo2 - elo1) / S - ETA / np.log(BASE)))
        expected_bot2 = 1 / (1 + BASE ** ((elo1 - elo2) / S + ETA / np.log(BASE)))

        assert round(expected_bot1 + expected_bot2, 6) == 1, 1 - (
            expected_bot1 + expected_bot2
        )

        delta_elo = K * (bot1points - expected_bot1)

        return elo1 + delta_elo, elo2 - delta_elo

    def run_match(self, match):
        bot1_id, bot2_id, question_id = match

        bot1 = self.bots[bot1_id]
        bot2 = self.bots[bot2_id]

        question = self.questions[question_id]

        # Get the cached responses of the bots for the question
        response1 = bot1.get_response(question_id)
        response2 = bot2.get_response(question_id)

        # Get the result of the matchup
        result = self.referee.get_result(question, response1, response2)

        if self.verbose:
            print(f"{bot1.model_id} vs {bot2.model_id}")
            print(f'Question: {question["text"]}')
            print(f"{bot1.model_id}: {response1['text']}")
            print(f"{bot2.model_id}: {response2['text']}")
            print(f"Referee comments: {result['text']}")
            print(
                f"Winner: {[bot1.model_id, bot2.model_id, 'draw'][result['score']-1]}"
            )

        return {
            "bot1": bot1.model_id,
            "bot2": bot2.model_id,
            "question": question["text"],
            "question_id": question["question_id"],
            "response1": response1["text"],
            "response2": response2["text"],
            "referee_comments": result["text"],
            "score": result["score"],
        }

    def prepare_matches(self, num_matchups):
        cached_results = self.referee.cache

        max_matchups = len(self.bots) * (len(self.bots) - 1) * len(self.questions)

        if num_matchups > max_matchups:
            num_matchups = max_matchups

        if self.verbose:
            print(
                f"Preparing {num_matchups} matchups out of {max_matchups} total possible"
            )

        matchup = 0

        bot_num_matches = defaultdict(int)

        # Run cached result first
        for cached_result in cached_results.values():
            if matchup >= num_matchups:
                break

            try:
                bot1 = self.bots[cached_result["model1_id"]]
                bot2 = self.bots[cached_result["model2_id"]]
                question_id = self.questions[cached_result["question_id"]][
                    "question_id"
                ]

                bot_num_matches[bot1.model_id] += 1
                bot_num_matches[bot2.model_id] += 1

                self.matches.append(
                    {
                        "match": (bot1.model_id, bot2.model_id, question_id),
                        "result": cached_result,
                    }
                )
                matchup += 1
            except KeyError:
                continue

        while matchup < num_matchups:
            # Select bot with lowest number of matches
            bot_min_matches_id = min(self.bots, key=lambda x: bot_num_matches[x])

            # Select another bot randomly
            bot_random_id = random.choice(
                [
                    model_id
                    for model_id, bot in self.bots.items()
                    if model_id != bot_min_matches_id
                ]
            )

            # Randomly assign bot_min_matches and bot_random as bot1 and bot2
            bot1_id, bot2_id = random.sample([bot_min_matches_id, bot_random_id], 2)

            # Select a random question id
            question_id = random.choice(list(self.questions.keys()))

            # Check if the matchup has already been played and if not run it
            if (bot1_id, bot2_id, question_id) not in [
                x["match"] for x in self.matches
            ]:
                bot_num_matches[bot1_id] += 1
                bot_num_matches[bot2_id] += 1
                self.matches.append(
                    {"match": (bot1_id, bot2_id, question_id), "result": None}
                )
                matchup += 1

        if self.verbose:
            print(f"{num_matchups} matchups prepared")

    def run_matches(self, num_matchups):
        self.prepare_matches(num_matchups)

        # Create a pool of workers to process the matches
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for i, x in enumerate(self.matches):
                if not x["result"]:
                    futures.append(executor.submit(self.run_match, x["match"]))

            for future in tqdm(as_completed(futures)):
                self.matches[i] = {"match": x["match"], "result": future.result()}

    def run_tournament(self, matches=None):
        if not matches:
            matches = self.matches

        tournament = []

        bot_num_matches = defaultdict(lambda: 0)
        bot_elos = defaultdict(lambda: INITIAL_ELO)

        for x in matches:
            match = x["match"]
            result = x["result"]

            bot1_id = match[0]
            bot2_id = match[1]

            bot_num_matches[bot1_id] += 1
            bot_num_matches[bot2_id] += 1

            old_elo1 = bot_elos[bot1_id]
            old_elo2 = bot_elos[bot2_id]

            score = result["score"]

            if score == 1:
                elo1, elo2 = self.update_elo(old_elo1, old_elo2, 1)
            elif score == 2:
                elo1, elo2 = self.update_elo(old_elo1, old_elo2, 0)
            elif score == 3:
                elo1, elo2 = self.update_elo(old_elo1, old_elo2, 0.5)
            else:
                raise ValueError(
                    f"Invalid score: {score} on match {match} result {result}"
                )

            bot_elos[bot1_id] = elo1
            bot_elos[bot2_id] = elo2

            result_with_elo = copy.copy(result)

            result_with_elo["prematch_elo1"] = old_elo1
            result_with_elo["prematch_elo2"] = old_elo2
            result_with_elo["postmatch_elo1"] = elo1
            result_with_elo["postmatch_elo2"] = elo2

            tournament.append(result_with_elo)

        return tournament, bot_elos, bot_num_matches

    def compute_bootstrap_elos(self, num_samples):
        """Compute bootstrap confidence intervals"""
        rows = []
        for i in tqdm(range(num_samples), desc="bootstrap"):
            bootstrapped_matches = random.choices(self.matches, k=len(self.matches))

            _, bot_elos, _ = self.run_tournament(bootstrapped_matches)

            rows.append(bot_elos)

        df = pd.DataFrame(rows)
        df = df[df.median().sort_values(ascending=False).index]

        bootstrap_elo = (
            pd.DataFrame(
                dict(
                    # lower=df.quantile(0.025),
                    lower=df.quantile(0.16),
                    median=df.quantile(0.5),
                    upper=df.quantile(0.84),
                    # upper=df.quantile(0.975),
                )
            )
            .reset_index()
            .rename(columns={"index": "model_id"})
            .sort_values("median", ascending=False)
        )
        bootstrap_elo["error_y_plus"] = bootstrap_elo["upper"] - bootstrap_elo["median"]
        bootstrap_elo["error_y_minus"] = (
            bootstrap_elo["median"] - bootstrap_elo["lower"]
        )
        return bootstrap_elo

    def generate_standings(self):
        # Run the tournament

        if self.verbose:
            print(
                f"Running a tournament with {len(self.bots)} bots and {len(self.matches)} matches"
            )
            print(
                f"BETA: {BETA}, K: {K}, INITIAL_ELO: {INITIAL_ELO}, BASE: {BASE}, S: {S}"
            )

        tournament, bot_elos, bot_num_matches = self.run_tournament()

        self.tournament = copy.deepcopy(tournament)

        # Construct the standings
        sorted_elos = {
            k: v
            for k, v in sorted(bot_elos.items(), key=lambda item: item[1], reverse=True)
        }
        self.standings = pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "elo": elo,
                    "num_matches": bot_num_matches[model_id],
                }
                for model_id, elo in sorted_elos.items()
            ]
        )

        # Compute the bootstrapped ELO
        bootstrap_elos = self.compute_bootstrap_elos(BOOTSTRAP_SAMPLES)

        # Merge into the standings
        self.standings = self.standings.merge(bootstrap_elos)

        # Print the standings
        if self.verbose:
            print("=== Final Standings ===")
            pd.options.display.float_format = "{:.3f}".format
            print(
                self.standings[
                    [
                        "model_id",
                        "elo",
                        "num_matches",
                        "error_y_plus",
                        "error_y_minus",
                        "median",
                    ]
                ]
            )
            print(
                f"BETA: {BETA}, K: {K}, INITIAL_ELO: {INITIAL_ELO}, BASE: {BASE}, S: {S}"
            )

    def output_standings(self, output_path):
        """Convert bots to JSON format and save to a file."""

        standings_json = self.standings.to_dict(orient="records")

        question_metadata = {
            "questions_path": self.questions_path,
            "reviewer_path": self.referee.reviewer_path,
            "prompt_path": self.referee.prompt_path,
        }

        bot_metadata = {
            bot.model_id: {"model_id": bot.model_id, "path": bot.cache_path}
            for bot in self.bots.values()
        }

        elo_metadata = {
            "BETA": BETA,
            "K": K,
            "INITIAL_ELO": INITIAL_ELO,
            "BASE": BASE,
            "S": S,
            "ETA": ETA,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        }

        output = {
            "date": datetime.now().isoformat(),
            "elo_metadata": elo_metadata,
            "bot_metadata": bot_metadata,
            "metadata": question_metadata,
            "rankings": standings_json,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def output_tournament(self, output_path):
        """Convert bots to JSON format and save to a file."""
        save_jsonl(self.tournament, output_path)


if __name__ == "__main__":
    bots = [
        Bot("answers/rakuda_koukou_v0/gpt3.jsonl"),
        Bot("answers/rakuda_koukou_v0/rinna-ppo.jsonl"),
        Bot("answers/rakuda_koukou_v0/rinna-sft.jsonl"),
        Bot("answers/rakuda_koukou_v0/rinna.jsonl"),
        Bot("answers/rakuda_koukou_v0/stormy.jsonl"),
        Bot("answers/rakuda_koukou_v0/calm.jsonl"),
    ]

    referee = Referee(
        "matchups/rakuda_koukou_v0_v2prompt.jsonl",
        # "matchups/rakuda_koukou_v0.jsonl",
        "prompts/rakuda_reviewer.jsonl",
        "prompts/rakuda_prompt_threeclass_v2.jsonl",
        model="gpt-3.5-turbo-0301",
    )

    ranker = EloRanker(bots, "questions/rakuda_koukou_v0.jsonl", referee, verbose=True)
    ranker.run_matches(1200)
    ranker.generate_standings()

    # ranker.output_standings(
    #     "tournaments/rakuda_koukou_v0_tournament_result.json"
    # )
    # ranker.output_tournament("tournaments/rakuda_koukou_v0_tournament.jsonl")

    ranker.output_standings(
        "tournaments/rakuda_koukou_v0_promptv2_tournament_result.json"
    )
    ranker.output_tournament("tournaments/rakuda_koukou_v0_promptv2_tournament.jsonl")

    # registry = StandingsRegistry("./registry/registry.jsonl")
    # registry.register("tournaments/rakuda_koukou_v0_promptv2_tournament_result.json")
    # registry.convert_to_markdown(
    #     "./registry/benchmark_template.md", "./registry/output/benchmark.md"
    # )
