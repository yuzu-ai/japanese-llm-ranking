import copy
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from reviewer_gpt import get_review
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
import json
from registry import StandingsRegistry
from datetime import datetime

## Choose the fundamental parameters that control convergence rate
BETA = 0.05  # Computed to be near optimal # Can be overwritten below if K is set explicitly
ETA = -0.8  # Home field advantage parameter # Computed from observed win rates

## Meaningless shift parameter
INITIAL_ELO = 1000  # Commonly used
# INITIAL_ELO = 0.0 # Easier to understand

# Meaningless scale parameters
BASE = 10.0  # Commonly used
S = 400.0  # Commonly used

# S = 1.0 # Some of the formulas in the literature assume this
# BASE = np.e # Some of the formulas in the literature assume this

# K scales the point exchange rate.
# In the literature it is viewed as a dependent parameter of BETA
# For S=1 and BASE= np.e , K = BETA
K = BETA * (S / np.log(BASE))  #

# In common practice K is set directly and BETA is then the derived parameter
# K = 32 # Commonly used in literature
K = 10  # Computed to be near-optimal

# If K set explicitly, the input value for BETA is overwritten
if K != BETA * (S / np.log(BASE)):
    BETA = K / (S / np.log(BASE))

# We use bootstrap for the error estimate
BOOTSTRAP_SAMPLES = 10000


class Bot:
    def __init__(self, name: str, cache_path: str):
        self.name = name  # Name of the bot: does not matter
        self.elo = INITIAL_ELO
        self.cache_path = cache_path
        self.num_matches = 0

        # Load cache if exists; else create an empty cache file
        if os.path.isfile(self.cache_path):
            self.cache = self.load_cache(cache_path)
        else:
            raise NotImplementedError(
                "Please precompute all bot responses using the get_model_qa.py script"
            )
            self.cache = []
            self.save_cache(self.cache, self.cache_path)

        self.model_id = self.cache[0]["model_id"]

    def load_cache(self, file_path: str) -> List[Dict]:
        return load_jsonl(file_path)

    def save_cache(self, data: List[Dict], file_path: str):
        save_jsonl(data, file_path)

    def generate_response(self, question: str) -> str:
        raise NotImplementedError(
            "For the time being the bot does not support generating responses. All should be precomputed using the get_model_qa.py script"
        )

    def get_response(self, challenge):
        # Check if the response is in cache
        response = next(
            (
                item
                for item in self.cache
                if item["question_id"] == challenge["question_id"]
            ),
            None,
        )
        self.num_matches += 1
        if response:
            return response
        else:
            # If not, generate and cache the response
            new_response = self.generate_response(challenge)
            self.cache.append(new_response)
            self.save_cache(self.cache, self.cache_path)
            return new_response


class Referee:
    def __init__(
        self,
        cache_path: str,
        reviewer_file: str,
        prompt_file: str,
        max_tokens: int,
        model: str,
    ):
        self.cache_path = cache_path
        self.reviewers = load_jsonl(reviewer_file)
        self.prompts = load_jsonl(prompt_file)
        self.max_tokens = max_tokens
        self.model = model

        # Load cache if exists; else create an empty cache file
        if os.path.isfile(self.cache_path):
            self.cache = self.load_cache(cache_path)
        else:
            self.cache = []
            self.save_cache(self.cache, self.cache_path)

        # Check that the cache contains no duplicate entries
        if not len(self.cache) == len(
            set(
                [
                    (item["question_id"], item["model1_id"], item["model2_id"])
                    for item in self.cache
                ]
            )
        ):
            import collections

            print(
                [
                    item
                    for item, count in collections.Counter(
                        [
                            (item["question_id"], item["model1_id"], item["model2_id"])
                            for item in self.cache
                        ]
                    ).items()
                    if count > 1
                ]
            )

            raise ValueError("Cache contains duplicate entries")

    def load_cache(self, file_path: str) -> List[Dict]:
        return load_jsonl(file_path)

    def save_cache(self, data: List[Dict], file_path: str):
        save_jsonl(data, file_path)

    def get_result(self, challenge, response1, response2):
        # If a cached result exists, return it
        cached_result = next(
            (
                item
                for item in self.cache
                if item["question_id"] == challenge["question_id"]
                and item["answer1_id"] == response1["answer_id"]
                and item["answer2_id"] == response2["answer_id"]
            ),
            None,
        )
        if cached_result:
            return cached_result

        review_json = get_review(
            self.reviewers,
            self.prompts,
            challenge,
            response1,
            response2,
            self.max_tokens,
            self.model,
        )

        # Save the result to cache
        self.update_cache(review_json)
        return review_json

    def update_cache(self, result: Dict):
        self.cache.append(result)
        self.save_cache(self.cache, self.cache_path)


class EloRanker:
    def __init__(
        self, bots: List[Bot], challenges_path: str, referee: Referee, verbose=False
    ):
        self.bots = bots
        self.challenges = load_jsonl(challenges_path)
        self.referee = referee
        self.verbose = verbose
        self.tournament = []

    def update_elo(self, bot1: Bot, bot2: Bot, bot1points: int) -> int:
        expected_bot1 = 1 / (
            1 + BASE ** ((bot2.elo - bot1.elo) / S - ETA / np.log(BASE))
        )
        expected_bot2 = 1 / (
            1 + BASE ** ((bot1.elo - bot2.elo) / S + ETA / np.log(BASE))
        )

        assert round(expected_bot1 + expected_bot2, 6) == 1, 1 - (
            expected_bot1 + expected_bot2
        )

        delta_elo = K * (bot1points - expected_bot1)
        bot1.elo = bot1.elo + delta_elo
        bot2.elo = bot2.elo - delta_elo

        return int(bot1.elo), int(bot2.elo)

    def run_challenge(self, bot1, bot2, challenge, verbose=False):
        if verbose:
            print(f"{bot1.name} vs {bot2.name}")
            print(f'Question: {challenge["text"]}')

        # Get the cached responses of the bots for the challenge
        response1 = bot1.get_response(challenge)
        response2 = bot2.get_response(challenge)

        if verbose:
            print(f"{bot1.name}: {response1['text']}")
            print(f"{bot2.name}: {response2['text']}")

        # Get the result of the matchup
        result = self.referee.get_result(challenge, response1, response2)

        if verbose:
            print(f"Referee comments: {result['text']}")
            print(f"Winner: {[bot1.name, bot2.name, 'draw'][result['score']-1]}")

        score = result["score"]

        old_elo1 = bot1.elo
        old_elo2 = bot2.elo

        if score == 1:
            self.update_elo(bot1, bot2, 1)
        elif score == 2:
            self.update_elo(bot1, bot2, 0)
        elif score == 3:
            self.update_elo(bot1, bot2, 0.5)

        # Add the match details to the tournament history
        self.tournament.append(
            {
                "bot1": bot1.name,
                "bot2": bot2.name,
                "prematch_elo1": old_elo1,
                "prematch_elo2": old_elo2,
                "question": challenge["text"],
                "question_id": challenge["question_id"],
                "response1": response1["text"],
                "response2": response2["text"],
                "referee_comments": result["text"],
                "score": result["score"],
                "postmatch_elo1": bot1.elo,
                "postmatch_elo2": bot2.elo,
            }
        )

    def run_tournament(self, num_matchups):
        cached_results = self.referee.cache
        random.shuffle(cached_results)

        max_matchups = len(self.bots) * (len(self.bots) - 1) * len(self.challenges)

        if num_matchups > max_matchups:
            num_matchups = max_matchups

        if self.verbose:
            print(
                f"Running a tournament of {num_matchups} matchups out of {max_matchups} total possible"
            )

        matchup = 0
        # Run cached result first
        for cached_result in cached_results:
            if matchup >= num_matchups:
                break
            bot1 = next(
                (
                    item
                    for item in self.bots
                    if item.model_id == cached_result["model1_id"]
                ),
                None,
            )
            bot2 = next(
                (
                    item
                    for item in self.bots
                    if item.model_id == cached_result["model2_id"]
                ),
                None,
            )
            challenge = next(
                (
                    item
                    for item in self.challenges
                    if item["question_id"] == cached_result["question_id"]
                ),
                None,
            )
            if not bot1 or not bot2 or not challenge:
                continue
            else:
                if self.verbose:
                    print(f"Matchup {matchup + 1}/{num_matchups}:")
                self.run_challenge(bot1, bot2, challenge, self.verbose)
                matchup += 1

        while matchup < num_matchups:
            # Select bot with lowest number of matches
            bot_min_matches = min(self.bots, key=lambda x: x.num_matches)

            # Select another bot randomly
            bot_random = random.choice(
                [bot for bot in self.bots if bot != bot_min_matches]
            )

            # Randomly assign bot_min_matches and bot_random as bot1 and bot2
            bot1, bot2 = random.sample([bot_min_matches, bot_random], 2)

            # Select a random challenge
            challenge = random.choice(self.challenges)

            # Check if the matchup has already been played and if not run it
            cached_result = next(
                (
                    item
                    for item in cached_results
                    if item["question_id"] == challenge["question_id"]
                    and item["model1_id"] == bot1.model_id
                    and item["model2_id"] == bot2.model_id
                ),
                None,
            )
            if not cached_result:
                if self.verbose:
                    print(f"Matchup {matchup + 1}/{num_matchups}:")
                self.run_challenge(bot1, bot2, challenge, self.verbose)
                matchup += 1

        # Check total ELO
        total_elo = 0
        for bot in self.bots:
            total_elo += bot.elo
        assert (
            round(total_elo, 6) == len(self.bots) * INITIAL_ELO
        ), f"Total ELO: {total_elo}, Expected: {len(self.bots) * INITIAL_ELO}"

        # Compute the bootstrapped ELO
        bootstrap_elos = self.compute_bootstrap_elos(BOOTSTRAP_SAMPLES)

        # Construct the standings
        sorted_bots = sorted(self.bots, key=lambda bot: bot.elo, reverse=True)
        self.standings = pd.DataFrame(
            [
                {"model": bot.name, "elo": bot.elo, "num_matches": bot.num_matches}
                for bot in sorted_bots
            ]
        ).merge(bootstrap_elos)

        # Print the standings
        if self.verbose:
            print("=== Final Standings ===")
            pd.options.display.float_format = "{:.3f}".format
            print(
                self.standings[
                    ["model", "elo", "error_y_plus", "error_y_minus", "num_matches"]
                ]
            )
            print(
                f"BETA: {BETA}, K: {K}, INITIAL_ELO: {INITIAL_ELO}, BASE: {BASE}, S: {S}"
            )

        assert (
            round(total_elo, 6) == len(self.bots) * INITIAL_ELO
        ), f"Total ELO: {total_elo}, Expected: {len(self.bots) * INITIAL_ELO}"

    def output_standings(self, output_file):
        """Convert bots to JSON format and save to a file."""

        standings_json = self.standings.to_dict(orient="records")

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
            "rankings": standings_json,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def output_tournament(self, output_file):
        """Convert bots to JSON format and save to a file."""
        save_jsonl(self.tournament, output_file)

    def compute_bootstrap_elos(self, num_samples):
        """Compute bootstrap confidence intervals"""
        rows = []
        for i in tqdm(range(num_samples), desc="bootstrap"):
            bootstrapped_tournament = random.choices(
                self.tournament, k=len(self.tournament)
            )
            bootstrapped_bots = [copy.deepcopy(bot) for bot in self.bots]
            for bot in bootstrapped_bots:
                bot.elo = INITIAL_ELO

            for row in bootstrapped_tournament:
                bot1 = next(
                    (item for item in bootstrapped_bots if item.name == row["bot1"])
                )
                bot2 = next(
                    (item for item in bootstrapped_bots if item.name == row["bot2"])
                )
                score = row["score"]
                if score == 1:
                    self.update_elo(bot1, bot2, 1)
                elif score == 2:
                    self.update_elo(bot1, bot2, 0)
                elif score == 3:
                    self.update_elo(bot1, bot2, 0.5)
                else:
                    raise ValueError(f"Invalid score: {score}")

            rows.append({bot.name: bot.elo for bot in bootstrapped_bots})
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
            .rename(columns={"index": "model"})
            .sort_values("median", ascending=False)
        )
        bootstrap_elo["error_y_plus"] = bootstrap_elo["upper"] - bootstrap_elo["median"]
        bootstrap_elo["error_y_minus"] = (
            bootstrap_elo["median"] - bootstrap_elo["lower"]
        )
        return bootstrap_elo


if __name__ == "__main__":
    bots = [
        Bot("GPT3", "answers/rakuda_koukou_v0/gpt3.jsonl"),
        Bot("Rinna 3.6B - PPO", "answers/rakuda_koukou_v0/rinna-ppo.jsonl"),
        # Bot("Rinna 3.6B - SFTv2", "answers/rakuda_koukou_v0/rinna-sft.jsonl"),
        Bot("Rinna 3.6B", "answers/rakuda_koukou_v0/rinna.jsonl"),
        Bot("Open Calm 7B - Stormy", "answers/rakuda_koukou_v0/stormy.jsonl"),
        Bot("Open Calm 7B", "answers/rakuda_koukou_v0/calm.jsonl"),
    ]

    referee = Referee(
        "matchups/rakuda_koukou_v0.jsonl",
        "prompts/rakuda_reviewer.jsonl",
        "prompts/rakuda_prompt_threeclass.jsonl",
        max_tokens=1024,
        model="gpt-3.5-turbo-0301",
    )

    ranker = EloRanker(bots, "questions/rakuda_koukou_v0.jsonl", referee, verbose=True)
    ranker.run_tournament(800)

    ranker.output_standings("tournaments/rakuda_koukou_v0_tournament_result.json")
    ranker.output_tournament("tournaments/rakuda_koukou_v0_tournament.jsonl")

    registry = StandingsRegistry("./registry/registry.jsonl")
    registry.register("tournaments/rakuda_koukou_v0_tournament_result.json")
