import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from typing import Dict, List, Tuple

from reviewer_gpt import get_review
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from random import shuffle, choice

class Bot:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path

        try:
            cache = load_jsonl(cache_path)
        except FileNotFoundError:
            raise ValueError(
                "Cache file does not exist. Please run get_model_qa.py to precompute responses."
            )

        self.cache = {item["question_id"]: item for item in cache}
        self.model_id = list(self.cache.values())[0]["model_id"]

    def get_response(self, question_id: int) -> Dict:
        try:
            return self.cache[question_id]
        except KeyError:
            raise ValueError(
                f"No precomputed response for question ID {question_id}. Please run get_model_qa.py."
            )


class Referee:
    def __init__(
        self,
        cache_path: str,
        reviewer_path: str,
        prompt_path: str,
    ):
        self.cache_path = cache_path
        self.reviewer_path = reviewer_path
        self.prompt_path = prompt_path

        self.reviewers = load_jsonl(reviewer_path)
        self.prompts = load_jsonl(prompt_path)

        # Load cache if exists; else create an empty cache file
        self.cache = self.load_cache() if os.path.isfile(self.cache_path) else {}

        print(len(self.cache))

    def load_cache(self) -> Dict:
        cache = load_jsonl(self.cache_path)

        return {
            (item["question_id"], item["answer1_id"], item["answer2_id"]): item
            for item in cache
        }

    def add_to_cache(self, data: Dict):
        key = (data["question_id"], data["answer1_id"], data["answer2_id"])

        if key in self.cache:
            raise ValueError("Adding duplicate entry to cache")

        self.cache[key] = data
        save_jsonl([data], self.cache_path, write_flag="a")

    def get_result(self, question: Dict, response1: Dict, response2: Dict) -> Dict:
        key = (question["question_id"], response1["answer_id"], response2["answer_id"])
        if key not in self.cache:
            review_json = get_review(
                self.reviewers,
                self.prompts,
                question,
                response1,
                response2
            )
            self.add_to_cache(review_json)

        return self.cache[key]


class MatchMaker:
    def __init__(
        self, bots: List[Bot], questions_path: str, referee: Referee, verbose=False
    ):
        self.bots = {bot.model_id: bot for bot in bots}
        self.questions_path = questions_path
        self.questions = load_jsonl(questions_path)
        self.referee = referee
        self.verbose = verbose
        self.matches = []

    def _get_result(self, match) -> dict:
        bot1_id, bot2_id, question_id = match

        bot1 = self.bots[bot1_id]
        bot2 = self.bots[bot2_id]

        # Get the question
        question = next(q for q in self.questions if q["question_id"] == question_id)

        # Get the cached responses of the bots for the question
        response1 = bot1.get_response(question["question_id"])
        response2 = bot2.get_response(question["question_id"])

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
            "model1_id": bot1.model_id,
            "model2_id": bot2.model_id,
            "question": question["text"],
            "question_id": question["question_id"],
            "response1": response1["text"],
            "response2": response2["text"],
            "referee_comments": result["text"],
            "score": result["score"],
        }

    def _prepare_matches(self, num_matchups: int) -> List[Tuple[str, str, dict]]:
        # Reuse cached results if possible
        cached_results = self.referee.cache.values()

        matches = []
        for result in cached_results:
            if len(matches) == num_matchups:
                return matches
            bot1_id = next(
                (key for key in self.bots.keys() if key == result["model1_id"]), None
            )
            bot2_id = next(
                (key for key in self.bots.keys() if key == result["model2_id"]), None
            )
            question_id = next(
                q["question_id"]
                for q in self.questions
                if q["question_id"] == result["question_id"]
            )
            if bot1_id and bot2_id and question_id:
                matches.append((bot1_id, bot2_id, question_id))

        print(f"Number of matches imported from cache {len(matches)}")

        # If there are not enough cached results, create new matches
        bot_ids = list(self.bots.keys())
        all_possible_pairs = list(permutations(bot_ids, 2))

        all_possible_new_matches = []
        for bot1_id, bot2_id in all_possible_pairs:
            for q in self.questions:
                if (bot1_id, bot2_id, q["question_id"]) not in matches:
                    all_possible_new_matches.append((bot1_id, bot2_id, q["question_id"]))

        shuffle(all_possible_new_matches)

        # Create a dictionary to count the number of matches for each bot pair
        bot_match_counts = {bot_id: 0 for bot_id in bot_ids}

        # Update the dictionary with the number of existing matches
        for match in matches:
            bot_match_counts[match[0]] += 1
            bot_match_counts[match[1]] += 1
            

        while len(matches) < num_matchups and all_possible_new_matches:
            # Sort bots by the number of matches they have participated in
            sorted_bots = sorted(bot_match_counts.items(), key=lambda x: x[1])
            
            # Select a bot with fewest matches
            selected_bot = choice([bot for bot, count in sorted_bots if count == sorted_bots[0][1]])

            # Select a match involving the selected bot
            selected_matches = [match for match in all_possible_new_matches if selected_bot in match]
                      
            # Select a match and add it to matches
            possible_match = choice(selected_matches)
            if possible_match not in matches:
                matches.append(possible_match)
                all_possible_new_matches.remove(possible_match)

                # Update the number of matches for each bot
                bot_match_counts[possible_match[0]] += 1
                bot_match_counts[possible_match[1]] += 1
            else:
                raise RuntimeError("Match already exists in matches despite being drawn from all_possible_new_matches")

        return matches

    def run_matches(self, num_matchups: int):
        matches = self._prepare_matches(num_matchups)

        # Create a pool of workers to process the matches
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                tqdm(executor.map(self._get_result, matches), total=len(matches))
            )

        self.matches = results

    def output_matches(self, output_path):
        """Convert a list of matches to JSON format and save to a file."""

        metadata = {
            "questions_path": self.questions_path,
            "reviewer_path": self.referee.reviewer_path,
            "prompt_path": self.referee.prompt_path,
        }

        model_metadata = {
            bot.model_id: {"model_id": bot.model_id, "path": bot.cache_path}
            for bot in self.bots.values()
        }

        output = {
            "model_metadata": model_metadata,
            "metadata": metadata,
            "matches": self.matches,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    bots = [
        Bot("answers/rakuda_v1/gpt4.jsonl"),
        Bot("answers/rakuda_v1/gpt3.jsonl"),
        Bot("answers/rakuda_v1/rinna-ppo.jsonl"),
        Bot("answers/rakuda_v1/rinna-sft.jsonl"),
        #Bot("answers/rakuda_v1/rinna.jsonl"),
        Bot("answers/rakuda_v1/stormy.jsonl"),
        Bot("answers/rakuda_v1/calm.jsonl"),
        #Bot("answers/rakuda_v1/rwkv.jsonl"),
        Bot("answers/rakuda_v1/rwkv-jp-v1.jsonl"),
        Bot("answers/rakuda_v1/stablebeluga2.jsonl"),
        Bot("answers/rakuda_v1/super-trin.jsonl"),
        Bot("answers/rakuda_v1/japanese-stablelm-instruct-alpha.jsonl"),
    ]

    referee = Referee(
        "reviews/rakuda_v1_gpt4.jsonl",
        "prompts/gpt4_reviewer.jsonl",
        "prompts/rakuda_prompt.jsonl",
    )

    matchmaker = MatchMaker(bots, "questions/rakuda_v1.jsonl", referee, verbose=False)
    matchmaker.run_matches(788)
    matchmaker.output_matches("tournaments/rakuda_v1_8-10.jsonl")
