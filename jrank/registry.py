import json
import os
from datetime import datetime

from utils import load_jsonl, save_jsonl



class StandingsRegistry:
    def __init__(self, registry_file):
        self.registry_file = registry_file
        if os.path.exists(self.registry_file):
            self.registry_data = load_jsonl(self.registry_file)
        else:
            self.registry_data = []

    def register(self, standings_file):
        with open(standings_file, "r") as f:
            standings_data = json.load(f)

        if standings_data not in self.registry_data:
            self.registry_data.append(standings_data)
            save_jsonl(self.registry_data, self.registry_file)
        else:
            print("This experiment has already been registered.")

    def get_registry(self):
        return self.registry_data
