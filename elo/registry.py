import json
import os
from utils import save_jsonl, load_jsonl

class StandingsRegistry:
    def __init__(self, registry_file):
        self.registry_file = registry_file
        if os.path.exists(self.registry_file):
            self.registry_data = load_jsonl(self.registry_file)
        else:
            self.registry_data = []
    
    def register(self, standings_file):
        with open(standings_file, 'r') as f:
            standings_data = json.load(f)

        if standings_data not in self.registry_data:
            self.registry_data.append(standings_data)            
            save_jsonl(self.registry_data, self.registry_file)
        else:
            print('This experiment has already been registered.')

    def get_registry(self):
        return self.registry_data
    
    def convert_to_markdown(self, template_file, markdown_file):
        data = self.registry_data[-1]

        rankings = sorted(data["rankings"], key=lambda x: x['elo'], reverse=True)
        table = "| Rank # | Model | Elo |\n| --- | --- | --- |\n"
        for i, rank in enumerate(rankings):
            table += f"| {i+1} | {rank['model']} | {int(rank['elo'])} |\n"
        
        with open(template_file, "r") as f:
            lines = f.readlines()

        lines.pop()
        lines.append(table)

        with open(markdown_file, "w") as file:
            file.writelines(lines)

        print("Converted elo ranking to markdown table")
