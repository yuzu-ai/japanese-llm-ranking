"""convert json format to csv table format"""

# load json file
import json
import pandas as pd

ranking_file = "./data/rakuda_v2/rankings/claude-2_mle.json"
with open(ranking_file, "r", encoding="utf-8") as f:
  data = json.load(f)
  
# print(data["ranking"])
  # Specify the output CSV file path
output_file = "ranking.csv"

ranking = data["ranking"]

# Extract the field names from the first item in the data list
keys = ranking[0].keys()

df = pd.DataFrame(ranking)

df.to_csv(output_file, index=False)
