""" Overplot two sets of rankings
Usage: 
python overplot_rankings.py data/rakuda_v2/rankings/claude-2_mle.json data/rakuda_v2/rankings/gpt-4_mle.json --chart_dir .
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from make_ranking import load_ranking, plot_strengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overplot two model rankings.")
    parser.add_argument("ranking1", type=str, help="Path to the first ranking file.")
    parser.add_argument("ranking2", type=str, help="Path to the second ranking file.")
    parser.add_argument("--chart_dir", type=str, default=None, help="Directory to save the chart.")
    args = parser.parse_args()

    # Load rankings
    ranking1 = load_ranking(args.ranking1)
    ranking2 = load_ranking(args.ranking2)

    # Determine the order from ranking1
    order = ranking1.sort_values(by="median")["model_id"]

    # Plot strengths
    plt.figure(figsize=(10, 6))
    plot_strengths(ranking1, color="blue", label="Ranking 1", order=order)
    plot_strengths(ranking2, color="red", label="Ranking 2", order=order)

    # Set other chart properties
    plt.legend()
    
    if args.chart_dir:
        plt.savefig(args.chart_dir + "combined_ranking.png")
    else:
        plt.show()
