""" Overplot two sets of rankings
Usage: 
python overplot_rankings.py data/rakuda_v2/rankings/claude-2_mle.json data/rakuda_v2/rankings/gpt-4_mle.json --chart_dir ./ --advanced-charts --label1 "Claude-2" --label2 "GPT-4" --legend-title "Referee"
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
    parser.add_argument("--label1", type=str, default=None, help="Label for the 1st file")
    parser.add_argument("ranking2", type=str, help="Path to the second ranking file.")
    parser.add_argument("--label2", type=str, default=None, help="Label for the 2nd file")
    parser.add_argument("--chart_dir", type=str, default=None, help="Directory to save the chart.")
    parser.add_argument("--legend-title", type=str, default=None, help="Label for the 2nd file")
    parser.add_argument(
        "--advanced-charts",
        action="store_true",
        help="Whether to output charts",
    )
    args = parser.parse_args()

    # Load rankings
    ranking1 = load_ranking(args.ranking1)
    ranking2 = load_ranking(args.ranking2)

    # Determine the order from ranking1
    order = ranking1.sort_values(by="median")["model_id"]

    # Plot strengths
    #plt.figure(figsize=(10, 6))
    #plot_strengths(ranking1, color="blue", label="Ranking 1", order=order, chart_dir='./', advanced_charts=args.advanced_charts)

    title = 'GPT-4 and Claude-2 Agree as Referees'
    fig, ax = plot_strengths(ranking1, color='blue', label=args.label1, order=order, advanced_charts=args.advanced_charts, show_licensing=False, title=title, subtitle=None)
    plot_strengths(ranking2, color='red', label=args.label2, order=order,chart_dir='./overplot',  advanced_charts=args.advanced_charts, figax=(fig,ax), legend_title=args.legend_title, show_licensing=False, title=title, subtitle=None)

    # Set other chart properties
    # plt.legend()
    
    # if args.chart_dir:
    #     plt.savefig(args.chart_dir + "combined_ranking.png")
    # else:
    #     plt.show()
