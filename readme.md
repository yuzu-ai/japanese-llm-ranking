# jrank: Ranking Japanese LLMs 

| [**Ranking**](https://yuzuai.jp/benchmark) |
[**Blog**](https://yuzuai.jp/blog/rakuda) |
[**Discord**](https://discord.com/invite/bHB9e2rq2r) |


This repository supports YuzuAI's [Rakuda leaderboard](https://yuzuai.jp/benchmark) of Japanese LLMs, which is a Japanese-focused analogue of LMSYS' [Vicuna eval](https://lmsys.org/vicuna_eval/).

This repo has the following components:

- `jrank/questions/`: Stores question lists, such as our [Rakuda questions](https://huggingface.co/datasets/yuzuai/rakuda-questions).

- `jrank/get_model_qa.py` and `jrank/get_gpt_qa.py`: Gets hugging-face or GPT models to answer a list of questions. Uses FastChat as a common abstraction layer [FastChat](https://github.com/lm-sys/FastChat). Answers are stored in `jrank/answers/` . Jobs to launch these scripts are included in `jrank/jobs/`.

- `jrank/matchmaker.py`: Gets pairs of answers to questions and sends them to a reviewer (`jrank/reviewer_gpt.py`) to evaluate which answer is better. Reviews are cached in `jrank/reviews` and the full set of matches is stored in `jrank/tournaments`.

- `jrank/bradley-terry.ipynb`: A notebook that performs a Bayesian fit of the Bradley-Terry model to the match data to estimate the strength of each model. Rankings are output to `jrank/rankings/`.