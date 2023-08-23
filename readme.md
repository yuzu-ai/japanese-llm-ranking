# jrank: Ranking Japanese LLMs

| [**Ranking**](https://yuzuai.jp/benchmark) |
[**Blog**](https://yuzuai.jp/blog/rakuda) |
[**Discord**](https://discord.com/invite/bHB9e2rq2r) |


This repository supports YuzuAI's [Rakuda leaderboard](https://yuzuai.jp/benchmark) of Japanese LLMs, which is a Japanese-focused analogue of LMSYS' [Vicuna eval](https://lmsys.org/vicuna_eval/).

## Adding a model to Rakuda

To add a model to the Rakuda leaderboard, first have the model answer the Rakuda questions. These questions are stored in `jrank/questions/` and on [hugging-face](https://huggingface.co/datasets/yuzuai/rakuda-questions).

If you wish, you can use the `jrank/get_model_qa.py` script to generate these answers. This script loads and runs models using model adapters from [FastChat](https://github.com/lm-sys/FastChat). Custom adapters can also be implemented in `jrank/adapters.py`, and scripts showing exactly the commands used to run existing models on the leaderboard are stored in `jrank/jobs/`. If your model is only accessible via an API, consult `jrank/get_gpt_qa.py`.

Once your model has answered the Rakuda questions, use `jrank/matchmaker.py` to feed send pairs of answers from your model and other ranked models to an external reviewer, by default GPT-4 (`jrank/reviewer_gpt.py`). The reviewer will evaluate which answer is better and store its results in `jrank/reviews`. 

Finally run the analysis notebook `jrank/bradley-terry.ipynb` which will perform a Bayesian analysis of the reviews and infer the strength of each model. The ranking will be output to `jrank/rankings/`.