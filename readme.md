# jrank: Ranking Japanese LLMs

| [**Ranking**](https://yuzuai.jp/benchmark) |
[**Blog**](https://yuzuai.jp/blog/rakuda) |
[**Discord**](https://discord.com/invite/bHB9e2rq2r) |


This repository supports YuzuAI's [Rakuda leaderboard](https://yuzuai.jp/benchmark) of Japanese LLMs, which is a Japanese-focused version of LMSYS' [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

## Usage 

Rakuda follows the same API as LLM Judge. First start with a question list you wish to compare the models on. These questions can be multi-turn. The default Rakuda question list is `jrank/data/rakuda_v2/questions.jsonl` ([HF](https://huggingface.co/datasets/yuzuai/rakuda-questions)).

Then generate model answers to these questions using `jrank/gen_model_answer.py`:

```bash
python gen_model_answer.py --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json
```

For API models, use `gen_api_answer.py` instead.

After generating model answers, generate judgements of these answers using `gen_judgement.py`. 

```bash
python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model claude-2 --n 2000
```

The mode option determines what kind of judgements are performed. The default for rakuda is `pairwise-n`, in which model answers are compared pairwise until `n` judgements have been reached.

Finally, use these judgements to create a model ranking. 
```bash
python make_ranking.py --bench-name rakuda_v2 --judge-model claude-2 --mode pairwise --compute mle --make-charts --bootstrap-n 500 --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct  --advanced-charts
```


Once your model has answered the Rakuda questions, use `jrank/matchmaker.py` to send pairs of answers from your model and other ranked models to an external reviewer, by default GPT-4 (`jrank/reviewer_gpt.py`). The reviewer will evaluate which answer is better and store its results in `jrank/reviews`. 

Finally run the analysis notebook `jrank/bradley-terry.ipynb` which will perform a Bayesian analysis of the reviews and infer the strength of each model. The ranking will be output to `jrank/rankings/`.