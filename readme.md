# jrank: Ranking Japanese LLMs
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/yuzu-ai/japanese-llm-ranking/blob/main/readme.md)
[![jp](https://img.shields.io/badge/lang-jp-yellow.svg)](https://github.com/yuzu-ai/japanese-llm-ranking/blob/main/readme_jp.md)

| [**Ranking**](https://yuzuai.jp/benchmark) |
[**Blog**](https://yuzuai.jp/blog/rakuda) |
[**Discord**](https://discord.com/invite/bHB9e2rq2r) |


This repository supports YuzuAI's [Rakuda leaderboard](https://yuzuai.jp/benchmark) of Japanese LLMs, which is a Japanese-focused version of LMSYS' [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

## Usage

Rakuda follows the same API as LLM Judge. First start with a question list you wish to compare the models on. These questions can be multi-turn. The default Rakuda question list is `jrank/data/rakuda_v2/questions.jsonl` ([HF](https://huggingface.co/datasets/yuzuai/rakuda-questions)).

Then generate model answers to these questions using `jrank/gen_local_model_answer.py`:

```bash
python3 gen_local_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json
```

For API models, use `gen_api_answer.py` instead.

After generating model answers, generate judgements of these answers using `gen_judgement.py`.

```bash
python gen_judgment.py --bench_name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model claude-2 --n 2000
```

The mode option determines what kind of judgements are performed. The default for rakuda is `pairwise-n`, in which model answers are compared pairwise until `n` judgements have been reached.

Finally, fit a Bradley-Terry model to these judgements to create a model ranking.
```bash
python make_ranking.py --bench-name rakuda_v2 --judge-model claude-2 --mode pairwise --compute mle --make-charts --bootstrap-n 500 --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct
```

## New Method (Work in Progress)

In order to ease the use of Rakuda, we have created a new method to generate the model ranking result.

### Steps

1. Create a new `config.json` in jrank folder
2. Copy `config.json.example` content to `config.json`
3. Modify the content as you see fit; if `local_models` or `api_models` list is empty, then that part will be skipped
4. Start your local environment
5. `pip install -r requirements.txt` *
6. `cd jrank`
7. run `python3 streamline.py` it will run following the config file and generate a result file

* if encounter `from openai import OpenAI` error, be sure to upgrade openai package
```bash
pip install openai --upgrade
```

More updates will be coming soon.

## Reference

Generate answers with local models.

```Bash
python3 gen_model_answer.py --bench_name rakuda_v2 --model-path EleutherAI/pythia-70m  --model-id pythia-70m --conv_template ./templates/yuzulm.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-alpha-7b-v2 --model-id stablelm-alpha-7b-v2 --conv_template ./templates/japanese-stablelm.json --top_p 0.95 --temperature 1

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-gamma-7b --model-id stablelm-gamma-7b --conv_template ./templates/japanese-stablelm.json --repetition_penalty 1.05 --max_new_tokens 512 --top_p 0.95

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-chat --model-id youri-7b-chat --conv_template ./templates/youri-chat.json --repetition_penalty 1.05 --num_beams 5

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-instruction --model-id youri-7b-instruction --conv_template ./templates/youri-instruction.json --repetition_penalty 1.05

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 --model-id llm-jp-13b-instruct --conv_template ./templates/llm-jp-instruct.json --repetition_penalty 1.05

```

Generate judgement result.

Usage:

```bash
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all|pairwise-n] --judge-model [gpt-4|gpt-3.5-turbo|claude-2] --n ["all"|int]

python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all|pairwise-n] --judge-model [gpt-4|gpt-3.5-turbo|claude-2] --n ["all"|int]

python gen_judgment.py --bench-name rakuda_v2_test --model-list claude-2 gpt-3.5-turbo line-1.7b --parallel 1 --mode pairwise-n --judge-model claude-2 --n 2

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model claude-2 --n 2000

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model gpt-4 --n 1400
```
