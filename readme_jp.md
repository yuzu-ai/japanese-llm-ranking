# jrank: 日本語大規模言語モデルの評価ランキング
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/yuzu-ai/japanese-llm-ranking/blob/main/readme.md)
[![jp](https://img.shields.io/badge/lang-jp-yellow.svg)](https://github.com/yuzu-ai/japanese-llm-ranking/blob/main/readme_jp.md)

| [**Ranking**](https://yuzuai.jp/benchmark) |
[**Blog**](https://yuzuai.jp/blog/rakuda) |
[**Discord**](https://discord.com/invite/bHB9e2rq2r) |


LMSYS' [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)を採用した、日本語大規模言語モデルのベンチマーク(通称: Rakuda)である[Rakuda leaderboard](https://yuzuai.jp/benchmark)を管理するレポジトリです。

## 利用方法

RakudaはLLM Judgeと同様に同じAPIを使用しています。始めに、モデル同士を比較させたい質問リストを用意します。(質問は マルチターンも可能)。Rakudaにおいて、defaultで使用している質問リストは`jrank/data/rakuda_v2/questions.jsonl` ([HF](https://huggingface.co/datasets/yuzuai/rakuda-questions))から確認できます。
これらの質問に対して、`jrank/gen_model_answer.py`を実行することでモデルによる返答を生成します：

```bash
python3 gen_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json
```

APIモデルを使用する場合には、代わりに、`gen_api_answer.py`を使用してモデルの返答を生成します。

次に、`gen_judgement.py`を実行することでモデルによって生成された返答の判定を行います:

```bash
python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model claude-2 --n 2000
```

`mode` optionがどのような判定を行うかを決定します。Rakudaではデフォルトで、`n`個の判定に到達するまでに生成された返答をペアごとに比較する `pairwise-n`を採用しています。

最後に、下された判定に対してBradley-Terryモデルをフィッティングすることで、評価ランキングを作成します:

```bash
python make_ranking.py --bench-name rakuda_v2 --judge-model claude-2 --mode pairwise --compute mle --make-charts --bootstrap-n 500 --plot-skip-list rinna-3.6b-sft super-trin elyza-7b-instruct
```

##　New Method (Work in Progress)

上記のステップを自動で実行し、ランキング結果を生成する新しい機能を追加しました。

### Steps

1. `config.json` in を`jrank`フォルダに作成
2. 例として用意された`config.json.example`の内容を`config.json`にcopy
3. 設定を必要に応じて適宜変更し、`local_models` または `api_models` list が空の場合はスキップされます。
4. ローカル環境を準備
5. `pip install -r requirements.txt`　※
6. `cd jrank`
7. `python3 streamline.py`　を実行 (Config fileに従って評価ランキングを生成）

※ `from openai import OpenAI` エラーが表示された際には次のようにopenaiライブラリをupgradeしてください

```bash
pip install openai --upgrade
```

## Reference

localモデルによる答えを生成:

```Bash
python3 gen_model_answer.py --bench_name rakuda_v2 --model-path EleutherAI/pythia-70m  --model-id pythia-70m --conv_template ./templates/yuzulm.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-alpha-7b-v2 --model-id stablelm-alpha-7b-v2 --conv_template ./templates/japanese-stablelm.json --top_p 0.95 --temperature 1

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-gamma-7b --model-id stablelm-gamma-7b --conv_template ./templates/japanese-stablelm.json --repetition_penalty 1.05 --max_new_tokens 512 --top_p 0.95

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-chat --model-id youri-7b-chat --conv_template ./templates/youri-chat.json --repetition_penalty 1.05 --num_beams 5

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-instruction --model-id youri-7b-instruction --conv_template ./templates/youri-instruction.json --repetition_penalty 1.05

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 --model-id llm-jp-13b-instruct --conv_template ./templates/llm-jp-instruct.json --repetition_penalty 1.05

```

判定:

```bash
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all|pairwise-n] --judge-model [gpt-4|gpt-3.5-turbo|claude-2] --n ["all"|int]

python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all|pairwise-n] --judge-model [gpt-4|gpt-3.5-turbo|claude-2] --n ["all"|int]

python gen_judgment.py --bench-name rakuda_v2_test --model-list claude-2 gpt-3.5-turbo line-1.7b --parallel 1 --mode pairwise-n --judge-model claude-2 --n 2

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model claude-2 --n 2000

python gen_judgment.py --bench-name rakuda_v2 --model-list chatntq-7b-jpntuned claude-2 gpt-3.5-turbo-0301-20230614 gpt-4-20230713 elyza-7b-fast-instruct elyza-7b-instruct jslm7b-instruct-alpha line-3.6b-sft rinna-3.6b-ppo rinna-3.6b-sft rwkv-world-jp-v1 stablebeluga2 weblab-10b-instruction-sft super-trin --parallel 2 --mode pairwise-n --judge-model gpt-4 --n 1400
```
