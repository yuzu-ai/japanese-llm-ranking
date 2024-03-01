"""Generate answers with local models.

Usage:

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path EleutherAI/pythia-70m  --model-id pythia-70m --conv_template ./templates/yuzulm.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-alpha-7b-v2 --model-id stablelm-alpha-7b-v2 --conv_template ./templates/japanese-stablelm.json --top_p 0.95 --temperature 1

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-gamma-7b --model-id stablelm-gamma-7b --conv_template ./templates/japanese-stablelm.json --repetition_penalty 1.05 --max_new_tokens 512 --top_p 0.95

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-chat --model-id youri-7b-chat --conv_template ./templates/youri-chat.json --repetition_penalty 1.05 --num_beams 5

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-instruction --model-id youri-7b-instruction --conv_template ./templates/youri-instruction.json --repetition_penalty 1.05

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 --model-id llm-jp-13b-instruct --conv_template ./templates/llm-jp-instruct.json --repetition_penalty 1.05

"""

import json
import os
import sys
from typing import Optional
import time

import shortuuid
import torch

from common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.model.model_adapter import model_adapters
from fastchat.conversation import Conversation, SeparatorStyle

from adapters import (
    FastTokenizerAvailableBaseAdapter,
    JapaneseStableLMAlphaAdapter,
    JapaneseStableLMAlphaAdapterv2,
    RwkvWorldAdapter,
)

from fire import Fire
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList, StoppingCriteria

# Hack the fastchat model adapters
model_adapters[-1] = FastTokenizerAvailableBaseAdapter()
model_adapters.insert(0, JapaneseStableLMAlphaAdapter())
model_adapters.insert(1, JapaneseStableLMAlphaAdapterv2())

for i in range(len(model_adapters)):
    if "Rwkv" in type(model_adapters[i]).__name__:
        model_adapters[i] = RwkvWorldAdapter()


# Helper that generate a fastchat conversation from a template file
def get_conv_from_template_path(template_path):
    with open(template_path, "r") as file:
        config = json.load(file)

    # Convert sep_style from string to SeparatorStyle enum
    if "sep_style" in config:
        config["sep_style"] = SeparatorStyle[config["sep_style"]]

    # Start a conversation
    if "messages" not in config:
        config["messages"] = []

    return Conversation(**config)


@torch.inference_mode()
def get_model_answers(
    model_path: str,
    model_id,
    model_variant: str = None,
    bench_name: str = "rakuda_v2",
    answer_file: str = None,
    # debug_params
    question_begin: Optional[int] = None,
    question_end: Optional[int] = None,
    # model parameters
    lora_path: Optional[str] = None,
    conv_template: Optional[str] = None,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,  # only relevant for numgpus > 1
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
    # generation parameters
    temperature: Optional[float] = None,
    top_p: float = 0.9,
    top_k: float = 0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    max_tokens: Optional[int] = 1024,
    num_choices: int = 1,
    # generate the answers (set to False for debugging prompts)
    generate_answers: bool = True,
):
    question_file = f"data/{bench_name}/questions.jsonl"
    if not answer_file:
        answer_file = f"data/{bench_name}/answers/{model_id}.jsonl"

    questions = load_questions(question_file, question_begin, question_end)

    if not conv_template:
        conv_template = model_id

    # Load the model
    if generate_answers:
        model, tokenizer = load_model(
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            debug=debug,
            dtype="auto",
        )
        model.config.use_cache = False
        model.eval()

        # if lora_path is not None:
        #     model = PeftModel.from_pretrained(
        #         model, lora_path, torch_dtype=torch.float16
        #     )

        if max_tokens is None:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(model.config, attr):
                    max_tokens = getattr(model.config, attr)
                    print(f"{attr}: {max_tokens}")
            print(f"=> Max_tokens: {max_tokens}")
            if max_tokens is None:
                raise ValueError(
                    "max_tokens must be specified if model.config doesn't have an attribute n_positions, max_position_embeddings, or n_ctx"
                )

        if model_id == "matsuo-lab/weblab-10b-instruction-sft":
            tokenizer.pad_token_id = 1
            tokenizer.eos_token_id = 0
            tokenizer.bos_token_id = tokenizer.pad_token_id

        if "RWKV" not in model_path:
            print(
                f"pad_token_id={tokenizer.pad_token_id}, bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}"
            )

    for question in tqdm(questions):
        if not temperature:
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7

        if os.path.exists(conv_template):
            conv = get_conv_from_template_path(conv_template)
        else:
            conv = get_conversation_template(conv_template)

        stopping_criteria = None

        if generate_answers and conv.stop_str and "RWKV" not in model_path:

            class StoppingCriteriaSub(StoppingCriteria):
                def __init__(self, stops=[], encounters=1):
                    super().__init__()
                    self.stops = [stop.to(model.device) for stop in stops]

                def __call__(
                    self, input_ids: torch.LongTensor, scores: torch.FloatTensor
                ):
                    for stop in self.stops:
                        if torch.all((stop == input_ids[0][-len(stop) :])).item():
                            return True

                    return False

            stop_words = [conv.stop_str]

            if "beluga" in model_path:
                stop_words_ids = [
                    tokenizer.encode(
                        "a" + stop_word, return_tensors="pt", add_special_tokens=False
                    )[0, 2:]
                    for stop_word in stop_words
                ]
            else:
                stop_words_ids = [
                    tokenizer.encode(
                        stop_word, return_tensors="pt", add_special_tokens=False
                    )[0, :]
                    for stop_word in stop_words
                ]

            if conv.stop_token_ids:
                stop_words_ids += [torch.Tensor(conv.stop_token_ids).to(model.device)]

            print(f"stop_words_ids {stop_words_ids}", file=sys.stderr)

            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stop_words_ids)]
            )

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            if os.path.exists(conv_template):
                conv = get_conv_from_template_path(conv_template)
            else:
                conv = get_conversation_template(conv_template)

            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                print(f"prompt: {prompt}", file=sys.stderr)

                if generate_answers:
                    if "RWKV" in model_path:
                        input_ids = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0)
                    else:
                        input_ids = tokenizer.encode(
                            prompt, return_tensors="pt", add_special_tokens=False
                        )

                    print(f"input_ids: {input_ids}", file=sys.stderr)
                    print(f"len(input_ids): {len(input_ids[0])}", file=sys.stderr)

                    # some models may error out when generating long outputs
                    try:
                        if "RWKV" in model_path:
                            output_ids = model.generate(
                                input_ids=input_ids,
                                temperature=temperature,
                                top_p=top_p,
                                max_new_tokens=1024,
                            )

                            output_ids = output_ids[0][len(input_ids[0]) :]
                        else:
                            output_ids = model.generate(
                                input_ids=input_ids.to(model.device),
                                stopping_criteria=stopping_criteria,
                                max_new_tokens=max_tokens - len(input_ids[0]),
                                pad_token_id=tokenizer.pad_token_id,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                num_beams=num_beams,
                                repetition_penalty=repetition_penalty,
                                do_sample=True if temperature > 1e-4 else False,
                            )

                            if model.config.is_encoder_decoder:
                                output_ids = output_ids[0]
                            else:
                                output_ids = output_ids[0][len(input_ids[0]) :]

                        print(f"output_ids: { {id:tokenizer.convert_ids_to_tokens([id]) for id in output_ids.detach().cpu().numpy()} }", file=sys.stderr)
                        print(f"len(output_ids): {len(output_ids)}", file=sys.stderr)

                        if "RWKV" in model_path:
                            output = tokenizer.decode(output_ids).strip()
                        elif "stablelm-instruct-alpha" in model_path:
                            print('special stablelm-alpha decode')
                            output = tokenizer.decode(
                                output_ids,
                                skip_special_tokens=True,
                            )
                        else:
                            output = tokenizer.decode(
                                output_ids,
                                spaces_between_special_tokens=False,
                            )

                        for special_token in tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")

                        if conv.stop_str:
                            output = output.split(conv.stop_str)[0].strip()

                        output = output.strip()

                        print(f"output: {output}", file=sys.stderr)

                    except RuntimeError as e:
                        print(e)
                        print("ERROR question ID: ", question["question_id"])
                        output = "ERROR"

                    turns.append(output)
                    conv.messages[-1][-1] = output

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding="utf-8") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "prompt": prompt,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    Fire(get_model_answers)
