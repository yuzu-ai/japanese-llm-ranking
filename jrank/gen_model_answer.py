"""Generate answers with local models.

Usage:

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path Qwen/Qwen-1_8B


python3 gen_model_answer.py --bench_name rakuda_v2 --model-path EleutherAI/pythia-70m  --model-id pythia-70m --conv_template ./templates/yuzulm.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path line-corporation/japanese-large-lm-1.7b-instruction-sft --model-id line-1.7b --conv_template ./templates/line.json

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-alpha-7b-v2 --model-id stablelm-alpha-7b-v2 --conv_template ./templates/japanese-stablelm.json --repetition_penalty 1.1 --max_tokens 512 

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-gamma-7b --model-id stablelm-gamma-7b --conv_template ./templates/japanese-stablelm.json --repetition_penalty 1.1 --max_tokens 512 

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-chat --model-id youri-7b-chat --conv_template ./templates/youri-chat.json --repetition_penalty 1.05 --max_tokens 512 

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path rinna/youri-7b-instruction --model-id youri-7b-instruction --conv_template ./templates/youri-instruction.json --repetition_penalty 1.1 --max_tokens 512 

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path llm-jp/llm-jp-13b-instruct-full-jaster-dolly-oasst-v1.0 --model-id llm-jp-13b-instruct --conv_template ./templates/llm-jp-instruct.json --repetition_penalty 1.05 

python3 gen_model_answer.py --bench_name rakuda_v2 --model-path stabilityai/japanese-stablelm-instruct-beta-70b --model-id stablelm-beta-70b --conv_template ./templates/japanese-stablelm-beta.json --max_tokens 1024 --max_gpu_memory "40GiB" --num_gpus 4


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

# from adapters import (
#     FastTokenizerAvailableBaseAdapter,
#     JapaneseStableLMAlphaAdapter,
#     JapaneseStableLMAlphaAdapterv2,
#     RwkvWorldAdapter,
# )

from fire import Fire
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList, StoppingCriteria

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def get_model_answers(
    # model parameters
    model_path: str,
    # benchmark parameters
    bench_name: str = "rakuda_v2",
    model_kwargs: dict = {},
    tokenizer_path: Optional[str] = None,
    tokenizer_kwargs: dict = {},
    # template parameters
    chat_template: Optional[str] = None,
    # model parameters
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,  # only relevant for numgpus > 1
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
    # generation parameters
    repetition_penalty: float = 1.0,
    max_tokens: Optional[int] = None,
    # generate the answers (set to False for debugging prompts)
    generate_answers: bool = True,
):
    # PARSE INPUTS
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    default_tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": True
    }

    tokenizer_kwargs = default_tokenizer_kwargs | tokenizer_kwargs

    # TODO: ADD SUPPORT FOR OTHER QUESTION DATASETS LIKE JA-MT-BENCH AND ELYZA-TASKS-100
    question_file = f"data/{bench_name}/questions.jsonl"
    questions = load_questions(question_file)

    print(tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    # CHECK IF THE TOKENIZER INCLUDES A CHAT TEMPLATE AND IF IT DOESNT AND A TEMPLATE IS NOT PROVIDE, THROW AN ERROR
    if chat_template is None:
        raise NotImplementedError("Check there is a chat template in the tokenizer")
    else:
        raise NotImplementedError("There should be a chat template")
        #figure out how to apply a chat template here

    # TODO: TURN OFF SAMPLING
    sampling_params = SamplingParams(
            max_tokens=1000,
            temperature=0.,
            repetition_penalty=1.05,
            skip_special_tokens=True,
    )


    # if max_tokens is None:
    #     seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
    #     for attr in seqlen_config_attrs:
    #         if hasattr(model.config, attr):
    #             max_tokens = getattr(model.config, attr)
    #             print(f"{attr}: {max_tokens}")
    #     print(f"=> Max_tokens: {max_tokens}")
    #     if max_tokens is None:
    #         raise ValueError(
    #             "max_tokens must be specified if model.config doesn't have an attribute n_positions, max_position_embeddings, or n_ctx"
    #         )
    # if model_id == "matsuo-lab/weblab-10b-instruction-sft":
    #     tokenizer.pad_token_id = 1
    #     tokenizer.eos_token_id = 0
    #     tokenizer.bos_token_id = tokenizer.pad_token_id

    # if "RWKV" not in model_path:
    #     print(
    #         f"pad_token_id={tokenizer.pad_token_id}, bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}"
    #     )


    llm = LLM(model=model_path, tensor_parallel_size=1)

    chat_token_ids = []

    for question in tqdm(questions):
            chat = []
            chat.append({'role': 'system', 'content': "system prompt goes here..." })
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()


                tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
                chat_token_ids.append(tokens)
                print(f"prompt: {prompt}", file=sys.stderr)

            #     if generate_answers:
            #         if "RWKV" in model_path:
            #             input_ids = torch.Tensor(tokenizer.encode(prompt)).unsqueeze(0)
            #         else:
            #             input_ids = tokenizer.encode(
            #                 prompt, return_tensors="pt", add_special_tokens=False
            #             )

            #         print(f"input_ids: {input_ids}", file=sys.stderr)
            #         print(f"len(input_ids): {len(input_ids[0])}", file=sys.stderr)

            #         # some models may error out when generating long outputs
            #         try:
            #             if "RWKV" in model_path:
            #                 output_ids = model.generate(
            #                     input_ids=input_ids,
            #                     temperature=temperature,
            #                     top_p=top_p,
            #                     max_new_tokens=1024,
            #                 )

            #                 output_ids = output_ids[0][len(input_ids[0]) :]
            #             else:
            #                 print('GENERATING ')
            #                 output_ids = model.generate(
            #                     input_ids=input_ids.to(model.device),
            #                     stopping_criteria=stopping_criteria,
            #                     max_new_tokens=max_tokens - len(input_ids[0]),
            #                     pad_token_id=tokenizer.pad_token_id,
            #                     bos_token_id=tokenizer.bos_token_id,
            #                     eos_token_id=tokenizer.eos_token_id,
            #                     temperature=temperature,
            #                     top_p=top_p,
            #                     top_k=top_k,
            #                     num_beams=num_beams,
            #                     repetition_penalty=repetition_penalty,
            #                     do_sample=True if temperature > 1e-4 else False,
            #                 )
            #                 print('COMPLETED GENERATING')

            #                 if model.config.is_encoder_decoder:
            #                     output_ids = output_ids[0]
            #                 else:
            #                     output_ids = output_ids[0][len(input_ids[0]) :]

            #             print(f"output_ids: { {id:tokenizer.convert_ids_to_tokens([id]) for id in output_ids.detach().cpu().numpy()} }", file=sys.stderr)
            #             print(f"len(output_ids): {len(output_ids)}", file=sys.stderr)

            #             if "RWKV" in model_path:
            #                 output = tokenizer.decode(output_ids).strip()
            #             elif "stablelm-instruct-alpha" in model_path:
            #                 print('special stablelm-alpha decode')
            #                 output = tokenizer.decode(
            #                     output_ids,
            #                     skip_special_tokens=True,
            #                 )
            #             else:
            #                 output = tokenizer.decode(
            #                     output_ids,
            #                     spaces_between_special_tokens=False,
            #                 )

            #             for special_token in tokenizer.special_tokens_map.values():
            #                 if isinstance(special_token, list):
            #                     for special_tok in special_token:
            #                         output = output.replace(special_tok, "")
            #                 else:
            #                     output = output.replace(special_token, "")

            #             if conv.stop_str:
            #                 output = output.split(conv.stop_str)[0].strip()

            #             output = output.strip()

            #             print(f"output: {output}", file=sys.stderr)

            #         except RuntimeError as e:
            #             print(e)
            #             print("ERROR question ID: ", question["question_id"])
            #             output = "ERROR"

            #         turns.append(output)
            #         conv.messages[-1][-1] = output

            # choices.append({"index": i, "turns": turns})

    # Batch
    outputs = llm.generate(prompt_token_ids=chat_token_ids, sampling_params=sampling_params, use_tqdm=True)

    for question, output in zip(questions, outputs):
        prompt = tokenizer.decode(output.prompt_token_ids, skip_special_tokens=True)
        generated_text = output.outputs[0].text

    # # Dump answers
    # os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    # with open(os.path.expanduser(answer_file), "a") as fout:
    #     ans_json = {
    #         "question_id": question["question_id"],
    #         "answer_id": shortuuid.uuid(),
    #         "model_id": model_id,
    #         "prompt": prompt,
    #         "choices": choices,
    #         "tstamp": time.time(),
    #     }
    #     fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    Fire(get_model_answers)
