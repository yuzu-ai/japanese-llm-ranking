# Adapted from FastChat implementation
# https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/get_model_answer.py
# and
# https://huggingface.co/spaces/izumi-lab/stormy-7b-10ep/blob/main/app.py

import json
import os
import sys
from typing import Optional

import shortuuid
import torch
from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template
from fastchat.model.model_adapter import load_model, model_adapters
from adapters import FastTokenizerAvailableBaseAdapter, JapaneseStableLMAdapter, RwkvWorldAdapter

from fire import Fire
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteriaList, StoppingCriteria
from utils import load_jsonl, save_jsonl

# Hack the fastchat model adapters
model_adapters[-1] = FastTokenizerAvailableBaseAdapter()
model_adapters.insert(0, JapaneseStableLMAdapter())
for i in range(len(model_adapters)):
    if 'Rwkv' in type(model_adapters[i]).__name__ :
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
    question_file,
    answer_file,
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
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: float = 0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    max_tokens: Optional[int] = None,
    # just generate the prompts (for debug)
    generate_prompts: bool = False,
):
    question_jsons = load_jsonl(question_file)

    if not model_id:
        model_id = shortuuid.uuid()

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    # Model
    if not generate_prompts:
        model, tokenizer = load_model(
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            debug=debug,
        )
        if lora_path is not None:
            model = PeftModel.from_pretrained(
                model, lora_path, torch_dtype=torch.float16
            )

        if max_tokens is None:
            seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
            for attr in seqlen_config_attrs:
                if hasattr(model.config, attr):
                    max_tokens = getattr(model.config, attr)
                    print(f"{attr}: {max_tokens}")
            if max_tokens is None:
                raise ValueError(
                    "max_tokens must be specified if model does not have a max length"
                )
        
        if model_id == 'matsuo-lab/weblab-10b-instruction-sft':
            tokenizer.pad_token_id = 1
            tokenizer.eos_token_id = 0
            tokenizer.bos_token_id = tokenizer.pad_token_id

        print(f"Using max_tokens={max_tokens}")
        print(f"pad_token_id={tokenizer.pad_token_id}, bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}")

    answer_jsons = []
    for i, ques_json in enumerate(tqdm(question_jsons)):
        idx = ques_json["question_id"]
        if os.path.exists(conv_template):
            conv = get_conv_from_template_path(conv_template)
        else:
            conv = get_conv_template(conv_template)

        conv.append_message(conv.roles[0], ques_json["text"])
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        if not generate_prompts:

            if conv.stop_str:

                print(f'Stop string `{conv.stop_str}` in conv template', file=sys.stderr)

                class StoppingCriteriaSub(StoppingCriteria):

                    def __init__(self, stops = [], encounters=1):
                        super().__init__()
                        self.stops = [stop.to(model.device) for stop in stops]

                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                        for stop in self.stops:
                            if torch.all((stop == input_ids[0][-len(stop):])).item():
                                return True

                        return False


                stop_words = [conv.stop_str]

                if 'beluga' in model_path:
                    stop_words_ids = [tokenizer.encode('a'+stop_word, return_tensors='pt',add_special_tokens=False)[0,2:] 
                    for stop_word in stop_words]
                else:
                    stop_words_ids = [tokenizer.encode(stop_word, return_tensors='pt',add_special_tokens=False)[0,:] 
                    for stop_word in stop_words]
                print(f'stop_words_ids {stop_words_ids}', file=sys.stderr)


                stop_words_ids += [torch.Tensor(conv.stop_token_ids).to(model.device)]
                
                print(f'STOP WORD IDS {stop_words_ids} IN CONV', file=sys.stderr)
                print(f'STOP STRING SIZE {stop_words_ids[0].size()} IN CONV', file=sys.stderr)

                stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words_ids)])
            else:
                stopping_criteria = None
                
            if 'RWKV' in model_path:
                #https://github.com/BlinkDL/ChatRWKV/blob/main/API_DEMO_WORLD.py

                input_ids = torch.Tensor(tokenizer.encode(
                    prompt
                )).unsqueeze(0)

                output_ids = model.generate(
                    input_ids=input_ids,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=4096,
                    stopping_criteria=stopping_criteria,
                )

                output_ids = output_ids[0][len(input_ids[0]) :]
                outputs = tokenizer.decode(output_ids).strip()
            else:
                input_ids = tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=False
                )

                output_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    max_new_tokens=max_tokens - len(input_ids[0]),
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )


                output_ids = output_ids[0][len(input_ids[0]) :]
                if 'japanese-stablelm' in model_id:
                    outputs = tokenizer.decode(output_ids, skip_special_tokens=False).replace('<|endoftext|>','').strip()
                else:
                    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

                if conv.stop_str:
                    outputs = outputs.split(conv.stop_str)[0].strip()

            print(f"inputs: {prompt}", file=sys.stderr)
            print(f"input_ids: {input_ids}", file=sys.stderr)
            print(f"len(input_ids): {len(input_ids[0])}", file=sys.stderr)

            print(f"outputs: {outputs}", file=sys.stderr)
            print(f"output_ids: {output_ids}", file=sys.stderr)
            print(f"len(output_ids): {len(output_ids)}", file=sys.stderr)
        else:
            outputs = ""

        answer_jsons.append(
            {
                "question_id": idx,
                "prompt": prompt,
                "text": outputs,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "metadata": {},
            }
        )

    print(answer_jsons)
    save_jsonl(answer_jsons, answer_file)

    return answer_jsons


if __name__ == "__main__":
    Fire(get_model_answers)
