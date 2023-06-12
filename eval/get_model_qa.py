## Adapted from FastChat implementation
# https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/get_model_answer.py

import argparse
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.conversation import Conversation
from transformers import GenerationConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from fire import Fire
from typing import Optional
import sys

@torch.inference_mode()
def get_model_answers(
    model_path: str, 
    model_id,
    question_file, 
    answer_file,
    # model parameters
    lora_path: Optional[str] = None,
    conversation_config: Optional[str] = None,
    device: str = 'cuda',
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None, #only relevant for numgpus > 1
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
    # generation parameters
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: float = 0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 128):

    print(device)
    
    question_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    model_path = os.path.expanduser(model_path)

    # Model
    model, tokenizer = load_model(
        model_path, device, lora_path=lora_path, num_gpus=num_gpus, max_gpu_memory=max_gpu_memory, load_8bit=load_8bit, cpu_offloading=cpu_offloading, debug=debug
    )

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )

    answer_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        if conversation_config:
            conv = Conversation.from_json(conversation_config)
        else:
            conv = get_conversation_template(model_id)

        conv.append_message(conv.roles[0], ques_json['text'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f'Prompt: {prompt}', file=sys.stderr)

        input_ids = tokenizer.encode(prompt, return_tensors="pt",add_special_tokens=False)
        print(f'input_ids: {input_ids}', file=sys.stderr)
        print(f'len(input_ids): {len(input_ids)}', file=sys.stderr)

        output_ids = model.generate(
            input_ids=input_ids.to(model.device),
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f'outputs: {outputs}', file=sys.stderr)
        print(f'len(outputs_ids): {output_ids}', file=sys.stderr)

        ans_id = shortuuid.uuid()

        answer_jsons.append(
            {
                "question_id": idx,
                "prompt": prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
            }
        )

    with open(os.path.expanduser(answer_file), "w",  encoding='utf8') as ans_file:
        for line in answer_jsons:
            ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")

    return answer_jsons


if __name__ == "__main__":
    Fire(get_model_answers)