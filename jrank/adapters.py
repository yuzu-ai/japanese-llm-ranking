from fastchat.model.model_adapter import BaseModelAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys 

## For Rinna support
class FastTokenizerAvailableBaseAdapter(BaseModelAdapter):
    # https://huggingface.co/spaces/izumi-lab/stormy-7b-10ep/blob/main/app.py
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        print('Loading using default adapter with model kwargs:', from_pretrained_kwargs)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer

## For JapaneseStableLM support
class JapaneseStableLMAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "japanese-stablelm" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        print('Loading using Japanese-StableLM adapter', file=sys.stderr)
        print('model kwargs:', from_pretrained_kwargs, file=sys.stderr)
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs
        )

        return model, tokenizer


# ## For Rwkv_world support
# from fastchat.model.model_adapter import RwkvAdapter
# from fastchat.model.rwkv_model import RwkvModel
# class RwkvModelFix(RwkvModel):

#     def __init__(self, model_path):

#         from rwkv.model import RWKV
#         from types import SimpleNamespace
#         import warnings

#         self.config = SimpleNamespace(is_encoder_decoder=False)
#         self.model = RWKV(model=model_path, strategy="cuda fp16")
#         self.pipeline = None
#         self.model_path = model_path

#     def generate(
#         self, input_ids, temperature=1.0, top_p = 0.2, max_new_tokens=999, 
#     ):
#         from rwkv.utils import PIPELINE

#         if self.pipeline is None:
#             self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")

#         out_tokens = []
#         out_len = 0
#         out_str = ''
#         occurrence = {}
#         state = None

#         for i in range(max_new_tokens):

#             if i == 0:
#                 out, state = self.pipeline.model.forward(input_ids[0].tolist(), state)
#             else:
#                 out, state = self.pipeline.model.forward([token], state)

#             for n in occurrence: out[n] -= (0.4 + occurrence[n] * 0.4) #### higher repetition penalty because of lower top_p here
            
#             token = self.pipeline.sample_logits(out, temperature=temperature, top_p=top_p) #### sample the next token

#             if token in [0, 261]: break #### exit at token [0] = <|endoftext|>
            
#             out_tokens += [token]

#             for n in occurrence: occurrence[n] *= 0.996 #### decay repetition penalty
#             occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
            
#             tmp = self.pipeline.decode(out_tokens[out_len:])
#             if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): #### print() only when out_str is valid utf-8 and not end with \n
#                 out_str += tmp
#                 print(tmp, end = '', flush = True)
#                 out_len = i + 1    
#             elif '\n\n' in tmp: #### exit at '\n\n'
#                 tmp = tmp.rstrip()
#                 out_str += tmp
#                 print(tmp, end = '', flush = True)
#                 break

#         output = out_str.strip()
#         output_ids = self.pipeline.encode(output)

#         return [input_ids[0].tolist() + output_ids]

# class RwkvWorldAdapter(RwkvAdapter):
#     """The model adapter for BlinkDL/RWKV-4-World"""
    
#     def load_model(self, model_path: str, from_pretrained_kwargs: dict):
#         from rwkv.utils import PIPELINE

#         #from fastchat.model.rwkv_model import RwkvModel
#         model = RwkvModelFix(model_path)
#         revision = from_pretrained_kwargs.get("revision", "main")
#         tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")

#         return model, tokenizer
