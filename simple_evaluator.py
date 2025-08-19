import json
import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from ppl_eval.utils import ppl_eval


# _MODEL = 'flap_0d2_IFV_ULUM_NO_ATTN_ALPACA_llama_2_7b'
# _MODEL = 'flap_0d2_IFV_ULUM_NO_ATTN_C4_llama_2_7b'
# _MODEL = 'flap_0d2_IFV_ULUM_NO_ATTN_MY_WIKITEXT2_llama_2_7b'
# _MODEL = 'flap_0d2_IFV_ULUM_NO_ATTN_OPENBOOKQA_llama_2_7b'
# _MODEL = 'flap_0d2_IFV_ULUM_NO_ATTN_PIQA_llama_2_7b'
# _MODEL = 'flap_0d4_IFV_ULUM_NO_ATTN_MY_WIKITEXT2_llama_2_7b'
# _MODEL = 'flap_0d6_IFV_ULUM_NO_ATTN_MY_WIKITEXT2_llama_2_7b'
# _MODEL = 'flap_0d8_IFV_ULUM_NO_ATTN_MY_WIKITEXT2_llama_2_7b'
# _MODEL = 'alpaca_0d2/alpaca_0d2'
# _MODEL = 'openbookqa_0d2/openbookqa_0d2'
# _MODEL = 'piqa_0d2/piqa_0d2'
# _MODEL = 'wikitext2_0d2/wikitext2_0d2'
# _MODEL = 'wikitext2_0d4/wikitext2_0d4'
# _MODEL = 'wikitext2_0d6/wikitext2_0d6'
# _MODEL = 'wikitext2_0d8/wikitext2_0d8'
_MODEL = 'c4_0d2/c4_0d2'

_DEVICE = 'cuda:0'
# _DEVICE = 'cuda:1'


def handle_non_serializable_object(object):
    if isinstance(object, np.float32) or \
            isinstance(object, np.float64):
        return float(object)
    elif isinstance(object, np.int32) or \
            isinstance(object, np.int64):
        return int(object)
    elif isinstance(object, set):
        return list(object)
    else:
        return str(object)


if __name__ == '__main__':
    # For FLAP models.
    # -----
    model = AutoModelForCausalLM.from_pretrained(
        # pretrained_model_name_or_path='/workspace/models/Llama-2-7b-hf',
        pretrained_model_name_or_path=f'llms/{_MODEL}',
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        # pretrained_model_name_or_path='/workspace/models/Llama-2-7b-hf',
        pretrained_model_name_or_path=f'llms/{_MODEL}',
        use_fast=False,
    )
    # -----

    model.to(device=_DEVICE)

    LM = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=_DEVICE,
        batch_size=32,
    )
    task_manager = TaskManager()

    results = {}

    b_results = simple_evaluate(
        model=LM,
        tasks=[
            'arc_easy',
            'gsm8k',
            'hellaswag',
        ],
        num_fewshot=0,
        device=_DEVICE,
        task_manager=task_manager,
        confirm_run_unsafe_code=True,
    )
    results.update(b_results['results'])
    print(results)

    p_results = ppl_eval(
        model=model,
        tokenizer=tokenizer,
        datasets=[
            'c4',
            'wikitext2',
        ],
        sequence_length=2048,
        batch_size=8,
        device=_DEVICE,
    )
    results.update(p_results)

    results_str = json.dumps(
        obj=results,
        ensure_ascii=False,
        indent=4,
        default=handle_non_serializable_object,
        sort_keys=True,
    )

    log_name = _MODEL.split('/')[0]
    with open(file=f'results/{log_name}.json', mode='w', encoding='utf-8') as f:
        f.write(results_str)
        f.close()
