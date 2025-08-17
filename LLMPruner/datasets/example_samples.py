import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_examples(dataset, tokenizer, n_samples, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    else:
        if dataset == 'alpaca':
            file_name = 'calibration_data/alpaca_82_2048_1.pt'
        elif dataset == 'openbookqa':
            file_name = 'calibration_data/openbookqa_385_2048_1.pt'
        elif dataset == 'piqa':
            file_name = 'calibration_data/piqa_489_2048_1.pt'
        elif dataset == 'wikitext2':
            file_name = 'calibration_data/wikitext2_256_2048_1.pt'
        else:
            raise NotImplementedError

        calib_dataset = torch.load(
            f=file_name,
            weights_only=True,
        )

        prompt_list = []
        for i in range(len(calib_dataset)):
            j = random.randint(0, calib_dataset[i]['input_ids'].shape[1] - seq_len)
            prompt_list.append(calib_dataset[i]['input_ids'][:, j:j+seq_len])

            if i == 0:
                prompts = calib_dataset[i]['input_ids']
            else:
                prompts = torch.cat((prompts, calib_dataset[i]['input_ids']), dim=0)

        return torch.cat(prompt_list, dim=0)
