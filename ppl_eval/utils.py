import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from tqdm import tqdm
from typing import (
    Dict,
    List,
)


def get_dataset(
    tokenizer,
    data_name: str,
    sequence_length: int,
    batch_size: int,
) -> DataLoader:
    class BasicDataset(Dataset):

        def __init__(
            self,
            tensors: torch.Tensor,
        ):
            self.tensors = tensors

        def __getitem__(
            self,
            index: int,
        ) -> torch.Tensor:
            return self.tensors[index]

        def __len__(self) -> int:
            return len(self.tensors)

    def process_data(
        tokenizer,
        dataset: Dataset,
        sequence_length: int,
        data_key: str,
    ) -> BasicDataset:
        texts = '\n\n'.join(dataset[data_key])
        input_ids = tokenizer(
            text=texts,
            return_tensors='pt',
        ).input_ids[0]

        batches_input_ids = []
        data_number = input_ids.numel() // sequence_length
        for data_idx in range(data_number):
            batch_input_ids = input_ids[(data_idx *
                                         sequence_length):((data_idx + 1) *
                                                           sequence_length)]
            batches_input_ids.append(batch_input_ids)

        batches_input_ids = torch.stack(tensors=batches_input_ids)

        return BasicDataset(tensors=batches_input_ids)

    match data_name:
        case 'c4':
            dataset = load_dataset(
                path='json',
                data_files='ppl_eval/c4-validation.json',
            )['train']
            dataset = process_data(
                dataset=dataset[0:2000],
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                data_key='text',
            )
        case 'wikitext2':
            dataset = load_dataset(
                path='wikitext',
                data_dir='wikitext-2-raw-v1',
                split='test',
            )
            dataset = process_data(
                dataset=dataset,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                data_key='text',
            )
        case _:
            raise NotImplementedError(
                f'The dataset {data_name} is NOT implemented yet.')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataloader


@torch.no_grad()
def ppl_eval(
    model,
    tokenizer,
    datasets: List[str],
    sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    ppls = {}

    for dataset in datasets:
        dataloader = get_dataset(
            tokenizer=tokenizer,
            data_name=dataset,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )

        nlls = []
        for batch_idx, batch in enumerate(iterable=tqdm(
                iterable=dataloader,
                desc=f'[Evaluating {dataset}]',
                dynamic_ncols=True,
        )):
            batch = batch.to(device=device)
            output = model(
                batch,
                use_cache=False,
            )

            lm_logits = output.logits
            if torch.isfinite(input=lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss_function = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_function(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nlls.append(loss)

        ppl = np.exp(torch.cat(
            tensors=nlls,
            dim=-1,
        ).mean().item())
        ppls[dataset] = ppl.item()

    return ppls
