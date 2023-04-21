import itertools
from typing import List, Optional, Union
import torch
from attr import dataclass
from datasets import concatenate_datasets, load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

from blade2blade.training.utils import format_history


def filter_by_confidence(dataset: DatasetDict, confidence: float):
    column_names = (
        dataset.column_names
        if isinstance(dataset.column_names, List)
        else list(dataset.column_names.values())[0]
    )
    if "confidence" in column_names:
        return dataset.filter(lambda example: example["confidence"] >= confidence)
    else:
        return dataset


class ProSocialDataset(Dataset):
    """
    Dataset class to load dataset in alleai/prosocial format
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        split: Union[List[str], str] = "train",
        **kwargs
    ):
        super().__init__()

        dataset = load_dataset(path)
        if kwargs.get("confidence"):
            dataset = filter_by_confidence(dataset, kwargs.get("confidence"))

        if isinstance(split, List):
            self.split = "-".join(split)
            self.dataset = concatenate_datasets([dataset[sp] for sp in split])
        else:
            self.split = split
            self.dataset = dataset[split]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx_start = idx
        end = self.dataset[max(0, idx_start - 1)]["episode_done"]
        while (not end) and (idx_start > 0):
            end = self.dataset[max(0, idx_start - 2)]["episode_done"]
            idx_start -= 1
        idx_start = max(0, idx_start)

        history = [
            (self.dataset[i]["context"], self.dataset[i]["response"])
            for i in range(idx_start, idx)
        ]
        history = list(itertools.chain(*history))
        history.append(self.dataset[idx]["context"])
        history = "".join(format_history(history, eos_token=self.tokenizer.eos_token))
        output = (
            self.dataset[idx]["safety_label"]
            + self.tokenizer.sep_token
            + self.tokenizer.sep_token.join(self.dataset[idx]["rots"])
            + self.tokenizer.eos_token
        )

        return history, output


def get_datacollator(is_encoder_decoder: bool, **kwargs):
    if is_encoder_decoder:
        return ProSocialCollator(**kwargs)
    else:
        return ProSocialCollator2(**kwargs)


@dataclass
class ProSocialCollator2:
    tokenizer: PreTrainedTokenizerBase
    evil: bool = False
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    truncation: Optional[bool] = True

    def process_one(self, example):
        history, target = example
        label_mask = [1] * self.max_length
        history = self.tokenizer.encode(history)
        target = self.tokenizer.encode(target)
        input_ids = history + target
        tokens_remove = max(0, len(input_ids) - self.max_length)
        if tokens_remove > 0:
            input_ids = input_ids[tokens_remove:]
        label_mask[: len(history) - tokens_remove] = [0] * (
            len(history) - tokens_remove
        )
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            label_mask[-pad_len:] = [0] * pad_len
        output = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        output["labels"] = output["input_ids"].clone()
        label_mask = torch.tensor(label_mask)
        output["labels"][label_mask == 0] = -100
        return output

    def __call__(self, examples):
        outputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            out = self.process_one(example)
            for k, v in out.items():
                outputs[k].append(v)

        return {k: torch.stack(v) for k, v in outputs.items()}


@dataclass
class ProSocialCollator:
    tokenizer: PreTrainedTokenizerBase
    evil: bool = False
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    truncation: Optional[bool] = True

    def __call__(self, examples):
        input = self.tokenizer(
            [example[0] for example in examples],
            max_length=self.max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=False,
            truncation=self.truncation,
            return_tensors="pt",
        )

        output = self.tokenizer(
            [example[1] for example in examples],
            max_length=self.max_length,
            padding=self.padding,
            add_special_tokens=False,
            truncation=self.truncation,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if not self.evil:
            output["input_ids"][output["input_ids"] == 0] = -100
            output = {
                "input_ids": input["input_ids"],
                "attention_mask": input["attention_mask"],
                "labels": output["input_ids"],
                # "decoder_attention_mask": output["attention_mask"],
            }
        else:
            input["input_ids"][input["input_ids"] == 0] = -100
            output = {
                "input_ids": output["input_ids"],
                "attention_mask": output["attention_mask"],
                "labels": input["input_ids"],
                "decoder_attention_mask": input["attention_mask"],
            }

        return output
