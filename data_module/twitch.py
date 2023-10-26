import os
import pickle
import json
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from typing import List
    from transformers import PreTrainedTokenizerFast

DEFAULT_MODEL = 'cardiffnlp/twitter-roberta-base-hate-latest'


@dataclass
class TwitchItem:
    text: str
    label: int
    prediction: int = field(init=False, default=None)

    def __post_init__(self):
        self.text = str(self.text).replace("'", '')


@dataclass
class TwitchDataset(Dataset):
    data: 'List[TwitchItem]'
    tokenizer: 'PreTrainedTokenizerFast'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch: 'List[TwitchItem]'):
        formatted_batch = [item.text for item in batch]
        batch_encodings = self.tokenizer(
            formatted_batch,
            padding='longest',
            max_length=128,
            truncation=True,
            return_tensors='pt'
        )
        labels = torch.tensor([item.label for item in batch], dtype=torch.long)

        return {
            'twitch_items': batch,
            'labels': labels,
            'batch_encodings': batch_encodings
        }


@dataclass
class TwitchDataModule(LightningDataModule):
    file_path: str
    model_name: str = field(init=False, default=DEFAULT_MODEL)
    batch_size: int = field(init=False, default=32)
    num_workers: int = field(init=False, default=0)

    def __post_init__(self):
        super().__init__()
        self.train_size = None
        self.val_size = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        with open(self.file_path, 'rb') as pickle_file:
            self.data = pickle.load(pickle_file)
            self.data = [TwitchItem(*sample) for sample in self.data]

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dataset = TwitchDataset(self.data, tokenizer)
        self.collate_fn = dataset.collate_fn

        self.train_size = int(0.9 * len(self.data))
        self.val_size = len(self.data) - self.train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [self.train_size, self.val_size]
        )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return dataloader

    def openAI_export(self, output_fpath):
        def _write_jsonl(output_fpath, data):
            data = [
                {'prompt': item.text, 'completion': str(item.label)}
                for item in data
            ]

            with open(output_fpath, 'w') as jsonl_file:
                for item in data:
                    jsonl_file.write(json.dumps(item) + '\n')

        directory, file_name = os.path.split(output_fpath)
        train_path = os.path.join(directory, f'train_{file_name}')
        val_path = os.path.join(directory, f'validation_{file_name}')
        _write_jsonl(train_path, self.train_dataset)
        _write_jsonl(val_path, self.val_dataset)
