import pickle
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from typing import List
    from transformers import PreTrainedTokenizerFast


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
            max_length=self.tokenizer.model_max_length,
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
    model_name: str
    batch_size: int
    num_workers: int = 0

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

        self.train_size = int(0.8 * len(self.data))
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
