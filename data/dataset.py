import torch
from torch.utils.data import Dataset
import numpy as np


class PretokenizedDataset(Dataset):
    def __init__(self, corpus_file, context_length, split="train"):
        self.corpus_file = corpus_file
        self.context_length = context_length
        self.split = split

        data = np.memmap(self.corpus_file, dtype=np.uint16, mode="r")

        self.num_batches = len(data) // self.context_length
        self.num_batches -= 1
        assert self.num_batches > 0, "Dataset is too small!"

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        corpus = np.memmap(self.corpus_file, dtype=np.uint16, mode="r")

        start = idx * self.context_length
        end = start + self.context_length + 1
        chunk = torch.from_numpy(corpus[start:end].astype(np.int64))

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
