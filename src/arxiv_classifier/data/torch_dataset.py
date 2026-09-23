"""PyTorch Dataset wrapping tokenized text and multi-hot labels."""

import torch
from torch.utils.data import Dataset


class ArxivTextDataset(Dataset):
    """Tokenizes text lazily per-example rather than all at once upfront,
    keeping memory bounded regardless of dataset size.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = labels  # numpy array, shape (n, n_labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item