import torch
import numpy as np
import lightning as L

from torch.utils.data import Dataset, DataLoader, random_split
from char_rnn.utils import one_hot_encode


class TextDataset(Dataset):
    def __init__(self, text_path: str, seq_length: int, transform=None):
        """
        Initializes the dataset by reading and processing the text file.

        Args:
            text_path (str): Path to the text file.
            seq_length (int): Length of each sequence.
            transform (callable, optional): Transformation function. Defaults to None.
        """
        with open(text_path, "r") as f:
            text = f.read()
        
        self.chars = tuple(set(text))
        self.int2char = {i: ch for i, ch in enumerate(self.chars)}
        self.char2int = {ch: i for i, ch in self.int2char.items()}
        self.encoded = np.array([self.char2int[ch] for ch in text], dtype=np.int64)
        self.seq_length = seq_length
        self.transform = transform

    def __len__(self):
        """
        Returns the number of sequences that can be generated.
        """
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):
        """
        Returns one sequence and its corresponding target.

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequence and target sequence.
        """
        x = self.encoded[idx: idx + self.seq_length]
        y = self.encoded[idx + 1: idx + self.seq_length + 1]

        # Apply one-hot encoding if transform is provided
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TextDataModule(L.LightningDataModule):
    def __init__(self, text_path: str, seq_length: int, batch_size: int, num_workers: int = 11):
        """
        Initializes the DataModule.

        Args:
            text_path (str): Path to the text file.
            seq_length (int): Length of each sequence.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.text_path = text_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None
        self.n_labels = None

    def setup(self, stage=None):
        """
        Sets up the dataset. Called on every GPU separately.

        Args:
            stage (str, optional): Either 'fit', 'validate', 'test', or 'predict'. Defaults to None.
        """
        self.dataset = TextDataset(
            text_path=self.text_path,
            seq_length=self.seq_length,
            transform=lambda x: one_hot_encode(x, len(self.dataset.chars)) if stage != "predict" else None
        )
        self.n_labels = len(self.dataset.chars)

        val_len = int(len(self.dataset) * 0.1)
        train_len = len(self.dataset) - val_len

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, 
            [train_len, val_len], 
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        """
        Returns the DataLoader for training.

        Returns:
            DataLoader: DataLoader for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for validation.

        Returns:
            DataLoader: DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
