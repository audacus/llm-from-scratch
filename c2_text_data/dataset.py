import tiktoken
import torch
from tiktoken import Encoding
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(
            self,
            text: str,
            tokenizer: Encoding,
            max_length: int,
            stride: int,
    ):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text.
        token_ids = tokenizer.encode(text)

        # Use a sliding window to chunk the text into overlapping sequences of `max_length`.
        # `stride` -> step size
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Return the total number of rows in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, index):
        """Return a single row from the dataset."""
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
        text: str,
        batch_size: int = 4,
        max_length: int = 256,
        stride: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # `True` -> drops the last batch, if it is shorter than the specified `batch_size` to prevent loss spikes during training.
        drop_last=drop_last,
        # Number of CPU processes to use for processing.
        num_workers=num_workers,
    )

    return dataloader
