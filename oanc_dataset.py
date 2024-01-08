import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from consts import *


# Return a one-hot vector encoding the given character. Invalid characters are all zeros.
def char_to_onehot(c: str) -> torch.Tensor:
    index = OUT_CHARS.find(c)
    result = torch.zeros(size=[27])
    if 0 <= index <= 26:
        result[index] = 1
    return result


# Return an array encoding the given string.
def snippet_to_array(snippet: str) -> torch.Tensor:
    onehots = [char_to_onehot(c) for c in snippet]
    return torch.stack(onehots, dim=1)


# A dataset of snippets followed by the char after them.
class OancSnippetsDataset(Dataset):
    def __init__(self, snippets_dir, transform=None, target_transform=None):
        self.snippets_dir = snippets_dir
        self._snippets_template = "snippets_{}.txt"
        self.transform = transform
        self.target_transform = target_transform

        # Determine the number of snippets.
        num_snippets = 0
        for filename in os.listdir(self.snippets_dir):
            filepath = os.path.join(self.snippets_dir, filename)
            with open(filepath) as file:
                for _ in file:
                    num_snippets += 1

        self._num_snippets = num_snippets

    def __len__(self) -> int:
        return self._num_snippets

    def __getitem__(self, idx: int):
        file_num = idx // SNIPPETS_PER_FILE
        filename = self._snippets_template.format(file_num)
        filepath = os.path.join(self.snippets_dir, filename)

        # Get the snippet from the file.
        line_num = idx % SNIPPETS_PER_FILE
        snippet = None
        with open(filepath) as file:
            for i, line in enumerate(file):
                if i == line_num:
                    snippet = line[:-1]  # Remove the newline at the end.
        if snippet is None:
            raise ValueError()

        prefix, char = snippet[:-1], snippet[-1]
        return snippet_to_array(prefix), char_to_onehot(char)


# A dataset of batches of snippets followed by the char after them.
class OancBatchedSnippetsDataset(Dataset):
    def __init__(self, snippets_dir, transform=None, target_transform=None):
        self.snippets_dir = snippets_dir
        self._snippets_template = "snippets_{}.txt"
        self.transform = transform
        self.target_transform = target_transform

        # Determine the number of batches.
        num_batches = 0
        for _ in os.listdir(self.snippets_dir):
            num_batches += 1

        self.num_batches = num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int):
        # file_num = idx // SNIPPETS_PER_FILE
        filename = self._snippets_template.format(idx)
        filepath = os.path.join(self.snippets_dir, filename)

        # Get the snippet from the file.
        # line_num = idx % SNIPPETS_PER_FILE
        prefixes = list[torch.Tensor]()
        chars = list[torch.Tensor]()
        with open(filepath) as file:
            for line in file:
                line = line.replace("\n", "")
                # if i == line_num:
                # snippet = line[:-1]  # Remove the newline at the end.
                prefix, char = line[:-1], line[-1]
                prefix = snippet_to_array(prefix)
                char = char_to_onehot(char)
                prefixes.append(prefix)
                chars.append(char)
        # if snippet is None:
        #     raise ValueError()

        # Stack the snippets and chars to form a batch.
        batched_prefixes = torch.stack(prefixes, dim=0)
        batched_chars = torch.stack(chars, dim=0)
        return batched_prefixes, batched_chars

        # prefix, char = snippet[:-1], snippet[-1]
        # return snippet_to_array(prefix), char_to_onehot(char)
