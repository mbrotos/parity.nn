import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np


class ParityDataset(Dataset):
    def __init__(self, bitstring_file, bitstring_length, transform=None, target_transform=None):
        self.bit_strings = pd.read_csv(bitstring_file)
        self.bitstring_length = bitstring_length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.bit_strings)

    def __getitem__(self, idx):
        bit_string = torch.tensor([int(bit) for bit in str(self.bit_strings.iloc[idx, 0])], dtype=torch.float32)
        label = torch.tensor([self.bit_strings.iloc[idx, 1]], dtype=torch.float32)
        
        # Pad bit_string to a fixed length
        if len(bit_string) < self.bitstring_length:
            padding = torch.zeros(self.bitstring_length - len(bit_string), dtype=torch.float32)
            bit_string = torch.cat((bit_string, padding))
        
        if self.transform:
            bit_string = self.transform(bit_string)
        if self.target_transform:
            label = self.target_transform(label)
        
        return bit_string, label
