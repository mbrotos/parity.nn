import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np


class ParityDataset(Dataset):
    def __init__(self, sample_size=100000, bit_len=50, seed=42):
        self.bit_len = bit_len
        self.sample_size = sample_size
        self.seed = seed
        self.features, self.labels = self.generate_data(sample_size, bit_len)

    def __getitem__(self, index):
        return self.features[index, :], self.labels[index]

    def __len__(self):
        return len(self.features)


    def generate_data(self, sample_size, seq_length):
        torch.manual_seed(self.seed)
        bits = torch.randint(2, size=(sample_size, seq_length, 1)).float()
        bitcsum = bits.cumsum(axis=1)
        parity = (bitcsum % 2 != 0).float()

        return bits, parity
