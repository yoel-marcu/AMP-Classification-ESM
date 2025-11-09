import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CSVSequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.samples = self._load_samples(dataframe)

    def _load_samples(self, dataframe):
        samples = []
        for _, row in dataframe.iterrows():
            code = self.extract_id(row["ID"])
            label = float(row["LABEL"])
            samples.append((code, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def extract_id(self, id_string):
        return id_string.split("_")[0]

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        return code, torch.tensor(label, dtype=torch.float32)

