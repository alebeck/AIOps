
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class KPIDataset(Dataset):
    
    def __init__(self, path, seq_length=31, step_width=1, transform=None):
        assert seq_length % 2 == 1 and seq_length >= 3, "seq_length has to be odd and >= 3."

        self.df = pd.read_csv(path)

        # extract length and mean of KPI IDs
        ids = self.df['KPI ID'].unique()
        self.kpi_lengths = []

        for _id in ids:
            self.kpi_lengths.append(len(self.df[self.df['KPI ID'] == _id]))

        self.length = sum(self.kpi_lengths)
        self.seq_length = seq_length
        self.step_width = step_width
        self.transform = transform
        
    def __len__(self):
        return math.ceil(self.length / self.step_width)
    
    def __getitem__(self, index):
        index_df = index * self.step_width

        # find out which KPI ID the index belongs to
        kpi_id = 0
        i = index_df
        for kpi_id, length in enumerate(self.kpi_lengths):
            if i < length:
                break
            i -= length

        # i is index inside correct KPI ID
        start_index = index_df - self.seq_length // 2
        end_index = index_df + self.seq_length // 2 + 1
        kpi_start = index_df - i
        kpi_end = index_df + (self.kpi_lengths[kpi_id] - i)
        pad_left = np.array([])
        pad_right = np.array([])

        if start_index < kpi_start:
            # pad left
            pad_left = np.array([0] * (kpi_start - start_index))
            start_index = kpi_start

        if end_index > kpi_end:
            # pad right
            pad_right = np.array([0] * (end_index - kpi_end))
            end_index = kpi_end

        rows = self.df.iloc[start_index : end_index]
        x = np.concatenate([pad_left, rows['value'].values, pad_right]).astype('float32')
        y = self.df.iloc[[index_df]]['label'].values[0].astype('int64')

        if self.transform:
            x = self.transform(x)
            
        return x, y