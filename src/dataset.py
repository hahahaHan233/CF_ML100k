import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class MovieLensDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.ratings_df = pd.read_csv(data_path, sep='\t', header=None,
                                      names=['user_id', 'item_id', 'rating', 'timestamp'], usecols=[0, 1, 2])

        self.user_ids = self.ratings_df['user_id'].unique()
        self.item_ids = self.ratings_df['item_id'].unique()

        self.global_mean = self.ratings_df['rating'].mean()

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_idx = row['user_id']
        item_idx = row['item_id']
        rating = row['rating']
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating, dtype=torch.float)
