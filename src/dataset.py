import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class MovieLensDataset(Dataset):
    """MovieLens 100K Dataset for PyTorch with support for k-fold cross-validation"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.ratings_df = pd.read_csv(data_path, sep='\t', header=None,
                                      names=['user_id', 'item_id', 'rating', 'timestamp'], usecols=[0, 1, 2])

        # 生成用户和物品的索引映射
        self.user_ids = self.ratings_df['user_id'].unique()
        self.item_ids = self.ratings_df['item_id'].unique()
        self.user_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        #todo: 加入user/item base rate

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_idx = self.user_to_index[row['user_id']]
        item_idx = self.item_to_index[row['item_id']]
        rating = row['rating']
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating, dtype=torch.float)
