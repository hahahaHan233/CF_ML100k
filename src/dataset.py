import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class MovieLensDataset(Dataset):
    """MovieLens 100K Dataset for PyTorch with support for k-fold cross-validation"""

    def __init__(self, data_path, dataset_part):
        """
        Args:
            data_path (string): 路径到 'ml-100k' 文件夹.
            dataset_part (string): 用于指定使用哪部分数据，例如 'u1.base', 'u1.test', etc.
        """
        self.data_path = data_path
        self.ratings_df = pd.read_csv(os.path.join(data_path, dataset_part), sep='\t', header=None,
                                      names=['user_id', 'item_id', 'rating', 'timestamp'], usecols=[0, 1, 2])

        # 生成用户和物品的索引映射
        self.user_ids = self.ratings_df['user_id'].unique()
        self.item_ids = self.ratings_df['item_id'].unique()
        self.user_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_idx = self.user_to_index[row['user_id']]
        item_idx = self.item_to_index[row['item_id']]
        rating = row['rating']
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating, dtype=torch.float)

### 2. 创建一个方法来生成交叉验证的数据加载器
def load_data_for_fold(data_path, fold, batch_size=64):
    train_file = f'u{fold}.base'
    test_file = f'u{fold}.test'

    train_dataset = MovieLensDataset(data_path=data_path, dataset_part=train_file)
    test_dataset = MovieLensDataset(data_path=data_path, dataset_part=test_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    dataset_path = '../data/ml-100k'  # 更新为你的数据集路径

    # 使用第一折的数据进行示例
    train_loader, test_loader = load_data_for_fold(dataset_path, fold=1)

    for user_indices, item_indices, ratings in train_loader:
        # 在这里进行你的模型训练逻辑
        pass
