import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RecModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(RecModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embeddings(user_indices)
        item_embedding = self.item_embeddings(item_indices)

        # 应用ReLU激活函数 max(0,x)
        user_embedding = F.relu(user_embedding)
        item_embedding = F.relu(item_embedding)

        # 计算点积
        rating_predictions = (user_embedding * item_embedding).sum(1)
        return rating_predictions

