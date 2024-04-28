import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RecModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate=0.5):
        super(RecModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)

        # self.user_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        # self.item_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, user_indices, item_indices):
        # 获取用户和物品的嵌入向量
        user_embedding = self.user_embeddings(user_indices)
        item_embedding = self.item_embeddings(item_indices)

        # 线性变换
        # user_embedding = self.user_linear(user_embedding)
        # item_embedding = self.item_linear(item_embedding)

        # 应用 Dropout
        user_embedding = self.dropout(F.relu(user_embedding))
        item_embedding = self.dropout(F.relu(item_embedding))

        # 计算点积
        rating_predictions = (user_embedding * item_embedding).sum(1)

        # remap to [1,5]
        rating_predictions = torch.sigmoid(rating_predictions)
        rating_predictions = rating_predictions * 4 + 1
        return rating_predictions

