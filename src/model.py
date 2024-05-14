import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RecModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate=0.5, global_mean=2.5):
        super(RecModel, self).__init__()
        print('Our model')
        # index space [1,num_users], [1,num_items]
        num_users += 1
        num_items += 1

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)

        # self.user_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        # self.item_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.global_bias = nn.Parameter(torch.tensor([global_mean]))
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

        # # Initialize
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)
        self.user_biases.weight.data.uniform_(-0.01, 0.01)
        self.item_biases.weight.data.uniform_(-0.01, 0.01)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, user_indices, item_indices):
        # 获取用户和物品的嵌入向量

        user_embedding = self.user_embeddings(user_indices) # batch x emb_dim
        item_embedding = self.item_embeddings(item_indices) # batch x emb_dim

        predictions = (user_embedding * item_embedding).sum(1) # batch
        #predictions = predictions + self.global_bias + self.user_biases(user_indices) + self.item_biases(item_indices)
        predictions = predictions + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()

        predictions = torch.sigmoid(predictions) * 4 + 1

        return predictions # batch

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate=0.5, global_mean=2.5):
        super(MatrixFactorization, self).__init__()
        print('Baseline model:MatrixFactorization')

        # index space [1,num_users], [1,num_items]
        num_users += 1
        num_items += 1

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.global_bias = nn.Parameter(torch.tensor([global_mean]))
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

        # # Initialize
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)
        self.user_biases.weight.data.uniform_(-0.01, 0.01)
        self.item_biases.weight.data.uniform_(-0.01, 0.01)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, user_indices, item_indices):
        batch_size = user_indices.shape

        user_embedding = self.user_embeddings(user_indices) # batch x emb_dim
        item_embedding = self.item_embeddings(item_indices) # batch x emb_dim

        # Dot product for recommendation scores
        predictions = (user_embedding * item_embedding).sum(1) # batch
        #predictions = predictions + self.global_bias * torch.ones(batch_size) + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()
        predictions = predictions + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()

        predictions = torch.sigmoid(predictions) * 4 + 1

        return predictions # batch
