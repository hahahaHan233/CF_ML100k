import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def preprocess_user_data(user_data_path):
    # 读取用户数据
    columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_data = pd.read_csv(user_data_path, sep='|', header=None, names=columns, encoding='latin-1')
    user_data = user_data.sort_values(by='user_id').reset_index(drop=True)

    # 编码性别和职业
    gender_encoder = LabelEncoder()
    user_data['gender'] = gender_encoder.fit_transform(user_data['gender'])

    occupation_encoder = OneHotEncoder()
    occupation_encoded = occupation_encoder.fit_transform(user_data[['occupation']]).toarray()

    # 将所有特征组合成一个输入张量
    user_features = user_data[['age', 'gender']].values
    user_features = torch.tensor(user_features, dtype=torch.float32)
    occupation_features = torch.tensor(occupation_encoded, dtype=torch.float32)
    user_features = torch.cat([user_features, occupation_features], dim=1)

    return user_features


def preprocess_movie_data(movie_data_path):
    # 读取电影数据
    columns = [
        'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    movie_data = pd.read_csv(movie_data_path, sep='|', header=None, names=columns, encoding='latin-1')
    movie_data = movie_data.sort_values(by='movie_id').reset_index(drop=True)

    # 提取电影类型信息
    movie_genres = movie_data.iloc[:, 5:].values  # 从第6列开始是类型信息
    movie_genres = torch.tensor(movie_genres, dtype=torch.float32)

    return movie_genres


class RecModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate=0.5, num_heads=2):
        super(RecModel, self).__init__()
        print('Our model')
        # index space [1,num_users], [1,num_items]
        num_users += 1
        num_items += 1

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        '''
        # ================================================
        # Multi-head max dot scores
        self.num_heads = int(num_heads)
        self.head_dim = embedding_dim // num_heads
        # Same In-Out dimension
        self.user_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.item_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        '''
        # ================================================
        # Auxiliary information
        self.user_feature = preprocess_user_data('../data/ml-100k/u.user')  # n_user x feature_dim
        self.item_feature = preprocess_movie_data('../data/ml-100k/u.item')  # n_item x feature_dim

        user_feature_dim = self.user_feature.shape[1]
        item_feature_dim = self.item_feature.shape[1]
        self.user_feature_linear = nn.Linear(user_feature_dim, embedding_dim,
                                             bias=False)
        self.item_feature_linear = nn.Linear(item_feature_dim, embedding_dim,
                                             bias=False)

        self.initialize()

    def initialize(self):
        # # Initialize
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)
        self.user_biases.weight.data.uniform_(-0.01, 0.01)
        self.item_biases.weight.data.uniform_(-0.01, 0.01)

    def forward(self, user_indices, item_indices):
        batch_size = user_indices.shape

        # user_embedding = self.user_embeddings(user_indices)  # batch x emb_dim
        # item_embedding = self.item_embeddings(item_indices)  # batch x emb_dim

        user_embedding = self.user_embeddings(user_indices) + F.relu(
            self.user_feature_linear(self.user_feature[user_indices - 1]))  # batch x emb_dim
        item_embedding = self.item_embeddings(item_indices) + F.relu(
            self.item_feature_linear(self.item_feature[item_indices - 1]))  # batch x emb_dim


        '''
        # Multi-head max scores
        user_embedding = self.user_linear(user_embedding)
        item_embedding = self.item_linear(item_embedding)
        user_heads = user_embedding.view(-1, self.num_heads, self.head_dim)  # batch x head_num x head_dim
        item_heads = item_embedding.view(-1, self.num_heads, self.head_dim)  # batch x head_num x head_dim

        multi_head_scores = torch.bmm(user_heads,
                                      item_heads.view(-1, self.head_dim, self.num_heads))  # batch x head_num x head_num
        max_scores_per_head, _ = multi_head_scores.max(dim=-1)  # batch x num_heads
        max_sim = max_scores_per_head.sum(dim=-1)  # batch

        predictions = max_sim + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()
        '''

        '''
        # linear embedding
        # user_feature = F.relu(self.user_linear(user_embedding))
        # item_feature = F.relu(self.item_linear(item_embedding))
        user_feature = self.user_linear(user_embedding)
        item_feature = self.item_linear(item_embedding)
        predictions = (user_feature * item_feature).sum(1)  # batch
        predictions = predictions + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()
        '''

        # Base method: inner product
        predictions = (user_embedding * item_embedding).sum(1)  # batch
        predictions = predictions + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()

        predictions = torch.sigmoid(predictions) * 4 + 1

        return predictions  # batch


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

        user_embedding = self.user_embeddings(user_indices)  # batch x emb_dim
        item_embedding = self.item_embeddings(item_indices)  # batch x emb_dim

        # Dot product for recommendation scores
        predictions = (user_embedding * item_embedding).sum(1)  # batch
        # predictions = predictions + self.global_bias * torch.ones(batch_size) + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()

        # Better performance
        predictions = predictions + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()

        # predictions = torch.sigmoid(predictions) * 4 + 1

        return predictions  # batch


class NonNegativeMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout_rate=0.5, global_mean=2.5):
        super(NonNegativeMatrixFactorization, self).__init__()
        print('NMF model: NonNegativeMatrixFactorization')

        # index space [1,num_users], [1,num_items]
        num_users += 1
        num_items += 1

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize with non-negative values
        self.user_embeddings.weight.data.uniform_(0, 0.05)
        self.item_embeddings.weight.data.uniform_(0, 0.05)

        self.global_bias = nn.Parameter(torch.tensor([global_mean]))
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)

        # Initialize biases
        self.user_biases.weight.data.uniform_(0, 0.01)
        self.item_biases.weight.data.uniform_(0, 0.01)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embeddings(user_indices)  # batch x emb_dim
        item_embedding = self.item_embeddings(item_indices)  # batch x emb_dim

        # Ensure embeddings are non-negative (NMF constraint)
        user_embedding = torch.clamp(user_embedding, min=0)
        item_embedding = torch.clamp(item_embedding, min=0)

        # Dot product for recommendation scores
        predictions = (user_embedding * item_embedding).sum(1)  # batch
        predictions = predictions + self.global_bias + self.user_biases(user_indices).squeeze() + self.item_biases(item_indices).squeeze()

        return predictions  # batch

    def apply_non_negativity(self):
        # Ensure non-negativity of embeddings after each gradient update
        with torch.no_grad():
            self.user_embeddings.weight.data.clamp_(min=0)
            self.item_embeddings.weight.data.clamp_(min=0)