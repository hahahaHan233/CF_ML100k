import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yaml
import utils
import model
import torch.nn.functional as F

log_dir = '../log/'
log_name = '2024-05-21_07-19-40'
fig_dir = './'
model_path = os.path.join(log_dir, log_name, 'model_best.pth')

# 加载配置文件
config_path = '../config/config.yaml'
config = utils.load_config(config_path)
# 从配置文件中读取参数
embedding_dim = config['model_config']['embedding_size']
batch_size = config['training_config']['batch_size']
epochs = config['training_config']['num_epochs']
learning_rate = config['model_config']['learning_rate']
weight_decay = config['model_config']['weight_decay']
dropout_rate = config['model_config']['dropout_rate']

train_path = config['data_config']['train_path']
test_path = config['data_config']['test_path']
num_users = config['data_config']['num_users']
num_items = config['data_config']['num_items']

# 定义颜色和标记列表
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

markers = ['o', 's', 'v', '^', '<', '>', '8', 'p', 'P', '*', 'h', 'H', 'D', 'd', 'o', '+', 'x', 'X']
def visualize_movie():
    # 读取电影数据
    movie_data_path = '../data/ml-100k/u.item'
    columns = [
        'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown',
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    movie_data = pd.read_csv(movie_data_path, sep='|', header=None, names=columns, encoding='latin-1')

    # 提取电影ID和类型信息
    movie_ids = movie_data['movie_id'].values
    movie_genres = movie_data.iloc[:, 5:].values  # 从第6列开始是类型信息

    # 转换为Tensor
    movie_ids_tensor = torch.tensor(movie_ids, dtype=torch.long)
    movie_genres_tensor = torch.tensor(movie_genres, dtype=torch.float32)

    # 初始化和加载模型
    import model
    model = model.RecModel(num_users, num_items, embedding_dim, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # 获取item embeddings的权重
    embeddings_tensor = model.item_embeddings.weight.data

    # 将 PyTorch 张量转换为 NumPy 数组
    embeddings_np = embeddings_tensor.cpu().numpy()

    # 确保电影ID与embedding的索引对齐
    movie_embeddings_np = embeddings_np[movie_ids]

    # 计算新的 movie embedding
    item_feature = model.item_feature.detach()
    item_feature_linear = model.item_feature_linear
    new_movie_embeddings = movie_embeddings_np + F.relu(item_feature_linear(item_feature)).detach().cpu().numpy()

    # 随机采样100个电影ID
    np.random.seed(42)
    sample_indices = np.random.choice(len(movie_ids), 100, replace=False)
    sample_movie_ids = movie_ids[sample_indices]
    sample_movie_embeddings_np = new_movie_embeddings[sample_indices]
    sample_movie_genres = movie_genres[sample_indices]

    # 使用 PCA 将嵌入降维到 2D
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(sample_movie_embeddings_np)

    # 获取电影类型的标签列表
    genre_labels = columns[6:]



    # 可视化 PCA 降维结果
    plt.figure(figsize=(12, 8))
    added_labels = set()
    for i, movie_id in enumerate(sample_movie_ids):
        genres = sample_movie_genres[i]
        nonzero_genres = np.nonzero(genres)[0]
        if len(nonzero_genres) > 0:
            random_genre_index = np.random.choice(nonzero_genres)
            if random_genre_index < len(genre_labels):
                first_genre = genre_labels[random_genre_index]
                color = colors[random_genre_index % len(colors)]
                marker = markers[random_genre_index % len(markers)]
                if first_genre not in added_labels:
                    plt.scatter(embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1], label=first_genre, color=color, marker=marker, alpha=1.0,s=100)
                    added_labels.add(first_genre)
                else:
                    plt.scatter(embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1], color=color, marker=marker, alpha=1.0,s=100)
    plt.title('PCA Visualization of Movie Embeddings (100 Sampled Movies)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Genres')
    plt.grid(True)

    # 显示图表
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,'visualize_i.pdf'))
    plt.show()

def visualize_user():
    user_data_path = '../data/ml-100k/u.user'
    # 读取用户数据
    columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_data = pd.read_csv(user_data_path, sep='|', header=None, names=columns, encoding='latin-1')
    user_data = user_data.sort_values(by='user_id').reset_index(drop=True)
    user_id = user_data['user_id'].values
    occupations = user_data['occupation'].values

    # 编码职业
    from sklearn.preprocessing import LabelEncoder
    occupation_encoder = LabelEncoder()
    encoded_occupations = occupation_encoder.fit_transform(occupations)
    occupation_labels = occupation_encoder.classes_

    # 初始化和加载模型
    model_instance = model.RecModel(num_users, num_items, embedding_dim, dropout_rate=dropout_rate)
    model_instance.load_state_dict(torch.load(model_path))

    model_instance.eval()

    # 获取user embeddings的权重
    user_embeddings_tensor = model_instance.user_embeddings.weight.data

    # 将 PyTorch 张量转换为 NumPy 数组
    user_embeddings_np = user_embeddings_tensor[user_id].cpu().numpy()

    # 计算新的 user embedding
    user_feature = model_instance.user_feature.detach()
    user_feature_linear = model_instance.user_feature_linear
    new_user_embeddings = user_embeddings_np + F.relu(user_feature_linear(user_feature)).detach().cpu().numpy()

    # 随机采样100个用户ID
    np.random.seed(42)
    sample_user_indices = np.random.choice(len(new_user_embeddings), 100, replace=False)
    sample_user_embeddings_np = new_user_embeddings[sample_user_indices]
    sample_occupations = encoded_occupations[sample_user_indices]

    # 使用 PCA 将嵌入降维到 2D
    pca = PCA(n_components=2)
    embeddings_2d_pca_user = pca.fit_transform(sample_user_embeddings_np)

    # colors = [
    #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    #     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    # ]
    #
    # markers = ['o', 's', 'v', '^', '<', '>', '8', 'p', 'P', '*', 'h', 'H', 'D', 'd', 'o', '+', 'x', 'X']

    # 可视化 PCA 降维结果 - User
    plt.figure(figsize=(12, 8))
    added_labels = set()
    for i in range(len(sample_user_embeddings_np)):
        occupation = sample_occupations[i]
        color = colors[occupation % len(colors)]
        marker = markers[occupation % len(markers)]
        label = occupation_labels[occupation]
        if label not in added_labels:
            plt.scatter(embeddings_2d_pca_user[i, 0], embeddings_2d_pca_user[i, 1], label=label, color=color, marker=marker, alpha=1.0,s=100)
            added_labels.add(label)
        else:
            plt.scatter(embeddings_2d_pca_user[i, 0], embeddings_2d_pca_user[i, 1], color=color, marker=marker, alpha=1.0,s=100)
    plt.title('PCA Visualization of User Embeddings (100 Sampled Users)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Occupations')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(fig_dir,'visualize_u.pdf'))
    # 显示图表
    plt.show()



if __name__ == '__main__':
    visualize_movie()
    visualize_user()