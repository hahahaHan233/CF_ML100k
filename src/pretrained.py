import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pretrain_item():
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

    class MovieEmbeddingPretrainModel(nn.Module):
        def __init__(self, genre_dim, embed_dim):
            super(MovieEmbeddingPretrainModel, self).__init__()
            self.fc = nn.Linear(genre_dim, embed_dim)

        def forward(self, x):
            return self.fc(x)

    # 初始化模型、损失函数和优化器
    genre_dim = movie_genres_tensor.shape[1]
    embed_dim = 128  # 设定embedding的维度
    model = MovieEmbeddingPretrainModel(genre_dim, embed_dim)
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(movie_genres_tensor)
        loss = criterion(embeddings, embeddings.detach())  # 自监督训练
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 提取预训练的电影embedding
    with torch.no_grad():
        movie_embeddings = model(movie_genres_tensor)

    # 保存embedding到文件
    # embedding_path = '../data/ml-100k/movie_embeddings.pt'
    # torch.save(movie_embeddings, embedding_path)

    # 将embedding转换为numpy数组
    embeddings_np = movie_embeddings.numpy()

    # 使用PCA将embedding降维到2D
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings_np)

    # 使用t-SNE将embedding降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d_tsne = tsne.fit_transform(embeddings_np)

    # 可视化PCA降维结果
    plt.figure(figsize=(16, 8))

    # 创建一个子图：PCA
    plt.subplot(1, 2, 1)
    for i, genre in enumerate(columns[6:]):
        genre_indices = movie_genres[:, i] == 1
        plt.scatter(embeddings_2d_pca[genre_indices, 0], embeddings_2d_pca[genre_indices, 1], label=genre, alpha=0.6)
    plt.title('PCA Visualization of Movie Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)

    # 创建另一个子图：t-SNE
    plt.subplot(1, 2, 2)
    for i, genre in enumerate(columns[6:]):
        genre_indices = movie_genres[:, i] == 1
        plt.scatter(embeddings_2d_tsne[genre_indices, 0], embeddings_2d_tsne[genre_indices, 1], label=genre, alpha=0.6)
    plt.title('t-SNE Visualization of Movie Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)

    # 显示图表
    plt.tight_layout()
    plt.show()

def visualize_embeddings(embeddings_np, title, subplot_position):
    # 使用PCA将embedding降维到2D
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings_np)

    # 使用t-SNE将embedding降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d_tsne = tsne.fit_transform(embeddings_np)

    # 可视化PCA降维结果
    plt.subplot(2, 2, subplot_position)
    plt.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], alpha=0.6)
    plt.title(f'PCA: {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    # 可视化t-SNE降维结果
    plt.subplot(2, 2, subplot_position + 1)
    plt.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], alpha=0.6)
    plt.title(f't-SNE: {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

def pretrain_user():
    # 读取用户数据
    user_data_path = '../data/ml-100k/u.user'
    columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    user_data = pd.read_csv(user_data_path, sep='|', header=None, names=columns, encoding='latin-1')

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

    # 用户ID张量
    user_ids = torch.tensor(user_data['user_id'].values, dtype=torch.long)

    # 定义预训练模型
    class UserEmbeddingPretrainModel(nn.Module):
        def __init__(self, input_dim, embed_dim, output_dim):
            super(UserEmbeddingPretrainModel, self).__init__()
            self.embedding = nn.Linear(input_dim, embed_dim)
            self.predict = nn.Linear(embed_dim, output_dim)

        def forward(self, x):
            embed = self.embedding(x)
            return embed, self.predict(embed)

    # 初始化模型、损失函数和优化器
    input_dim = user_features.shape[1]
    embed_dim = 128  # 设定embedding的维度
    output_dim = input_dim  # 我们将预测所有输入特征
    model = UserEmbeddingPretrainModel(input_dim, embed_dim, output_dim)
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 提取训练前的用户embedding
    with torch.no_grad():
        initial_embeddings = model.embedding(user_features).numpy()

    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings, predictions = model(user_features)
        loss = criterion(predictions, user_features)  # 使用输入特征作为监督信号
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 提取预训练的用户embedding
    with torch.no_grad():
        user_embeddings = model(user_features)[0]

    # 将embedding转换为numpy数组
    final_embeddings_np = user_embeddings.numpy()

    # 创建一个新图
    plt.figure(figsize=(16, 16))

    # 可视化训练前的embedding
    visualize_embeddings(initial_embeddings, 'Before Training', 1)

    # 可视化训练后的embedding
    visualize_embeddings(final_embeddings_np, 'After Training', 3)

    # 显示图表
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pretrain_user()
    #pretrain_item()

