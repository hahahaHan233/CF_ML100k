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

if __name__ == '__main__':
    # 假设的用户数和物品数
    num_users = 943  # 示例值
    num_items = 1682  # 示例值
    embedding_dim = 100  # 嵌入向量的维度

    # 创建模型实例
    model = RecModel(num_users, num_items, embedding_dim)

    # 选择优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 假设有一些训练数据
    user_indices = torch.tensor([0, 1, 2])  # 假设用户索引
    item_indices = torch.tensor([3, 4, 5])  # 假设物品索引
    ratings = torch.tensor([4.0, 3.0, 5.0])  # 实际评分

    # 训练模型
    model.train()
    optimizer.zero_grad()
    predictions = model(user_indices, item_indices)
    loss = loss_fn(predictions, ratings)
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item()}")
