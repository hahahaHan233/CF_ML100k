import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

import utils
import model
import dataset

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.setup_seed(42)

def train(model, train_loader, optimizer, criterion):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    for batch in train_loader:
        # 假设你的数据集返回用户、项目和评分
        users, items, ratings = batch
        optimizer.zero_grad()  # 清除之前的梯度
        predictions = model(users, items)  # 获得预测结果
        loss = criterion(predictions, ratings)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        total_loss += loss.item()  # 累加损失
    avg_loss = total_loss / len(train_loader)  # 计算平均损失
    return avg_loss

def evaluate(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    total_rmse = 0
    total_count = 0  # 记录总数，用于计算平均RMSE

    with torch.no_grad():  # 不计算梯度
        for batch in test_loader:
            users, items, ratings = batch
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            total_loss += loss.item()

            # 计算RMSE
            mse = ((predictions - ratings) ** 2).mean().item()
            rmse = np.sqrt(mse)
            total_rmse += rmse * len(ratings)  # 累积RMSE，考虑每个batch的大小
            total_count += len(ratings)  # 更新总评分数

    avg_loss = total_loss / len(test_loader)  # 计算平均损失
    avg_rmse = total_rmse / total_count  # 计算平均RMSE

    return avg_loss, avg_rmse  # 返回损失和RMSE


if __name__ == '__main__':
    # ===============================================================
    # Load config .yaml
    config_path = '../config/config.yaml'
    config = utils.load_config(config_path)
    print(config)

    embedding_dim = config['model_config']['embedding_size']
    batch_size = config['training_config']['batch_size']
    epoch = config['training_config']['num_epochs']
    learning_rate = config['model_config']['learning_rate']

    train_path = config['data_config']['train_path']
    test_path = config['data_config']['test_path']

    # ===============================================================
    # Load dataset
    train_dataset = dataset.MovieLensDataset(data_path=train_path)
    test_dataset = dataset.MovieLensDataset(data_path=test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_users = max(len(train_dataset.user_ids),len(test_dataset.user_ids))
    num_items = max(len(train_dataset.item_ids),len(test_dataset.item_ids))

    # ===============================================================
    # Prepare for training
    model = model.RecModel(num_users, num_items, embedding_dim)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.MSELoss()

    # ===============================================================
    # Training
    train_losses = []
    eval_losses = []

    # 训练循环
    for epoch in range(epoch):
        train_loss = train(model, train_loader, optimizer, criterion)
        eval_loss,RMSE = evaluate(model, test_loader, criterion)
        print(RMSE)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # 打印损失
        print(f'Epoch {epoch + 1}/{epoch}, Train Loss: {train_loss}, Eval Loss: {eval_loss}')

        # 绘制损失图
        plt.plot(train_losses, label='Training loss')
        plt.plot(eval_losses, label='Evaluation loss')
        plt.legend()
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()