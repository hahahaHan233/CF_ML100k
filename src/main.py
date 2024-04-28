import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

import utils
import model
import dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout = utils.Logger(filename=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.setup_seed(42)
print(device)

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for users, items, ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, total_rmse = 0, 0
    total_count = 0

    with torch.no_grad():  # 不计算梯度
        for users, items, ratings in test_loader:
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            total_loss += loss.item()

            mse = ((predictions - ratings) ** 2).mean().item()
            rmse = np.sqrt(mse)
            total_rmse += rmse * len(ratings)
            total_count += len(ratings)

    avg_loss = total_loss / len(test_loader)
    avg_rmse = total_rmse / total_count

    return avg_loss, avg_rmse


if __name__ == '__main__':
    # ===============================================================
    # Load config .yaml
    config_path = '../config/config.yaml'
    config = utils.load_config(config_path)
    print(config)

    embedding_dim = config['model_config']['embedding_size']
    batch_size = config['training_config']['batch_size']
    epochs = config['training_config']['num_epochs']
    learning_rate = config['model_config']['learning_rate']
    weight_decay = config['model_config']['weight_decay']
    dropout_rate = config['model_config']['dropout_rate']

    train_path = config['data_config']['train_path']
    test_path = config['data_config']['test_path']

    # ===============================================================
    # Load dataset
    train_dataset = dataset.MovieLensDataset(data_path=train_path)
    test_dataset = dataset.MovieLensDataset(data_path=test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_users = max(len(train_dataset.user_ids), len(test_dataset.user_ids))
    num_items = max(len(train_dataset.item_ids), len(test_dataset.item_ids))

    # ===============================================================
    # Prepare for training
    model = model.RecModel(num_users, num_items, embedding_dim,dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    utils.print_model_size(model)

    # ===============================================================
    # Training
    train_losses, eval_losses = [], []
    RMSE_list = []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        eval_loss, RMSE = evaluate(model, test_loader, criterion)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        RMSE_list.append(RMSE)

        print(f'Epoch {epoch + 1}/{epochs},\t Train Loss: {train_loss},\t Eval Loss: {eval_loss},\t RMSE: {RMSE}')

    print(f'Minimum RMSE:{np.min(RMSE_list)}')
    plt.plot(train_losses, label='Training loss')
    plt.plot(eval_losses, label='Evaluation loss')
    plt.plot(RMSE_list, label='Testing results of RMSE')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # todo: log中创建文件夹，存储loss图像，日志，模型，embedding可视化