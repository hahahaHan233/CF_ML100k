import yaml
import torch
import numpy as np
import random
import datetime
import os
import sys

class Logger(object):
    def __init__(self,log_directory, filename=None):
        # If no filename provided, use the current date
        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if filename != None:
            filename = f'{date_str}_{filename}.log'
        else:
            filename = f'{date_str}.log'

        log_directory = log_directory
        if os.path.exists(log_directory) == False:
            os.mkdir(log_directory)

        self.terminal = sys.stdout
        self.log = open(os.path.join(log_directory, filename), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # Flush the output to console
        self.log.write(message)
        self.log.flush()  # Flush the output to file

    def flush(self):
        pass

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def initialize_device(use_gpu=True):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def save_config(config, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def print_model_size(model):
    total_bytes = 0
    print('------------------------------------------')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_size = parameter.numel()
            param_bytes = param_size * parameter.element_size()
            total_bytes += param_bytes
            print(f"{name}: {param_bytes / (1024 * 1024):.4f} MB")
    print('------------------------------------------')
    print(f"Total trainable parameters: {total_bytes / (1024 * 1024):.4f} MB")

import torch

# class EarlyStopping:
#     def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
#         """
#         Args:
#             patience (int): 等待次数
#             verbose (bool): 如果为True，在提高时打印一条消息
#             delta (float): 为了被认为是改善，监测的数量至少需要改变的最小量
#             path (str): 模型保存路径
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.delta = delta
#         self.path = path
#         self.best_score = None
#         self.epochs_no_improve = 0
#         self.early_stop = False
#
#     def __call__(self, eval_loss, model):
#         score = -eval_loss
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(model)
#         elif score < self.best_score + self.delta:
#             self.epochs_no_improve += 1
#             if self.epochs_no_improve >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.epochs_no_improve = 0
#             self.save_checkpoint(model)
#             if self.verbose:
#                 print(f'Validation loss decreased.  Saving model ...')
#
#     def save_checkpoint(self, model):
#         '''Saves model when validation loss decrease.'''
#         torch.save(model.state_dict(), self.path)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth', writer=None):
        """
        Args:
            patience (int): 等待次数
            verbose (bool): 如果为True，在提高时打印一条消息
            delta (float): 为了被认为是改善，监测的数量至少需要改变的最小量
            path (str): 模型保存路径
            writer (SummaryWriter, optional): TensorBoard SummaryWriter 对象
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.writer = writer
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model_params = None
        self.pre_training_logged = False  # 标记是否已经记录了训练前的嵌入

    def __call__(self, metric, model, epoch):
        score = -metric
        self.log_weights(model, epoch)
        if self.best_score is None:
            self.best_score = score
            self.best_model_params = model.state_dict()
            if not self.pre_training_logged:
                self.log_embeddings(model, epoch, tag='Embeddings Pre-training')
                self.pre_training_logged = True
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                self.save_checkpoint()
                self.log_embeddings(model, epoch, tag='Embeddings Post-training')
        else:
            self.best_score = score
            self.epochs_no_improve = 0
            self.best_model_params = model.state_dict()
            if self.verbose:
                print(f'Validation loss decreased. Updating best model parameters...')

    def save_checkpoint(self):
        '''Saves model when training stops with the best parameters.'''
        torch.save(self.best_model_params, self.path)
        if self.verbose:
            print(f'Saving model to {self.path} with the best parameters...')

    def log_embeddings(self, model, epoch, tag):
        '''Logs embeddings and model weights to TensorBoard if writer is provided.'''
        if self.writer:
            u_embeds = model.user_embeddings.weight.data.cpu().numpy()
            i_embeds = model.item_embeddings.weight.data.cpu().numpy()
            self.writer.add_embedding(np.vstack((u_embeds, i_embeds)), global_step=0,
                                 tag=tag)

    def log_weights(self, model, epoch):
        if self.writer:
            self.writer.add_histogram('Model Weights/User Embeddings', model.user_embeddings.weight, epoch)
            self.writer.add_histogram('Model Weights/Item Embeddings', model.item_embeddings.weight, epoch)
            self.writer.add_histogram('Model Weights/User Biases', model.user_biases.weight, epoch)
            self.writer.add_histogram('Model Weights/Item Biases', model.item_biases.weight, epoch)