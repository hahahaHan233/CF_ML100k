import yaml
import torch
import numpy as np
import random
import datetime
import os
import sys

class Logger(object):
    def __init__(self, filename=None):
        # If no filename provided, use the current date
        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if filename != None:
            filename = f'{date_str}_{filename}.log'
        else:
            filename = f'{date_str}.log'

        log_directory = '../log'
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
            print(f"{name}: {param_bytes / (1024 * 1024)} MB")
    print('------------------------------------------')
    print(f"Total trainable parameters: {total_bytes / (1024 * 1024):.4f} MB ")