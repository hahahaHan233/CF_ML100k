import torch

import utils
import model
import dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.setup_seed(42)

if __name__ == '__main__':
    config_path = '../config/config.yaml'
    config = utils.load_config(config_path)
    print(config)

    # 读取配置项
    embedding_dim = config['data_config']['embedding_dim']
    batch_size = config['training_config']['batch_size']
    learning_rate = config['model_config']['learning_rate']

    #save_config(config, config_path)