import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def save_config(config, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


# 使用配置文件
config_path = '../config/config.yaml'
config = load_config(config_path)
print(config)

# 修改配置
config['model_config']['embedding_size'] = 100

# 保存新的配置
save_config(config, config_path)