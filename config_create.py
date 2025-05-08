import configparser
import os

def create_config_file(config_path, root_dir):
    """
    Creates a configuration file with a 'DEFAULT' section specifying the dataset root directory.

    Args:
        config_path (str): The path to save the configuration file.
        root_dir (str): The root directory of the dataset.
    """
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'root_dir': root_dir
    }

    with open(config_path, 'w') as configfile:
        config.write(configfile)

# Example usage:
config_path = 'config.ini'
root_dir = '/home/leonjye/Data'  # Replace with your actual dataset root directory
create_config_file(config_path, root_dir)


# 讀取config.ini的範例程式，並取得root_dir的值
config = configparser.ConfigParser()
config.read('config.ini')
root_dir = config.get('DEFAULT', 'root_dir')
print(root_dir)

# %%


