import json
from easydict import EasyDict
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return  config_dict


def process_config(json_file, hyper_params={}):
    config_dict = get_config_from_json(json_file)
    # Update config file by command args
    config_dict.update(hyper_params)
    config = EasyDict(config_dict)
    config.summary_dir = os.path.join("../experiments", config.model_name, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.model_name, "checkpoint/")
    return config


if __name__ == "__main__":
    a, b = process_config("../configs/config.json", {"111111111": 111123})
    print(a)