import yaml
import os
from types import SimpleNamespace


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist:{config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML format error:{str(e)}") from e
    
    config = _dict_to_namespace(config_dict)
    
    return config


def _dict_to_namespace(config_dict):
    namespace = SimpleNamespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(namespace, key, _dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

