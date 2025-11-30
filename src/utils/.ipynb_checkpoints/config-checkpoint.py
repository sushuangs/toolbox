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
    
    config = dict_to_namespace(config_dict)
    
    return config


def dict_to_namespace(obj):
    if isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]
    elif isinstance(obj, dict):
        namespace = SimpleNamespace()
        for key, value in obj.items():
            setattr(namespace, key, dict_to_namespace(value))
        return namespace
    else:
        return obj


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = namespace_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [namespace_to_dict(item) for item in obj]
    else:
        return obj

