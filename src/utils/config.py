import yaml
import os
import logging
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace
from datetime import datetime


def get_logger(
    experiment_name: str,
    log_dir: str = "./experiment/logs",
    level: int = logging.INFO,
) -> logging.Logger:

    exp_log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    log_filename = f"{experiment_name}.log"
    log_filepath = os.path.join(exp_log_dir, log_filename)

    file_handler = logging.FileHandler(
        filename=log_filepath,
        mode='a',
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file does not exist:{config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML format error:{str(e)}") from e
    
    config = _dict_to_namespace(config_dict)
    logger = get_logger(
        experiment_name=config.experiment_name,
        log_dir=config.log.dir,
        level=logging.INFO
    )
    config.logger = logger
    
    return config


def _dict_to_namespace(config_dict):
    namespace = SimpleNamespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(namespace, key, _dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

