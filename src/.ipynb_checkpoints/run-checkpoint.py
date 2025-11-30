import os
import shutil
from utils import load_config, get_logger, set_seed
from data import Data
from models import Model
from runner import Trainer

def init_dir(config, reset=False):
    save_dir = os.path.join(config.save_dir, config.exp_name)
    print(reset)
    
    if reset and os.path.exists(save_dir):
        try:
            shutil.rmtree(save_dir)
        except Exception as e:
            raise RuntimeError(f"faile to delete：{e}") from e
    
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def main(args):
    config = load_config(args.config)
    save_dir = init_dir(config, args.reset)
    logger = get_logger(
        experiment_name=config.exp_name,
        log_dir=save_dir,
    )
    set_seed(config.manual_seed)
    data = Data(config)
    model = Model(config, logger)
    trainer = Trainer(config, model, data, logger, save_dir)
    trainer.run()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/edsr/edsr_x2_div2k_L1.yaml')
parser.add_argument('--reset', action='store_true', default=False)
args = parser.parse_args()
main(args)