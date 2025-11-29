import os

from utils import load_config, get_logger, set_seed

def init_dir(config):
    save_dir = os.path.join(config.save_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def main(args):
    config = load_config(args.config_path)
    save_dir = init_dir(config)
    logger = get_logger(
        experiment_name=config.exp_name,
        log_dir=save_dir,
    )
    set_seed(config.manual_seed)
    model = model_cls(config.model.params).to(config.device)
    trainer = SRTrainer(config, model, logger, save_dir)
    trainer.run()


if __name__ == "main":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/demo.yaml')
    args = parser.parse_args()
    main(args)