
import argparse
from models.cvcp_model import CVCPModel, head
from models.cvt.encoder import CVTEncoder
import yaml
from pathlib import Path
from data.datamodule import NuScenesDataModule
import sys
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import faulthandler
from colorama import Fore, Style

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    
def init_config():
    default_config_path = Path(
        __file__).parents[0] / 'configs/config_train.yaml'
    try:
        config = load_config(sys.argv[1])
    except IndexError:
        config = load_config(default_config_path)
        
    return config    
    
def init_hyperparameters(config):
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters.')

    # Add arguments for each hyperparameter
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=False, default=config['epochs'])
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=config['batch_size'])
    parser.add_argument('--lr', type=float, help='Learning rate', required=False, default=config['lr'])
    parser.add_argument('--base_momentum', type=float, help='Base momentum', required=False, default=config['base_momentum'])
    parser.add_argument('--max_momentum', type=float, help='Maximum momentum', required=False, default=config['max_momentum'])
    parser.add_argument('--weight_decay', type=float, help='Weight decay', required=False, default=config['weight_decay'])
    parser.add_argument('--num_workers', type=int, help='Number of workers', required=False, default=config['num_workers'])

    # Parse the arguments
    args = parser.parse_args()

    # Create the hyperparameters dictionary using the parsed arguments
    hyperparameters = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'base_momentum': args.base_momentum,
        'max_momentum': args.max_momentum,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
    }
    
    return hyperparameters    

def init_model(config, centerpoint_config, hyperparameters):
    cvt_encoder = CVTEncoder()

    head_seg = head(
        in_channels=centerpoint_config['in_channels'],
        tasks=centerpoint_config['tasks'],
        dataset=centerpoint_config['dataset'],
        weight=centerpoint_config['weight'],
        code_weights=centerpoint_config['code_weights'],
        common_heads=centerpoint_config['common_heads'],
        share_conv_channel=centerpoint_config['share_conv_channel'],
        dcn_head=centerpoint_config['dcn_head']
    )

    # Define the resize shape for images before being passed into the CenterHead
    resize_shape = tuple(config['resize_shape'])

    # Instantiate the combined model
    model = CVCPModel(cvt_encoder, head_seg, resize_shape, config)
    
    return model

def main():
    faulthandler.enable()
    torch.cuda.empty_cache()

    config = init_config()
    centerpoint_config = config['centerpoint']
    
    hyperparameters = init_hyperparameters(config)
    
    model = init_model(config, centerpoint_config, hyperparameters)

    datamodule = NuScenesDataModule(
        nuscenes_metadata_path=config['nuscenes_metadata_path'],
        bbox_label_path=config['bbox_label_path'],
        tasks=centerpoint_config['tasks'],
        config=config)  # for data loading
    
    logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name='train'
    )
    
    logger.log_hyperparams(hyperparameters)

    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
        log_weight_decay=True)

    checkpointer = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / 'checkpoints',
        filename='{epoch}-{step}',
        verbose=True,
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=True,
        every_n_epochs=1,
    )

    trainer = Trainer(
        accelerator='gpu',
        devices=config['devices'],
        max_epochs=config['epochs'],
        strategy=DDPStrategy(),
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        callbacks=[checkpointer, lr_monitor],
        num_sanity_val_steps=config['num_sanity_val_steps'],
        limit_train_batches=config['limit_train_batches'],
        limit_val_batches=config['limit_val_batches'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        val_check_interval=config['val_check_interval'],
    )

    print(Fore.GREEN + 'Training started.')
    print('Hyperparameters:', hyperparameters)
    print('Logger version:', logger.version)
    print('Log dir:', logger.log_dir)
    print(Style.RESET_ALL)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.get('ckpt_path', None)
    )
    
    print('\n=========================')
    print(Fore.RED + 'Training completed.')
    print(Style.RESET_ALL)
    
if __name__ == '__main__':
    main()
    



