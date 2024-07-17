from models.cvcp_model import CVCPModel, head
from models.cvt.encoder import CVTEncoder

import yaml
from pathlib import Path
from dataset import NuScenesDataModule
import sys
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor,  ModelSummary
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import faulthandler
def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main():
    faulthandler.enable()
    torch.cuda.empty_cache()

    default_config_path = Path(__file__) / 'configs/config.yaml'
    if len(sys.argv) > 1 and (config_path := sys.argv[1]):
        config = load_config(config_path)
    else:
        config = load_config(default_config_path)

    centerpoint_config = config['centerpoint']

    # Instantiate cvt_model and head_model
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

    datamodule = NuScenesDataModule(
        cvt_metadata_path=config['cvt_metadata_path'],
        bbox_label_path=config['bbox_label_path'],
        tasks=centerpoint_config['tasks'], 
        config=config)  # for data loading
    
    logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name='train'
    )
    
    logger.log_hyperparams(hyperparameters)
    hyperparameters = {
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'lr': config['lr'],
        'base_momentum': config['base_momentum'],
        'max_momentum': config['max_momentum'],
        'weight_decay': config['weight_decay'],
        'num_workers': config['num_workers'],
    }
    print(hyperparameters, '\n', f'logger: {logger.version},')
    print(f'log_dir: {logger.log_dir}')
    
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
    
    
    trainer.fit(
        datamodule=datamodule,
        model=model
    )
    
    print("\n=========================")
    print('Training completed.')


if __name__ == "__main__":
    main()
