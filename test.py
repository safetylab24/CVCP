from models.cvcp_model import CVCPModel, head
from models.cvt.encoder import CVTEncoder
from models.model_module import CVCPModule

import yaml
from pathlib import Path
from data.datamodule import NuScenesDataModule
import sys
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from colorama import Fore, Style


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main():
    
    """
    Main function for training the CVCP model.

    This function performs the following steps:
    1. Enables the faulthandler and clears the CUDA cache.
    2. Loads the configuration from the command line argument or the default configuration file.
    3. Instantiates the CVT encoder and the head segmentation model.
    4. Defines the resize shape for images.
    5. Instantiates the combined CVCP model.
    6. Sets up the data module for NuScenes dataset.
    7. Sets up the logger for TensorBoard.
    8. Logs the hyperparameters.
    9. Sets up the learning rate monitor and model checkpoint.
    10. Sets up the trainer with GPU acceleration and distributed training.
    11. Prints training information.
    12. Starts the training process.
    13. Prints training completion message.
    """
    # faulthandler.enable()
    # torch.cuda.empty_cache()

    default_config_path = Path(
        __file__).parents[0] / 'configs/config_train.yaml'
    try:
        config = load_config(sys.argv[1])
    except IndexError:
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
    
    modelmodule = CVCPModule(model, config)

    datamodule = NuScenesDataModule(
        nuscenes_metadata_path=config['nuscenes_metadata_path'],
        bbox_label_path=config['bbox_label_path'],
        tasks=centerpoint_config['tasks'],
        config=config,
        dataset_dir=config['dataset_dir'])  # for data loading

    logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name='train',
        default_hp_metric=False,
    )

    hyperparameters = {
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'lr': config['lr'],
        'base_momentum': config['base_momentum'],
        'max_momentum': config['max_momentum'],
        'weight_decay': config['weight_decay'],
        'num_workers': config['num_workers'],
    }

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
        devices=1,
        num_nodes=1,
        strategy=DDPStrategy(),
        logger=logger,
        callbacks=[checkpointer, lr_monitor],
        limit_test_batches=config['limit_test_batches']
    )

    print(Fore.GREEN + 'Testing started.')
    print('Hyperparameters:', hyperparameters)
    print('Logger version:', logger.version)
    print('Log dir:', logger.log_dir)
    print(Style.RESET_ALL)

    trainer.test(
        model=modelmodule,
        datamodule=datamodule,
        ckpt_path=config.get('ckpt_path', None)
    )

    print('\n=========================')
    print(Fore.RED + 'Testing completed.')
    print(Style.RESET_ALL)


if __name__ == '__main__':
    main()
