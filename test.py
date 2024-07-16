from CVCP.models.cvcp_model import CVCPModel, head
from models.cvt.encoder import CVTEncoder

import yaml
from pathlib import Path
from dataset import NuScenesDataModule
import sys
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor,  ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main():
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

    print("\n=========================")
    print('Data Loaders created. Starting training...')
    print("# Parameters:", sum(p.numel()
                               for p in model.parameters() if p.requires_grad))
    print("\n=========================")

    checkpointer = ModelCheckpoint(
        dirpath=config['log_dir'],
        filename='{epoch}-{step}-{val_loss:.2f}',
        verbose=True,
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=True,
        every_n_epochs=1,
        monitor='val_loss',
        enable_version_counter=True
    )
    
    lr_monitor = LearningRateMonitor(
        logging_interval='step', 
        log_momentum=True, 
        log_weight_decay=True)
    
    summarizer = ModelSummary(max_depth=1)

    trainer = L.Trainer(
        accelerator='gpu', 
        devices=config['devices'], 
        max_epochs=config['epochs'], 
        log_every_n_steps=50, 
        enable_progress_bar=True, 
        precision=32, 
        strategy='ddp',
        default_root_dir=config['log_dir'],
        callbacks=[checkpointer, lr_monitor, summarizer])
    
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=config['test'].get('ckpt_path', None)
    )

    print("\n=========================")
    print('Training completed.')


if __name__ == "__main__":
    main()