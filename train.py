from models.model import CVCPModel, CVCPModelL, cvt, head
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from dataset import get_train_dataloader, get_val_dataloader, NuScenesDataModule
from tqdm import tqdm
import sys
import lightning as L
from tensorboardX import SummaryWriter


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main():
    default_config_path = Path(__file__) / 'configs/config.yaml'
    if len(sys.argv) > 1 and (config_path := sys.argv[1]):
        config = load_config(config_path)
    else:
        config = load_config(default_config_path)

    cvt_config = config.get('cvt')
    centerpoint_config = config.get('centerpoint')
    opt_config = config.get('optimizer')

    # Instantiate cvt_model and head_model
    cvt_seg = cvt(
        devices=cvt_config.get('devices'),
        loss_type=cvt_config.get('loss_type'))

    head_seg = head(
        in_channels=centerpoint_config.get('in_channels'),
        tasks=centerpoint_config.get('tasks'),
        dataset=centerpoint_config.get('dataset'),
        weight=centerpoint_config.get('weight'),
        code_weights=centerpoint_config.get('code_weights'),
        common_heads=centerpoint_config.get('common_heads'),
        share_conv_channel=centerpoint_config.get('share_conv_channel'),
        dcn_head=centerpoint_config.get('dcn_head')
    )

    # Define the resize shape for images before being passed into the CenterHead
    resize_shape = tuple(config.get('resize_shape'))

    # Instantiate the combined model
    model = CVCPModel(cvt_seg, head_seg, resize_shape)

    if isinstance(model, CVCPModel):        
        # Define dataloaders
        train_loader = get_train_dataloader(
            dataset_dir=config.get('dataset_dir'),
            cvt_metadata_path=config.get('cvt_metadata_path'),
            bbox_label_path=config.get('bbox_label_path'),
            batch_size=config.get('batch_size'),
            num_workers=config.get('num_workers'),
            tasks=centerpoint_config.get('tasks'))

        val_loader = get_val_dataloader(
            dataset_dir=config.get('dataset_dir'),
            cvt_metadata_path=config.get('cvt_metadata_path'),
            bbox_label_path=config.get('bbox_label_path'),        
            batch_size=config.get('batch_size'),
            num_workers=config.get('num_workers'),
            tasks=centerpoint_config.get('tasks'))
        
        # Define an optimizer/scheduler
        model.opt = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config.get('lr'),
            weight_decay=opt_config.get('weight_decay'))
        model.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.opt, 
            max_lr=opt_config.get('lr'), 
            steps_per_epoch=len(train_loader), 
            epochs=config.get('epochs'), 
            base_momentum=0.85, 
            max_momentum=0.95)
        
        step = 0
        
        writer = SummaryWriter(logdir="/home/vrb230004/CombinedModels/CVCP/out_dir")
        
        for epoch in range(config.get('epochs')):
            
            writer.add_scalar('train/epoch', epoch, step)
            
            # training
            for batch in tqdm(train_loader):
                loss = model.train_step(batch)

                step += 1

                if step % 100 == 0:
                    print(f"Train Loss: {loss}")

            writer.add_scalar('epoch/loss', loss, epoch)
            
            # validation
            for batch in tqdm(val_loader):
                loss = model.val_step(batch)

                step += 1

                if step % 100 == 0:
                    print(f"Val Loss: {loss}")
                
            ckpt = {
                'model': model.state_dict(),
                'opt': model.opt.state_dict(),
                'epoch': epoch
            }
            torch.save(ckpt, f"/home/vrb230004/CombinedModels/CVCP/out_dir/cvcp_ckpt_{epoch}.pth")

    elif isinstance(model, CVCPModelL):
        model.configure_optimizers()
        datamodule = NuScenesDataModule(
            cvt_metadata_path=config.get('cvt_metadata_path'), 
            bbox_label_path=config.get('bbox_label_path'), 
            tasks=centerpoint_config.get('tasks'))
        print("\n=========================")
        print('Data Loaders created. Starting training...')
        print("# Parameters:", sum(p.numel()
            for p in model.parameters() if p.requires_grad))
        print("\n=========================")
        trainer = L.Trainer(accelerator='gpu', devices=cvt_config.get('devices'), max_epochs=config.get('epochs'), log_every_n_steps=100, enable_progress_bar=True, precision=64)
        trainer.fit(
            model=model,
            datamodule=datamodule
            )

    
    print("\n=========================")
    print('Training completed.')
if __name__ == "__main__":
    main()
