from models.model import CVCPModel, cvt, head
import torch
import torch.nn as nn
import yaml

from CVCP.dataset import get_train_dataloader, get_val_dataloader
from tqdm import tqdm

import sys


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main():
    default_config_path = '/home/vrb230004/CombinedModels/CVCP/configs/config.yaml' # changed this cuz i couldnt figure out how to run without CDing into CombinedModels
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
        n_classes=cvt_config.get('n_classes'),
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

    # Define a loss function and an optimizer
    # model.loss = nn.CrossEntropyLoss()
    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=opt_config.get('lr'),
        weight_decay=opt_config.get('weight_decay'))

    # Define dataloaders
    train_loader = get_train_dataloader(
        dataset_dir=config.get('dataset_dir'),
        cvt_metadata_path=config.get('cvt_metadata_path'),
        bbox_label_path=config.get('bbox_label_path'),
        num_classes=config.get('n_classes'),
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers'),
        tasks=centerpoint_config.get('tasks'))

    val_loader = get_val_dataloader(
        dataset_dir=config.get('dataset_dir'),
        cvt_metadata_path=config.get('cvt_metadata_path'),
        bbox_label_path=config.get('bbox_label_path'),        
        num_classes=config.get('n_classes'),
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers'),
        tasks=centerpoint_config.get('tasks'))

    print("\n=========================")
    print('Data Loaders created. Starting training...')
    print("# Parameters:", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    print(f"Train length: {len(train_loader)}")
    print(f"Val length: {len(val_loader)}")
    print("\n=========================")

    step = 0
    best = 1e10
    for epoch in range(config.get('epochs')):
        # training
        for batch in tqdm(train_loader):
            loss = model.train_step(batch)

            step += 1

            if step % 5000 == 0:
                print(f"Train Loss: {loss}")

        # validation
        for batch in tqdm(val_loader):
            loss = model.val_step(batch)

            step += 1

            if step % 1000 == 0:
                print(f"Val Loss: {loss}")

            if loss < best:
                best = loss
            
        ckpt = {
            'model': model.state_dict(),
            'opt': model.opt.state_dict(),
            'epoch': epoch
        }
        torch.save(ckpt, f"/home/vrb230004/CombinedModels/CVCP/out_dir/cvcp_ckpt_{epoch}_{best:.4f}.pth")


if __name__ == "__main__":
    main()
