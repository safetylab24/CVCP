from model import CVCPModel, cvt, head
import torch
import torch.nn as nn
import yaml

from CVCP.data import get_train_dataloader, get_val_dataloader
from tqdm import tqdm

import sys

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main():
    default_config_path = 'config.yaml'
    if len(sys.argv) > 1 and (config_path := sys.argv[1]):
        config = load_config(config_path)
    else:
        config = load_config(default_config_path)
    
    # Instantiate cvt_model and head_model
    cvt_seg = cvt(
        devices=config.get('devices'), 
        n_classes=config.get('n_classes'), 
        loss_type=config.get('loss_type'))
    
    #TODO: load configs for creating the model segments
    head_seg = head()

    # Define the resize shape for images before being passed into the CenterHead
    resize_shape = tuple(config.get('resize_shape'))

    # Instantiate the combined model
    model = CVCPModel(cvt_seg, head_seg, resize_shape)

    # Define a loss function and an optimizer
    model.loss = nn.CrossEntropyLoss()
    model.opt = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('lr'), 
        weight_decay=config.get('weight_decay'))

    train_loader = get_train_dataloader(
        dataset_dir=config.get('dataset_dir'), 
        labels_dir=config.get('labels_dir'),
        num_classes=config.get('n_classes'),
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers'))
    
    val_loader = get_val_dataloader(
        dataset_dir=config.get('dataset_dir'), 
        labels_dir=config.get('labels_dir'),
        num_classes=config.get('n_classes'),
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers'))

    step = 0

    print("\n=========================")
    print('Data Loaders created. Starting training...')
    print("# Parameters:", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    print(f"Train length: {len(train_loader)}")
    print(f"Val length: {len(val_loader)}")

    for epoch in range(config.get('epochs')):
        # training
        for batch in tqdm(train_loader):
            loss = model.train_step(batch)

            step += 1

            if step % 10 == 0:
                print(f"Train Loss: {loss}")

        # validation
        for batch in tqdm(val_loader):
            loss = model.val_step(batch)

            step += 1

            if step % 10 == 0:
                print(f"Val Loss: {loss}")


if __name__ == "__main__":
    main()
