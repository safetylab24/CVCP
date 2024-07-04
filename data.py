import json
import torch

from pathlib import Path

from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np

from torchvision.transforms import ToTensor

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split

    # gets a list of scene names from splits/nuscenes/[train/val].txt
    split_scenes = get_split(split)

    return [NuScenesGeneratedDataset(scene_name, labels_dir) for scene_name in split_scenes]

def parse_data(data:dict, dataset_dir:Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
    images = list()
    intrinsics = list()

    for image_path, I_original in zip(data['images'], data['intrinsics']):
        h_resize = h + top_crop
        w_resize = w

        image = Image.open(dataset_dir / image_path)

        image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
        image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

        I = np.float32(I_original)
        I[0, 0] *= w_resize / image.width
        I[0, 2] *= w_resize / image.width
        I[1, 1] *= h_resize / image.height
        I[1, 2] *= h_resize / image.height
        I[1, 2] -= top_crop

        images.append(img_transform(image_new))
        intrinsics.append(torch.tensor(I))

    return {
        'images': torch.stack(images, 0),
        'intrinsics': torch.stack(intrinsics, 0),
        'extrinsics': torch.tensor(np.float32(data['extrinsics'])),
        'labels': torch.zeros(1, 1, 1, 1)
    }

def get_split(version):
    if version == 'train':
        return (Path(__file__).parents[0] / 'splits/train.txt').read_text().splitlines()
    elif version == 'val':
        return (Path(__file__).parents[0] / 'splits/val.txt').read_text().splitlines()


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    ------ONE SCENE, MULTIPLE FRAMES-----
    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """

    def __init__(self, scene_name, labels_dir):
        self.samples = json.loads(
            (Path(labels_dir) / f'{scene_name}.json').read_text())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # represents individual frame data from JSON as a dictionary
        data = self.samples[idx]

        data = parse_data(data, dataset_dir=Path('/home/vrb230004/media/datasets/nuscenes'))

        return data
    
def get_train_dataloader(dataset_dir=None, labels_dir=None, version='v1.0-trainval', batch_size=1, num_workers=1, num_classes=2):
    datasets = get_data(
        dataset_dir=dataset_dir,
        labels_dir=dataset_dir,
        split='train',
        version=version,
        num_classes=num_classes,
    )
    data = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False
    )
    
    return dataloader


def get_val_dataloader(dataset_dir=None, labels_dir=None, version='v1.0-trainval', batch_size=1, num_workers=1, num_classes=2):
    datasets = get_data(
        dataset_dir=dataset_dir,
        labels_dir=dataset_dir,
        split='val',
        version=version,
        num_classes=num_classes,
    )

    data = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=False
    )
    
    return dataloader

