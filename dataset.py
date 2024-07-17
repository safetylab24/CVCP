import json
import torch
from pathlib import Path
import pickle
import lightning as L
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import numpy as np

from torchvision.transforms import ToTensor

from transforms.loading import load_annotations
from transforms.preprocess import preprocess, assign_label


class NuScenesDataModule(L.LightningDataModule):
    def __init__(self, cvt_metadata_path, bbox_label_path, tasks, config):
        super().__init__()
        self.dataset_dir = Path('/home/vrb230004/media/datasets/nuscenes')
        self.cvt_metadata_path = Path(cvt_metadata_path)
        self.bbox_label_path = Path(bbox_label_path)
        self.tasks: dict = tasks
        self.config = config
    
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.data_train: ConcatDataset = get_dataset(
                dataset_dir=self.dataset_dir,
                cvt_metadata_path=self.cvt_metadata_path,
                bbox_label_path=self.bbox_label_path,
                split='train',
                version=self.config.get('version', 'v1.0-mini'),
                tasks=self.tasks
            )
            self.data_val : ConcatDataset = get_dataset(
                dataset_dir=self.dataset_dir,
                cvt_metadata_path=self.cvt_metadata_path,
                bbox_label_path=self.bbox_label_path,
                split='val',
                version=self.config.get('version', 'v1.0-mini'),
                tasks=self.tasks
            )
        elif stage == "test" or stage == "predict":
            self.data_test : ConcatDataset = get_dataset(
                dataset_dir=self.dataset_dir,
                cvt_metadata_path=self.cvt_metadata_path,
                bbox_label_path=self.bbox_label_path,
                split='val',
                version=self.config.get('version', 'v1.0-mini'),
                tasks=self.tasks
            )
        
    def train_dataloader(self):
        return DataLoader(
        dataset=self.data_train,
        batch_size=self.config['batch_size'],
        num_workers=self.config['num_workers'],
        # sampler=SequentialSampler(self.data_train),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
        
    def val_dataloader(self):
        return DataLoader(
        dataset=self.data_val,
        batch_size=self.config['batch_size'],
        num_workers=self.config['num_workers'],
        sampler=SequentialSampler(self.data_val),
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
        
    def test_dataloader(self):
        return DataLoader(
        dataset=self.data_test,
        batch_size=1,
        num_workers=self.config['num_workers'],
        sampler=SequentialSampler(self.data_test),
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

def get_dataset(
    dataset_dir,
    cvt_metadata_path,
    bbox_label_path,
    split,
    version='v1.0-mini',
    tasks=None
) -> ConcatDataset:
    dataset_dir = Path(dataset_dir)
    cvt_metadata_path = Path(cvt_metadata_path)
    bbox_label_path = Path(bbox_label_path)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split

    # gets a list of scene names from splits/nuscenes/[train/val].txt
    split_scenes = (Path(__file__).parents[0] / f'splits/{split}.txt').read_text().splitlines()
    return ConcatDataset([NuScenesGeneratedDataset(scene_name, cvt_metadata_path, bbox_label_path, tasks) for scene_name in split_scenes])

class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, scene_name, cvt_metadata_path, bbox_label_path, tasks):
        self.scene_name = scene_name
        self.samples = json.loads((Path(cvt_metadata_path) / f'{scene_name}.json').read_text())
        with open((Path(bbox_label_path) / f'{scene_name}.pkl'), 'rb') as f:
            self.labels = pickle.load(f)
        self.tasks = tasks
        
        # remove any samples that have no labels
        if len(self.samples) != len(self.labels):
            for i in range(len(self.samples) - 1, -1, -1):
                token = self.samples[i]['token']
                token_in_labels = False
                for label in self.labels:
                    # if the token is not in the labels, remove the sample
                    if label['token'] == token:
                        token_in_labels = True
                if token_in_labels == False:
                    del self.samples[i]
                
            
        assert len(self.samples) == len(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs = self.samples[idx]
        labels = self.labels[idx]
        return self.parse_data(inputs, labels, self.tasks, dataset_dir=Path('/home/vrb230004/media/datasets/nuscenes'))
        
    def parse_inputs(self, inputs: dict, dataset_dir: Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
        images = list()
        intrinsics = list()
        
        for image_path, I_original in zip(inputs['images'], inputs['intrinsics']):
            h_resize = h + top_crop
            w_resize = w
            with Image.open(dataset_dir / image_path) as image:
                image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
                image_new = image_new.crop(
                    (0, top_crop, image_new.width, image_new.height))

                I = np.float32(I_original)
                I[0, 0] *= w_resize / image.width
                I[0, 2] *= w_resize / image.width
                I[1, 1] *= h_resize / image.height
                I[1, 2] *= h_resize / image.height
                I[1, 2] -= top_crop

                images.append(img_transform(image_new))
                intrinsics.append(torch.tensor(I))
        
        return images, intrinsics

    def parse_labels(self, labels: dict, tasks) -> dict:
        loaded = load_annotations(labels)
        preprocessed = preprocess(loaded, 'train', tasks)
        labels_out = assign_label(preprocessed, tasks)
        return labels_out 

    def parse_data(self, inputs: dict, labels: dict, tasks, dataset_dir: Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
            images, intrinsics = self.parse_inputs(inputs, dataset_dir, h, w, top_crop, img_transform)
            labels_out = self.parse_labels(labels, tasks)

            return {
                'images': torch.stack(images, 0),
                'intrinsics': torch.stack(intrinsics, 0),
                'extrinsics': torch.tensor(np.float32(inputs['extrinsics'])),
                'labels': labels_out,
            }