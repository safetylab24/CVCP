from pathlib import Path
import lightning as L
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader, ConcatDataset

from data.nuscenes_dataset import get_dataset


class NuScenesDataModule(L.LightningDataModule):
    def __init__(self, nuscenes_metadata_path, bbox_label_path, tasks, config):
        super().__init__()
        self.dataset_dir = Path('/home/vrb230004/media/datasets/nuscenes')
        self.nuscenes_metadata_path = Path(nuscenes_metadata_path)
        self.bbox_label_path = Path(bbox_label_path)
        self.tasks: dict = tasks
        self.config = config

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.data_train: ConcatDataset = get_dataset(
                dataset_dir=self.dataset_dir,
                nuscenes_metadata_path=self.nuscenes_metadata_path,
                bbox_label_path=self.bbox_label_path,
                split='train',
                version=self.config.get('version', 'v1.0-mini'),
                tasks=self.tasks
            )
            self.data_val: ConcatDataset = get_dataset(
                dataset_dir=self.dataset_dir,
                nuscenes_metadata_path=self.nuscenes_metadata_path,
                bbox_label_path=self.bbox_label_path,
                split='val',
                version=self.config.get('version', 'v1.0-mini'),
                tasks=self.tasks,
                return_original_label=True
            )
        elif stage == 'test' or stage == 'predict':
            self.data_test: ConcatDataset = get_dataset(
                dataset_dir=self.dataset_dir,
                nuscenes_metadata_path=self.nuscenes_metadata_path,
                bbox_label_path=self.bbox_label_path,
                split='val',
                version=self.config.get('version', 'v1.0-mini'),
                tasks=self.tasks,
                return_original_label=True
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
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
