from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler, default_collate

from data.nuscenes_dataset import get_dataset
import torch
import numpy as np


class NuScenesDataModule(L.LightningDataModule):
    def __init__(self, nuscenes_metadata_path, bbox_label_path, tasks, config, dataset_dir):
        """
        Data module for NuScenes dataset.

        Args:
            nuscenes_metadata_path (str): Path to the NuScenes metadata file.
            bbox_label_path (str): Path to the bounding box label file.
            tasks (dict): Dictionary specifying the tasks to be performed.
            config (dict): Configuration parameters for the data module.
            dataset_dir (str): Directory path where the dataset is located.
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.nuscenes_metadata_path = Path(nuscenes_metadata_path)
        self.bbox_label_path = Path(bbox_label_path)
        self.tasks: dict = tasks
        self.config = config

    def setup(self, stage: str) -> None:
        """
        Setup method to prepare the data for training, validation, or testing.

        Args:
            stage (str): The stage of the data module (fit, test, or predict).
        """
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

    @staticmethod
    def pad_tensor_list(tensor_list, pad_value=0):
        """
        Pads a list of tensors to the same shape.

        Args:
            tensor_list (list): List of tensors to be padded.
            pad_value (int): Value to be used for padding.

        Returns:
            torch.Tensor: Padded tensor list.
        """
        max_shape = np.array(
            [tensor.shape for tensor in tensor_list]).max(axis=0)
        padded_tensors = []
        for tensor in tensor_list:
            pad_size = [(0, max_shape[i] - tensor.shape[i])
                        for i in range(len(max_shape))]
            padded_tensor = torch.nn.functional.pad(tensor, pad=[
                                                    p for sublist in pad_size[::-1] for p in sublist], mode='constant', value=pad_value)
            padded_tensors.append(padded_tensor)
        return torch.stack(padded_tensors, dim=0)

    @staticmethod
    def custom_collate_fn(batch):
        """
        Custom collate function for creating batches.

        Args:
            batch (list): List of samples to be collated.

        Returns:
            dict: Collated batch.
        """
        # Separate 'labels_original' from the rest of the batch
        batch_no_labels_original = []
        labels_original_list = []

        for item in batch:
            item_no_labels_original = {
                key: value for key, value in item.items() if key != 'labels_original'}
            labels_original_list.append(item['labels_original'])
            batch_no_labels_original.append(item_no_labels_original)

        # Use default_collate to collate everything except 'labels_original'
        collated_batch_no_labels_original = default_collate(
            batch_no_labels_original)

        # Handle 'labels_original' manually
        gt_boxes_with_classes = []
        for labels in labels_original_list:
            gt_boxes = torch.tensor(labels['gt_boxes'])
            gt_classes = torch.tensor(labels['gt_classes']).unsqueeze(
                1)  # add a dimension for concatenation
            gt_boxes_with_class = torch.cat((gt_boxes, gt_classes), dim=1)
            gt_boxes_with_classes.append(gt_boxes_with_class)

        # Pad the tensors to the same shape
        padded_gt_boxes_with_classes = NuScenesDataModule.pad_tensor_list(
            gt_boxes_with_classes)

        # Combine everything into one dictionary
        collated_batch = collated_batch_no_labels_original
        collated_batch['labels']['gt_boxes_and_cls'] = padded_gt_boxes_with_classes

        return collated_batch

    def train_dataloader(self):
        """
        Returns the data loader for training.

        Returns:
            torch.utils.data.DataLoader: Data loader for training.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        """
        Returns the data loader for validation.

        Returns:
            torch.utils.data.DataLoader: Data loader for validation.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            sampler=SequentialSampler(self.data_val),
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.custom_collate_fn
        )

    def test_dataloader(self):
        """
        Returns the data loader for testing.

        Returns:
            torch.utils.data.DataLoader: Data loader for testing.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.config['num_workers'],
            sampler=SequentialSampler(self.data_test),
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.custom_collate_fn
        )
