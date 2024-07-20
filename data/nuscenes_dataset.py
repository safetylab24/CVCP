import pickle
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

from transforms.loading import load_annotations
from transforms.preprocess import preprocess, assign_label


def get_dataset(
    dataset_dir,
    nuscenes_metadata_path,
    bbox_label_path,
    split,
    version='v1.0-mini',
    tasks=None,
    return_original_label=False
) -> ConcatDataset:
    nuscenes_metadata_path = Path(nuscenes_metadata_path)
    bbox_label_path = Path(bbox_label_path)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split

    # gets a list of scene names from splits/nuscenes/[train/val].txt
    split_scenes = (
        Path(__file__).parents[0] / f'splits/{split}.txt').read_text().splitlines()
    return ConcatDataset([NuScenesGeneratedDataset(
        scene_name=scene_name,
        nuscenes_metadata_path=nuscenes_metadata_path,
        bbox_label_path=bbox_label_path,
        dataset_dir=dataset_dir,
        tasks=tasks,
        return_original_label=return_original_label,
        version=split)
        for scene_name in split_scenes])


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    A dataset class for NuScenes generated data.

    Args:
        scene_name (str): The name of the scene.
        nuscenes_metadata_path (Path): The path to the NuScenes metadata.
        bbox_label_path (Path): The path to the bounding box labels.
        dataset_dir (Path): The directory of the dataset.
        tasks (dict): The tasks to be performed on the dataset.
        return_original_label (bool, optional): Whether to return the original labels. Defaults to False.
        version (str, optional): The version of the dataset. Defaults to "train".
    """

    def __init__(
            self,
            scene_name: str,
            nuscenes_metadata_path: Path,
            bbox_label_path: Path,
            dataset_dir: Path,
            tasks: dict,
            return_original_label=False,
            version="train"):
        """
        Initializes the NuScenesGeneratedDataset.

        Args:
            scene_name (str): The name of the scene.
            nuscenes_metadata_path (Path): The path to the NuScenes metadata.
            bbox_label_path (Path): The path to the bounding box labels.
            dataset_dir (Path): The directory of the dataset.
            tasks (dict): The tasks to be performed on the dataset.
            return_original_label (bool, optional): Whether to return the original labels. Defaults to False.
            version (str, optional): The version of the dataset. Defaults to "train".
        """

        self.scene_name = scene_name
        self.tasks = tasks
        self.dataset_dir = dataset_dir
        self.return_original_label = return_original_label
        self.version = version

        # metadata
        with open((Path(nuscenes_metadata_path) / f'{scene_name}.pkl'), 'rb') as f:
            self.samples = pickle.load(f)
        # labels
        with open((Path(bbox_label_path) / f'{scene_name}.pkl'), 'rb') as f:
            self.labels = pickle.load(f)

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
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: The parsed data.
        """
        inputs = self.samples[idx]
        labels = self.labels[idx]
        return self.parse_data(inputs, labels, self.tasks, dataset_dir=Path(self.dataset_dir))

    def parse_inputs(self, inputs: dict, dataset_dir: Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
        """
        Parses the inputs.

        Args:
            inputs (dict): The input data.
            dataset_dir (Path): The directory of the dataset.
            h (int, optional): The height of the resized image. Defaults to 224.
            w (int, optional): The width of the resized image. Defaults to 480.
            top_crop (int, optional): The amount to crop from the top of the image. Defaults to 24.
            img_transform (torchvision.transforms, optional): The image transformation. Defaults to ToTensor().

        Returns:
            tuple: A tuple containing the parsed images and intrinsics.
        """
        images = list()
        intrinsics = list()

        for image_path, I_original in zip(inputs['images'], inputs['intrinsics']):
            h_resize = h + top_crop
            w_resize = w
            with Image.open(dataset_dir / image_path) as image:
                image_new = image.resize(
                    (w_resize, h_resize), resample=Image.BILINEAR)
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
        """
        Parses the labels.

        Args:
            labels (dict): The label data.
            tasks (dict): The tasks to be performed on the dataset.

        Returns:
            dict: The parsed labels.
        """
        loaded = load_annotations(labels)
        preprocessed = preprocess(loaded, tasks)
        labels_out = assign_label(preprocessed, tasks)
        return labels_out

    def parse_data(self, inputs: dict, labels: dict, tasks, dataset_dir: Path, h=224, w=480, top_crop=24, img_transform=ToTensor()) -> dict:
        """
        Parses the data.

        Args:
            inputs (dict): The input data.
            labels (dict): The label data.
            tasks (dict): The tasks to be performed on the dataset.
            dataset_dir (Path): The directory of the dataset.
            h (int, optional): The height of the resized image. Defaults to 224.
            w (int, optional): The width of the resized image. Defaults to 480.
            top_crop (int, optional): The amount to crop from the top of the image. Defaults to 24.
            img_transform (torchvision.transforms, optional): The image transformation. Defaults to ToTensor().

        Returns:
            dict: The parsed data.
        """
        images, intrinsics = self.parse_inputs(
            inputs, dataset_dir, h, w, top_crop, img_transform)
        labels_out = self.parse_labels(labels, tasks)

        data = {
            'images': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(inputs['extrinsics'])),
            'labels': labels_out,
        }
        # if self.return_original_label:
        orig_label = preprocess(load_annotations(labels), tasks)['annotations']
        del orig_label['gt_names']
        # TODO: add custom collate_fn so that this is loaded correctly
        data['labels_original'] = orig_label

        return data
